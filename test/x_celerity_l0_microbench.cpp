// celerity_l0_microbench.cpp
// SYCL micro-benchmark for H2D/D2H/D2D copies across sizes and modes.
// Build:  dpcpp -O3 -std=c++17 celerity_l0_microbench.cpp -o l0bench
// Run:    SYCL_DEVICE_FILTER=level_zero:gpu ./l0bench --csv results.csv

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>

using clk = std::chrono::high_resolution_clock;
using ns  = std::chrono::nanoseconds;

struct Args {
  bool csv = false;
  std::string csv_path;
  bool human = true;
  bool pin_host = true;     // use pinned host alloc if available
  bool batch = false;       // enqueue all then one wait (vs sync after each)
  size_t min_bytes = 1<<10; // 1 KiB
  size_t max_bytes = 1<<20; // 1 MiB (adjust if you like)
  int    steps     = 11;    // sizes: geometric from min..max
  int    secs_target = 1;   // per size target seconds budget (cap reps)
  int    device_index = 0;  // pick GPU index
};

static void print_help() {
  std::cout <<
R"(l0bench - SYCL copy micro-benchmark (H2D/D2H/D2D)
Options:
  --csv <file>        Write CSV results to <file>
  --no-human          Suppress human table output
  --no-pin            Use pageable host memory (slower H2D/D2H, but reveals staging effects)
  --batch             Enqueue all copies then single wait (vs. wait after each)
  --min <bytes>       Minimum size (default 1024)
  --max <bytes>       Maximum size (default 1048576)
  --steps <n>         Geometric steps between min..max (default 11)
  --secs <s>          Time budget per size (default 1s; reps adapt)
  --dev <i>           GPU device index (default 0)
  --help              Show this help
Env you probably want:
  SYCL_DEVICE_FILTER=level_zero:gpu
  UR_ADAPTERS_FORCE_ORDER=LEVEL_ZERO
)";
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i) {
    std::string s(argv[i]);
    auto need = [&](int k){ if(i+k>=argc) { std::cerr<<"Missing value for "<<s<<"\n"; std::exit(2);} return std::string(argv[i+k]); };
    if (s=="--csv")            { a.csv=true; a.csv_path=need(1); ++i; }
    else if (s=="--no-human")  { a.human=false; }
    else if (s=="--no-pin")    { a.pin_host=false; }
    else if (s=="--batch")     { a.batch=true; }
    else if (s=="--min")       { a.min_bytes=std::stoull(need(1)); ++i; }
    else if (s=="--max")       { a.max_bytes=std::stoull(need(1)); ++i; }
    else if (s=="--steps")     { a.steps=std::stoi(need(1)); ++i; }
    else if (s=="--secs")      { a.secs_target=std::stoi(need(1)); ++i; }
    else if (s=="--dev")       { a.device_index=std::stoi(need(1)); ++i; }
    else if (s=="--help")      { print_help(); std::exit(0); }
    else { std::cerr<<"Unknown arg: "<<s<<"\n"; print_help(); std::exit(2); }
  }
  if (a.min_bytes<1) a.min_bytes=1;
  if (a.max_bytes<a.min_bytes) a.max_bytes=a.min_bytes;
  if (a.steps<1) a.steps=1;
  return a;
}

struct ResultRow {
  std::string op;     // H2D/D2H/D2D
  size_t bytes;
  int reps;
  double avg_us;      // average per copy (microseconds)
  double gbps;        // effective bandwidth (GiB/s)
  bool   batch;
  bool   pinned;
};

static std::vector<size_t> make_sizes(size_t minb, size_t maxb, int steps) {
  if (steps==1) return {maxb};
  std::vector<size_t> v; v.reserve(steps);
  double r = std::pow(double(maxb)/double(minb), 1.0/(steps-1));
  for (int i=0;i<steps;++i) {
    size_t s = size_t(double(minb)*std::pow(r,i));
    s = std::max<size_t>(1, s);
    v.push_back(s);
  }
  v.back() = maxb;
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

static inline double to_gib(double bytes){ return bytes / double(1024ull*1024ull*1024ull); }

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // Select GPU device i
  std::vector<sycl::device> gpus;
  for (auto& d: sycl::device::get_devices(sycl::info::device_type::gpu)) gpus.push_back(d);
  if (gpus.empty()) { std::cerr<<"No GPU device found.\n"; return 1; }
  if (args.device_index<0 || args.device_index>=int(gpus.size())) {
    std::cerr<<"GPU index out of range. Found "<<gpus.size()<<" GPUs.\n"; return 2;
  }
  sycl::device dev = gpus[args.device_index];
  sycl::context ctx{dev};
  sycl::queue q{ctx, dev, sycl::property::queue::in_order{}}; // in-order for stable ordering

  auto name = dev.get_info<sycl::info::device::name>();
  auto be   = dev.get_backend(); // show that it's level_zero ideally

  if (args.human) {
    std::cout << "=== SYCL Micro-Benchmark (Copies) ===\n";
    std::cout << "Device: " << name << "\n";
    std::cout << "Backend: " << (be==sycl::backend::ext_oneapi_level_zero ? "ext_oneapi_level_zero" :
                                  be==sycl::backend::ext_oneapi_cuda ? "ext_oneapi_cuda" :
                                  be==sycl::backend::opencl ? "opencl" :
                                  be==sycl::backend::native_cpu ? "native_cpu" : "other")
              << "\n";
    std::cout << "Mode: " << (args.batch ? "batch" : "sync-each") << ", HostPinned: " << (args.pin_host?"yes":"no") << "\n\n";
  }

  // Allocate one large buffer once; we reuse subranges for all sizes.
  const size_t BIG = args.max_bytes;
  // host memory
  void* hbuf_src = nullptr;
  void* hbuf_dst = nullptr;
  if (args.pin_host) {
    hbuf_src = sycl::aligned_alloc_host(4096, BIG, q);
    hbuf_dst = sycl::aligned_alloc_host(4096, BIG, q);
  } else {
    hbuf_src = std::aligned_alloc(4096, BIG);
    hbuf_dst = std::aligned_alloc(4096, BIG);
  }
  if(!hbuf_src || !hbuf_dst) { std::cerr<<"Host alloc failed\n"; return 3; }
  std::memset(hbuf_src, 0x5A, BIG);
  std::memset(hbuf_dst, 0x00, BIG);

  // device memory
  void* dbuf_src = sycl::malloc_device(BIG, q);
  void* dbuf_dst = sycl::malloc_device(BIG, q);
  if(!dbuf_src || !dbuf_dst) { std::cerr<<"Device alloc failed\n"; return 4; }

  // Warm-up: move something
  q.memcpy(dbuf_src, hbuf_src, BIG).wait();
  q.memcpy(dbuf_dst, dbuf_src, BIG).wait();

  auto sizes = make_sizes(args.min_bytes, args.max_bytes, args.steps);
  std::vector<ResultRow> rows; rows.reserve(sizes.size()*6);

  auto run_one = [&](const char* op, size_t bytes, bool batch) -> ResultRow {
    // choose src/dst pointers
    const void* src=nullptr; void* dst=nullptr;
    if (std::string(op)=="H2D"){ src=hbuf_src; dst=dbuf_dst; }
    else if (std::string(op)=="D2H"){ src=dbuf_src; dst=hbuf_dst; }
    else /*D2D*/ { src=dbuf_src; dst=dbuf_dst; }

    // pick reps adaptively to ~secs_target per size
    // measure one to estimate
    q.wait();
    auto t0 = clk::now();
    q.memcpy(dst, src, bytes).wait();
    auto t1 = clk::now();
    double one_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0;
    // aim for args.secs_target seconds per size
    int reps = std::max(5, int((args.secs_target*1e6) / std::max(1.0, one_us)));
    // cap reps to something sane for large sizes
    reps = std::min(reps, 5000);

    std::vector<sycl::event> evs; evs.reserve(reps);
    t0 = clk::now();
    if (batch) {
      for (int i=0;i<reps;++i) {
        evs.push_back(q.memcpy(dst, src, bytes));
      }
      q.wait(); // one wait for all
    } else {
      for (int i=0;i<reps;++i) {
        q.memcpy(dst, src, bytes).wait(); // sync each
      }
    }
    t1 = clk::now();
    double total_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0;
    double avg_us   = total_us / double(reps);
    double gbps     = to_gib(double(bytes)) / (avg_us/1e6); // GiB / seconds

    // quick data check for D2D path to ensure not optimized away
    if (std::string(op)=="D2D" && bytes>=16) {
      // flip one byte to vary
      q.memcpy((char*)dbuf_dst, (char*)dbuf_src, bytes).wait();
    }
    return {op, bytes, reps, avg_us, gbps, batch, args.pin_host};
  };

  auto push = [&](ResultRow r){ rows.push_back(r); if(args.human) {
      std::cout<< std::left << std::setw(4) << r.op
               << " size="<< std::setw(8) << r.bytes
               << " reps="<< std::setw(6) << r.reps
               << " avg(us)="<< std::setw(10) << std::fixed<<std::setprecision(3)<< r.avg_us
               << " GiB/s="<< std::setw(8) << std::fixed<<std::setprecision(2)<< r.gbps
               << (r.batch? " [batch]":" [sync]") << (r.pinned? " [pinned]":" [pageable]") << "\n";
  }};

  if (args.human) {
    std::cout << "Running sizes:";
    for (auto s: sizes) std::cout<<" "<<s;
    std::cout << "\n\n";
  }

  for (auto bytes: sizes) {
    // D2D measures dein Backend-Fastpath direkt
    push(run_one("D2D", bytes, args.batch));
    // Host <-> Device (zeigt Nutzen von Host-Pinning und Staging)
    push(run_one("H2D", bytes, args.batch));
    push(run_one("D2H", bytes, args.batch));
  }

  if (args.csv) {
    FILE* f = std::fopen(args.csv_path.c_str(), "w");
    if (!f) { std::perror("fopen"); }
    else {
      std::fprintf(f, "device,backend,mode,pinned,op,bytes,reps,avg_us,gib_per_s\n");
      for (auto& r: rows) {
        std::fprintf(f, "\"%s\",%s,%s,%s,%s,%zu,%d,%.3f,%.6f\n",
          name.c_str(),
          (be==sycl::backend::ext_oneapi_level_zero ? "level_zero" :
           be==sycl::backend::ext_oneapi_cuda ? "cuda" :
           be==sycl::backend::opencl ? "opencl" : "other"),
          (r.batch ? "batch" : "sync"),
          (r.pinned? "yes":"no"),
          r.op.c_str(), r.bytes, r.reps, r.avg_us, r.gbps);
      }
      std::fclose(f);
      if (args.human) std::cout << "\nCSV written: " << args.csv_path << "\n";
    }
  }

  // cleanup
  sycl::free(dbuf_src, q);
  sycl::free(dbuf_dst, q);
  if (args.pin_host) {
    sycl::free(hbuf_src, q);
    sycl::free(hbuf_dst, q);
  } else {
    std::free(hbuf_src);
    std::free(hbuf_dst);
  }

  return 0;
}
