#include "common/bench_util.hpp"
#include "common/sycl_helpers.hpp"
#include <vector>
#include <cstring>
#include <ctime>
#include <unistd.h>

using namespace bench;

struct Row {
	std::string op; size_t bytes; int reps;
	double avg_us; double gib_s; bool batch; bool pinned;
};

static Row run_one(sycl::queue& q, const Args& args,
	const char* op, void* hsrc, void* hdst, void* dsrc, void* ddst, size_t bytes) {
	
	const void* src=nullptr; void* dst=nullptr;
	if (std::string(op)=="H2D"){ src=hsrc; dst=ddst; }
	else if (std::string(op)=="D2H"){ src=dsrc; dst=hdst; }
	else /*D2D*/ { src=dsrc; dst=ddst; }
	
	// warm-up: 5 reps
	for (int w=0; w<5; w++) {
		q.memcpy(dst, src, bytes).wait();
	}
	
	// calibrate reps
	q.wait();
	auto t0 = clk::now();
	q.memcpy(dst, src, bytes).wait();
	auto t1 = clk::now();
	double one_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0;
	int reps = std::max(5, int((args.secs_target*1e6) / std::max(1.0, one_us)));
	reps = std::min(reps, args.reps_cap);
	
	t0 = clk::now();
	if (args.batch) {
		std::vector<sycl::event> evs; evs.reserve(reps);
		for (int i=0;i<reps;i++) evs.push_back(q.memcpy(dst, src, bytes));
		q.wait(); // one wait for all
	} else {
		for (int i=0;i<reps;i++) q.memcpy(dst, src, bytes).wait(); // sync-each
	}
	t1 = clk::now();
	
	double total_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0;
	double avg = total_us / double(reps);
	double bw  = to_gib(double(bytes)) / (avg/1e6);
	return {op, bytes, reps, avg, bw, args.batch, args.pin_host};
}

int main(int argc, char** argv){
	auto args = parse_args(argc, argv);
	auto q = make_queue(args.device_index, /*profiling*/true);
	auto di = device_info(q);
	
	// Metadata for reproducibility
	char hostname[256];
	gethostname(hostname, sizeof(hostname));
	std::string host_s(hostname);
	std::string tstamp = std::to_string(std::time(nullptr));
	#ifndef GIT_SHA
	#define GIT_SHA "unknown"
	#endif
	std::string git_sha = GIT_SHA;
	
	if (args.human) {
		std::cout << "=== memcpy_linear ===\n";
		std::cout << "Device: " << di.name << "\n";
		std::cout << "Backend: " << backend_name(di.backend) << "\n";
		std::cout << "Mode: " << (args.batch?"batch":"sync-each") << ", HostPinned: " << (args.pin_host?"yes":"no") << "\n";
		std::cout << "Host: " << host_s << ", Timestamp: " << tstamp << ", Git: " << git_sha << "\n\n";
	}
	
	// allocate once
	size_t BIG = args.max_bytes;
	void* hsrc = alloc_host(BIG, q, args.pin_host);
	void* hdst = alloc_host(BIG, q, args.pin_host);
	if(!hsrc||!hdst){ std::cerr<<"host alloc failed\n"; return 1; }
	std::memset(hsrc, 0x5A, BIG);
	std::memset(hdst, 0x00, BIG);
	
	void* dsrc = sycl::malloc_device(BIG, q);
	void* ddst = sycl::malloc_device(BIG, q);
	if(!dsrc||!ddst){ std::cerr<<"device alloc failed\n"; return 2; }
	
	// warmup
	q.memcpy(dsrc, hsrc, BIG).wait();
	q.memcpy(ddst, dsrc, BIG).wait();
	
	auto sizes = geometric_sizes(args.min_bytes, args.max_bytes, args.steps);
	
	Csv csv;
	if (args.csv) csv.open(args.csv_path, "ts,host,git,bench,backend,device,mode,pinned,op,bytes,reps,avg_us,gib_per_s");
	
	for (auto bytes: sizes) {
		for (auto op: { "D2D", "H2D", "D2H" }) {
			auto row = run_one(q, args, op, hsrc, hdst, dsrc, ddst, bytes);
			
			if (args.human) {
				std::cout
					<< std::left << std::setw(4) << row.op
					<< " size="
					<< std::setw(10) << row.bytes
					<< " reps="
					<< std::setw(6) << row.reps
					<< " avg(us)="
					<< std::setw(10) << std::fixed<<std::setprecision(3)
					<< row.avg_us
					<< " GiB/s="
					<< std::setw(8) << std::fixed<<std::setprecision(2)
					<< row.gib_s
					<< (row.batch? " [batch]":" [sync]") << (row.pinned? " [pinned]":" [pageable]") << "\n";
			}
			
			if (args.csv) {
				std::stringstream ss;
				ss<<tstamp<<","<<host_s<<","<<git_sha
					<<",memcpy_linear,"<<backend_name(di.backend)<<",\""<<di.name<<"\""
					<<","<<(row.batch?"batch":"sync")
					<<","<<(row.pinned?"yes":"no")
					<<","<<row.op
					<<","<<row.bytes
					<<","<<row.reps
					<<","<<std::fixed<<std::setprecision(3)<<row.avg_us
					<<","<<std::fixed<<std::setprecision(6)<<row.gib_s;
				csv.row(ss.str());
			}
		}
	}
	
	sycl::free(dsrc, q); sycl::free(ddst, q);
	free_host(hsrc, q, args.pin_host); free_host(hdst, q, args.pin_host);
	return 0;
}
