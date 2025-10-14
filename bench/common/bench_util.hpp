#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <cmath>

namespace bench {

using clk = std::chrono::high_resolution_clock;
using ns  = std::chrono::nanoseconds;

struct Args {
	bool csv = false;
	std::string csv_path;
	bool human = true;
	bool batch = false;
	bool pin_host = true;
	size_t min_bytes = 1<<10; // 1 KiB
	size_t max_bytes = 1<<20; // 1 MiB
	int    steps     = 13;    // geometric
	int    secs_target = 1;   // per-size budget
	int    reps_cap   = 5000; // cap reps to avoid very long runs
	int    device_index = 0;
	bool   verbose = false;
};

inline void print_help(const char* prog){
	std::cout <<
		"Options:\n"
		"  --csv <file>        Write CSV results to <file>\n"
		"  --no-human          Suppress human table output\n"
		"  --batch             Enqueue all ops then one wait (vs. sync after each)\n"
		"  --no-pin            Use pageable host memory\n"
		"  --min <bytes>       Minimum size (default 1024)\n"
		"  --max <bytes>       Maximum size (default 1048576)\n"
		"  --steps <n>         Geometric steps between min..max (default 13)\n"
		"  --secs <s>          Time budget per size (default 1s)\n"
		"  --reps-cap <n>      Upper cap for reps (default 5000)\n"
		"  --dev <i>           GPU device index (default 0)\n"
		"  --verbose           Print extra info\n"
		"  --help              Show this help\n";
}

inline Args parse_args(int argc, char** argv) {
	Args a;
	for (int i=1;i<argc;++i) {
		std::string s(argv[i]);
		auto need = [&](int k){ if(i+k>=argc) { std::cerr<<"Missing value for "<<s<<"\n"; std::exit(2);} return std::string(argv[i+k]); };
		if (s=="--csv")            { a.csv=true; a.csv_path=need(1); ++i; }
		else if (s=="--no-human")  { a.human=false; }
		else if (s=="--batch")     { a.batch=true; }
		else if (s=="--no-pin")    { a.pin_host=false; }
		else if (s=="--min")       { a.min_bytes=std::stoull(need(1)); ++i; }
		else if (s=="--max")       { a.max_bytes=std::stoull(need(1)); ++i; }
		else if (s=="--steps")     { a.steps=std::stoi(need(1)); ++i; }
		else if (s=="--secs")      { a.secs_target=std::stoi(need(1)); ++i; }
		else if (s=="--reps-cap")  { a.reps_cap=std::stoi(need(1)); ++i; }
		else if (s=="--dev")       { a.device_index=std::stoi(need(1)); ++i; }
		else if (s=="--verbose")   { a.verbose=true; }
		else if (s=="--help")      { print_help(argv[0]); std::exit(0); }
		else { std::cerr<<"Unknown arg: "<<s<<"\n"; print_help(argv[0]); std::exit(2); }
	}
	if (a.min_bytes<1) a.min_bytes=1;
	if (a.max_bytes<a.min_bytes) a.max_bytes=a.min_bytes;
	if (a.steps<1) a.steps=1;
	return a;
}

inline std::vector<size_t> geometric_sizes(size_t minb, size_t maxb, int steps) {
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

inline double to_gib(double bytes){ return bytes / double(1024ull*1024ull*1024ull); }

struct Csv {
	FILE* f=nullptr;
	~Csv(){ if(f) std::fclose(f); }
	void open(const std::string& path, const char* header){
		f = std::fopen(path.c_str(), "w");
		if(!f){ std::perror("fopen"); std::exit(1); }
		std::fprintf(f, "%s\n", header);
	}
	void row(const std::string& s){
		if(!f) return;
		std::fprintf(f, "%s\n", s.c_str());
	}
};

} // namespace bench
