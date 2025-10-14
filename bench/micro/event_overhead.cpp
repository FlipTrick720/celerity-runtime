#include "common/bench_util.hpp"
#include "common/sycl_helpers.hpp"
#include <vector>
#include <iostream>
#include <ctime>
#include <unistd.h>

using namespace bench;

struct Row { std::string kind; int reps; double avg_us; };

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
		std::cout << "=== event_overhead ===\n";
		std::cout << "Device: " << di.name << "\n";
		std::cout << "Backend: " << backend_name(di.backend) << "\n";
		std::cout << "Host: " << host_s << ", Timestamp: " << tstamp << ", Git: " << git_sha << "\n\n";
	}
	
	Csv csv; if(args.csv) csv.open(args.csv_path, "ts,host,git,bench,backend,device,kind,reps,avg_us");
	
	auto measure = [&](const char* kind, auto submit)->Row{
		int reps = 1000;
		auto t0 = clk::now();
		for(int i=0;i<reps;i++) submit();
		q.wait();
		auto t1 = clk::now();
		double one_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0 / reps;
		
		reps = std::max(100, int((args.secs_target*1e6)/std::max(1.0, one_us)));
		reps = std::min(reps, args.reps_cap);
		
		t0 = clk::now();
		for(int i=0;i<reps;i++) submit();
		q.wait();
		t1 = clk::now();
		
		double total_us = std::chrono::duration_cast<ns>(t1-t0).count() / 1000.0;
		double avg = total_us / double(reps);
		return {kind, reps, avg};
	};
	
	auto empty_kernel = [&](){ q.single_task([=](){ }); };
	static uint8_t dummy = 0;
	auto tiny_copy = [&](){ q.memcpy(&dummy, &dummy, 1); };
	
	std::vector<Row> rows;
	rows.push_back(measure("single_task_empty", empty_kernel));
	rows.push_back(measure("memcpy_1B", tiny_copy));
	
	for(auto& r: rows){
		if(args.human){
			std::cout
				<< std::left << std::setw(20) << r.kind
				<< " reps="
				<< std::setw(6) << r.reps
				<< " avg(us)="
				<< std::setw(10) << std::fixed<<std::setprecision(3)
				<< r.avg_us << "\n";
		}
		if(args.csv){
			std::stringstream ss;
			ss<<tstamp<<","<<host_s<<","<<git_sha
				<<",event_overhead,"<<backend_name(di.backend)<<",\""<<di.name<<"\","
				<<r.kind<<","<<r.reps<<","<<std::fixed<<std::setprecision(3)<<r.avg_us;
			csv.row(ss.str());
		}
	}
	
	return 0;
}
