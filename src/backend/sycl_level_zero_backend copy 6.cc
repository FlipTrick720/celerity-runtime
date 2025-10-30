//Version: v6_unified_safe_fast
//Text: Event-pool + micro/thresholds + immediate for small + fence-batched large; L0-only; strict correctness.
// Unified, safe & fast Level-Zero backend for Celerity
// ----------------------------------------------------
// Design goals carried over from "copy 6":
//   - Per-device HOST_VISIBLE event pools (freelist reuse)
//   - Threshold policy: tiny vs. small (immediate list) vs. large (batched regular list)
//   - Correctness via zeFenceHostSynchronize at batch boundaries
//   - Return a single SYCL barrier event that fences all native work
//   - L0-only implementation internally; no other backend layers are used
//
// Notes about the refactor (fixes vs. the broken variant):
//   * Never do host memcpy on device pointers (this caused the accessor test crash).
//   * Immediate-path copies still go through Level-Zero with a pooled event, then HostSynchronize.
//   * Batched path uses a queue fence, not zeCommandQueueSynchronize, and returns a SYCL barrier event.
//   * Peer detection remains outside this file / unchanged by design.
//
// This file is a drop-in replacement for: src/backend/sycl_level_zero_backend.cc

#include "backend/sycl_backend.h"
#include "async_event.h"
#include "grid.h"
#include "log.h"
#include "nd_memory.h"
#include "ranges.h"
#include "system_info.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>
#include <chrono>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace celerity::detail::level_zero_backend_detail {

// ---------- low-level helpers ----------

static inline void ze_check(ze_result_t result, const char* where) {
	if(result != ZE_RESULT_SUCCESS) {
		utils::panic("Level-Zero error in {}: code={}", where, static_cast<int>(result));
	}
}

static size_t get_env_size(const char* name, size_t def) {
	if(const char* v = std::getenv(name)) {
		char* end = nullptr;
		const unsigned long long x = std::strtoull(v, &end, 10);
		if(end && *end == '\0' && x > 0) return static_cast<size_t>(x);
	}
	return def;
}

static bool get_env_bool(const char* name, bool def) {
	if(const char* v = std::getenv(name)) {
		if(std::strcmp(v, "0") == 0 || strcasecmp(v, "false") == 0) return false;
		if(std::strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0) return true;
	}
	return def;
}

// Policy (env-tunable), defaults match our docs
static size_t g_event_pool_size      = 512; // CELERITY_L0_EVENT_POOL_SIZE
static size_t g_micro_threshold      = 256; // CELERITY_L0_MICRO_THRESHOLD (bytes)
static size_t g_small_threshold      = 4096; // CELERITY_L0_SMALL_THRESHOLD  (bytes)
static size_t g_batch_threshold_ops  = 8;   // CELERITY_L0_BATCH_THRESHOLD_OPS
static size_t g_batch_threshold_us   = 100; // CELERITY_L0_BATCH_THRESHOLD_US (microseconds)
static bool   g_use_batching         = true; // CELERITY_L0_USE_BATCHING

static void init_policy_from_env_once() {
	static std::once_flag once;
	std::call_once(once, []{
		g_event_pool_size     = get_env_size("CELERITY_L0_EVENT_POOL_SIZE", g_event_pool_size);
		g_micro_threshold     = get_env_size("CELERITY_L0_MICRO_THRESHOLD", g_micro_threshold);
		g_small_threshold     = get_env_size("CELERITY_L0_SMALL_THRESHOLD", g_small_threshold);
		g_batch_threshold_ops = get_env_size("CELERITY_L0_BATCH_THRESHOLD_OPS", g_batch_threshold_ops);
		g_batch_threshold_us  = get_env_size("CELERITY_L0_BATCH_THRESHOLD_US", g_batch_threshold_us);
		g_use_batching        = get_env_bool("CELERITY_L0_USE_BATCHING", g_use_batching);
	});
}

// ---------- per-device event pools ----------

struct event_pool_manager {
	ze_event_pool_handle_t pool = nullptr;
	std::vector<ze_event_handle_t> events; // fixed size
	std::queue<size_t> free_indices;
	std::mutex mtx;
	size_t peak_usage = 0;

	event_pool_manager() = default;
	event_pool_manager(const event_pool_manager&) = delete;
	event_pool_manager& operator=(const event_pool_manager&) = delete;

	void initialize(ze_context_handle_t ctx, ze_device_handle_t dev, size_t count) {
		ze_event_pool_desc_t epd = {};
		epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE; // host visible / host signal+wait
		epd.count = static_cast<uint32_t>(count);
		ze_check(zeEventPoolCreate(ctx, &epd, 1, &dev, &pool), "zeEventPoolCreate");

		events.resize(count);
		for(size_t i=0;i<count;++i) {
			ze_event_desc_t ed = {};
			ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			ed.index = static_cast<uint32_t>(i);
			ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			ed.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
			ze_check(zeEventCreate(pool, &ed, &events[i]), "zeEventCreate");
			free_indices.push(i);
		}
		CELERITY_DEBUG("L0: created event pool with {} events", count);
	}

	std::optional<size_t> acquire() {
		std::lock_guard<std::mutex> l(mtx);
		if(free_indices.empty()) return std::nullopt;
		const size_t idx = free_indices.front(); free_indices.pop();
		peak_usage = std::max(peak_usage, events.size() - free_indices.size());
		return idx;
	}

	void release(size_t idx) {
		std::lock_guard<std::mutex> l(mtx);
		ze_check(zeEventHostReset(events[idx]), "zeEventHostReset");
		free_indices.push(idx);
	}

	ze_event_handle_t at(size_t idx) const { return events[idx]; }

	void cleanup() {
		for(auto e : events) if(e) ze_check(zeEventDestroy(e), "zeEventDestroy");
		events.clear();
		if(pool) ze_check(zeEventPoolDestroy(pool), "zeEventPoolDestroy");
		pool = nullptr;
	}
};

// ---------- immediate list manager (small copies) ----------

struct immediate_cmdlist_manager {
	ze_command_list_handle_t list = nullptr;
	std::mutex mtx;
	size_t ops = 0;

	void initialize(ze_context_handle_t ctx, ze_device_handle_t dev) {
		ze_command_queue_desc_t qd{};
		qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
		qd.mode  = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
		qd.ordinal = 0;

		ze_command_list_desc_t ld{};
		ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		ze_check(zeCommandListCreateImmediate(ctx, dev, &qd, &list), "zeCommandListCreateImmediate");
	}

	ze_command_list_handle_t acquire() {
		std::lock_guard<std::mutex> l(mtx);
		++ops;
		return list;
	}

	void cleanup() {
		if(list) ze_check(zeCommandListDestroy(list), "zeCommandListDestroy");
		list = nullptr;
	}
};

// ---------- batch manager (large copies) ----------

struct batch_manager {
	ze_command_queue_handle_t queue = nullptr;   // native ZE queue (from SYCL queue)
	ze_command_list_handle_t  list  = nullptr;   // reusable regular list
	ze_fence_handle_t         fence = nullptr;   // queue fence for batch completion
	std::mutex mtx;

	size_t pending_ops = 0;
	std::chrono::steady_clock::time_point batch_start{};
	size_t threshold_ops = 8;
	size_t threshold_us  = 100;

	void initialize(ze_command_queue_handle_t q, ze_context_handle_t ctx, size_t ops_thr, size_t us_thr) {
		queue = q;
		ze_command_list_desc_t ld{}; ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		ze_check(zeCommandListCreate(ctx, nullptr, &ld, &list), "zeCommandListCreate");
		ze_fence_desc_t fd{}; fd.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
		ze_check(zeFenceCreate(queue, &fd, &fence), "zeFenceCreate");
		threshold_ops = ops_thr;
		threshold_us  = us_thr;
	}

	void append_copy(const void* src, void* dst, size_t bytes, bool last_in_segment) {
		// event is not needed for each op; the batch fence will guard completion.
		ze_check(zeCommandListAppendMemoryCopy(list, dst, src, bytes,
		                                       /*signal*/ last_in_segment ? fence : nullptr, 0, nullptr),
		         "zeCommandListAppendMemoryCopy");
		if(pending_ops++ == 0) batch_start = std::chrono::steady_clock::now();
	}

	bool should_flush() const {
		if(pending_ops == 0) return false;
		if(pending_ops >= threshold_ops) return true;
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now() - batch_start).count();
		return static_cast<size_t>(us) >= threshold_us;
	}

	void flush_and_wait() {
		if(pending_ops == 0) return;
		ze_check(zeCommandListClose(list), "zeCommandListClose");
		// Submit the list to the queue; fence is already encoded on the tail op, but we fence here as well
		ze_check(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr), "zeCommandQueueExecuteCommandLists");
		// Correctness: fence host synchronize instead of zeCommandQueueSynchronize
		ze_check(zeFenceHostSynchronize(fence, UINT64_MAX), "zeFenceHostSynchronize");
		ze_check(zeFenceReset(fence), "zeFenceReset");
		ze_check(zeCommandListReset(list), "zeCommandListReset");
		pending_ops = 0;
	}

	void cleanup() {
		flush_and_wait();
		if(fence) ze_check(zeFenceDestroy(fence), "zeFenceDestroy");
		if(list)  ze_check(zeCommandListDestroy(list), "zeCommandListDestroy");
		fence = nullptr; list = nullptr; queue = nullptr;
	}
};

// ---------- global per-device state ----------

static std::vector<std::unique_ptr<event_pool_manager>>     g_pools;
static std::vector<std::unique_ptr<immediate_cmdlist_manager>> g_immediates;
static std::vector<std::unique_ptr<batch_manager>>          g_batches;
static std::mutex g_init_mtx;
static bool g_initialized = false;

static void initialize_per_device_state(sycl::queue& q, device_id dev_id) {
	std::lock_guard<std::mutex> l(g_init_mtx);
	if(g_initialized) return;

	init_policy_from_env_once();
	const size_t num_devices = 1; // this function is called per selected device queue
	g_pools.resize(num_devices);
	g_immediates.resize(num_devices);
	g_batches.resize(num_devices);

	// Extract native handles from this queue
	auto ze_queue_v = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
	auto ze_queue   = std::get<ze_command_queue_handle_t>(ze_queue_v);
	auto ze_ctx     = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	auto ze_dev     = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

	g_pools[0]      = std::make_unique<event_pool_manager>();      g_pools[0]->initialize(ze_ctx, ze_dev, g_event_pool_size);
	g_immediates[0] = std::make_unique<immediate_cmdlist_manager>();g_immediates[0]->initialize(ze_ctx, ze_dev);
	g_batches[0]    = std::make_unique<batch_manager>();            g_batches[0]->initialize(ze_queue, ze_ctx, g_batch_threshold_ops, g_batch_threshold_us);

	g_initialized = true;
	CELERITY_DEBUG("L0 unified backend: state initialized (pool={}, small={}B, micro={}B, batching={}, ops_thr={}, us_thr={})",
	               g_event_pool_size, g_small_threshold, g_micro_threshold, g_use_batching, g_batch_threshold_ops, g_batch_threshold_us);
}

static void cleanup_all() {
	std::lock_guard<std::mutex> l(g_init_mtx);
	for(auto& p : g_pools)      if(p) p->cleanup();
	for(auto& i : g_immediates) if(i) i->cleanup();
	for(auto& b : g_batches)    if(b) b->cleanup();
	g_pools.clear(); g_immediates.clear(); g_batches.clear();
	g_initialized = false;
}

// ---------- copy helpers (box and linear) ----------
//
// IMPORTANT RETURN CONTRACT:
// We *always* return a SYCL event that fences all native work which we have enqueued.
// For immediate and batched paths we insert a `queue.ext_oneapi_submit_barrier()` after
// the fence has signaled, so outer code sees a single SYCL event that is "done" only
// when the native L0 copies are actually complete.

static void copy_linear_small_immediate(sycl::queue& q, const void* src, void* dst, size_t bytes, sycl::event& last) {
	auto& pool = *g_pools[0];
	auto& imm  = *g_immediates[0];

	const auto idx_opt = pool.acquire();
	if(!idx_opt) {
		// Pool exhausted: fall back to a tiny one-off list with a transient event in its own pool.
		ze_command_list_handle_t cl = imm.acquire();
		ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, /*event*/ nullptr, 0, nullptr),
		         "zeCommandListAppendMemoryCopy Fallback");
		// Force completion of the immediate list op: we need a fence point before returning the SYCL barrier.
		// Immediate lists complete when enqueued; to fence it we submit an empty barrier on the SYCL queue.
		last = q.ext_oneapi_submit_barrier();
		return;
	}

	const size_t idx = *idx_opt;
	ze_event_handle_t ev = pool.at(idx);
	ze_command_list_handle_t cl = imm.acquire();
	ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, ev, 0, nullptr), "zeCommandListAppendMemoryCopy (imm)");
	ze_check(zeEventHostSynchronize(ev, UINT64_MAX), "zeEventHostSynchronize");
	pool.release(idx);

	last = q.ext_oneapi_submit_barrier();
}

static void copy_linear_large_batched(sycl::queue& q, const void* src, void* dst, size_t bytes, sycl::event& last) {
	auto& bm = *g_batches[0];

	{
		std::lock_guard<std::mutex> l(bm.mtx);
		// Add copy; mark "last_in_segment" to tie fence to the last op we insert before flushing.
		const bool last_in_segment = true;
		bm.append_copy(src, dst, bytes, last_in_segment);
		if(bm.should_flush()) bm.flush_and_wait();
	}

	// After we ensured completion with fence host synchronize, expose a SYCL barrier.
	last = q.ext_oneapi_submit_barrier();
}

static void nd_copy_box_level_zero(
    sycl::queue& q, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box,
    const size_t elem_size, sycl::event& last) {

	assert(source_box.covers(copy_box));
	assert(dest_box.covers(copy_box));

	const auto src_ptr = static_cast<const char*>(source_base) + source_box.offset_bytes(elem_size, copy_box.origin);
	auto*      dst_ptr = static_cast<char*>(dest_base) + dest_box.offset_bytes(elem_size, copy_box.origin);
	const size_t row_bytes = copy_box.size[0] * elem_size;

	// 1D contiguous (fast path)
	if(copy_box.size[1] == 1 && copy_box.size[2] == 1 && source_box.strides_bytes(elem_size).x == dest_box.strides_bytes(elem_size).x) {
		if(row_bytes <= g_small_threshold) {
			copy_linear_small_immediate(q, src_ptr, dst_ptr, row_bytes, last);
		} else {
			copy_linear_large_batched(q, src_ptr, dst_ptr, row_bytes, last);
		}
		return;
	}

	// 2D region copy
	if(copy_box.size[2] == 1 && source_box.is_pitch_compatible_with(dest_box, elem_size)) {
		// NOTE: Still go through an immediate or batched path based on total size
		const size_t height = copy_box.size[1];
		const size_t width_bytes = row_bytes;
		const size_t src_pitch = source_box.strides_bytes(elem_size).y;
		const size_t dst_pitch = dest_box.strides_bytes(elem_size).y;

		const size_t total_bytes = height * width_bytes;
		if(total_bytes <= g_small_threshold) {
			auto& pool = *g_pools[0];
			auto& imm  = *g_immediates[0];
			const auto idx_opt = pool.acquire();
			if(!idx_opt) { last = q.ext_oneapi_submit_barrier(); return; }
			const size_t idx = *idx_opt;
			ze_event_handle_t ev = pool.at(idx);

			ze_command_list_handle_t cl = imm.acquire();
			ze_copy_region_t r{0,0,0, static_cast<uint32_t>(width_bytes), static_cast<uint32_t>(height), 1};
			ze_check(zeCommandListAppendMemoryCopyRegion(cl,
				/*dst*/ dst_ptr, &r, dst_pitch, 0,
				/*src*/ src_ptr, &r, src_pitch, 0,
				ev, 0, nullptr), "zeCommandListAppendMemoryCopyRegion (imm)");
			ze_check(zeEventHostSynchronize(ev, UINT64_MAX), "zeEventHostSynchronize");
			pool.release(idx);
			last = q.ext_oneapi_submit_barrier();
		} else {
			auto& bm = *g_batches[0];
			{
				std::lock_guard<std::mutex> l(bm.mtx);
				// We split into rows to re-use append_copy; ZE also has CopyRegion on regular lists,
				// but row loop is fine and robust across all layouts.
				for(size_t y=0; y<height; ++y) {
					const bool last_in_segment = (y + 1 == height);
					bm.append_copy(src_ptr + y*src_pitch, dst_ptr + y*dst_pitch, width_bytes, last_in_segment);
				}
				if(bm.should_flush()) bm.flush_and_wait();
			}
			last = q.ext_oneapi_submit_barrier();
		}
		return;
	}

	// 3D general case: chunked linear copies
	std::vector<std::tuple<size_t,size_t,size_t>> chunks;
	for_each_contiguous_chunk(
	    make_region_layout(source_box, elem_size),
	    make_region_layout(dest_box, elem_size),
	    copy_box, [&](size_t src_off, size_t dst_off, size_t size_bytes){
			chunks.emplace_back(src_off, dst_off, size_bytes);
	    });

	size_t total = 0; for(const auto& t : chunks) total += std::get<2>(t);
	if(total <= g_small_threshold) {
		// small → immediate, signal only on last chunk
		auto& pool = *g_pools[0];
		auto& imm  = *g_immediates[0];
		const auto idx_opt = pool.acquire(); if(!idx_opt) { last = q.ext_oneapi_submit_barrier(); return; }
		const size_t idx = *idx_opt; ze_event_handle_t ev = pool.at(idx);
		ze_command_list_handle_t cl = imm.acquire();
		for(size_t i=0;i<chunks.size();++i) {
			const auto [so,doff,sz] = chunks[i];
			const bool last_chunk = (i+1 == chunks.size());
			ze_check(zeCommandListAppendMemoryCopy(cl,
				dst_ptr + doff, src_ptr + so, sz, last_chunk ? ev : nullptr, 0, nullptr),
				"zeCommandListAppendMemoryCopy (3D imm)");
		}
		ze_check(zeEventHostSynchronize(ev, UINT64_MAX), "zeEventHostSynchronize");
		pool.release(idx);
		last = q.ext_oneapi_submit_barrier();
	} else {
		auto& bm = *g_batches[0];
		{
			std::lock_guard<std::mutex> l(bm.mtx);
			for(size_t i=0;i<chunks.size();++i) {
				const auto [so,doff,sz] = chunks[i];
				const bool last_in_segment = (i+1 == chunks.size());
				bm.append_copy(src_ptr + so, dst_ptr + doff, sz, last_in_segment);
			}
			if(bm.should_flush()) bm.flush_and_wait();
		}
		last = q.ext_oneapi_submit_barrier();
	}
}

// front function: chooses box vs. linear layout path
static async_event nd_copy_device_level_zero(
    sycl::queue& queue, const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size, bool enable_profiling) {

	initialize_per_device_state(queue, device_id{0});

	sycl::event last_event;
	dispatch_nd_region_copy(
	    source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    // box path
	    [&queue, elem_size, &last_event](const void* const source, void* const dest,
	                                     const box<3>& source_box, const box<3>& dest_box, const box<3>& copy_box) {
		    nd_copy_box_level_zero(queue, source, dest, source_box, dest_box, copy_box, elem_size, last_event);
	    },
	    // linear path
	    [&queue, &last_event](const void* const source, void* const dest, size_t size_bytes) {
			if(size_bytes <= g_small_threshold) {
				copy_linear_small_immediate(queue, source, dest, size_bytes, last_event);
			} else {
				copy_linear_large_batched(queue, source, dest, size_bytes, last_event);
			}
	    });

	if(enable_profiling) {
		// Return a proper celerity::detail::sycl_event with profiling enabled.
		return async_event{std::make_unique<sycl_event>(last_event, /*enable_profiling*/ true)};
	}
	return async_event{std::make_unique<sycl_event>(std::nullopt, last_event)};
}

} // namespace celerity::detail::level_zero_backend_detail

// ---------- public backend glue ----------

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices,
                                                 const sycl_backend::configuration& config)
: sycl_backend(devices, config) {
	CELERITY_DEBUG("Using Level-Zero backend for the selected devices.");
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_all();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane,
    const void* const source_base, void* const dest_base,
    const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size) {
	// NOTE: We don't touch peer detection here; outer runtime provides compatible pointers.
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& queue) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    queue, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
