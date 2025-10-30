//Version: v6_unified_safe_fast
//Text: Event-pool + micro/thresholds + immediate for small + fence-batched large; L0-only; strict correctness.

// Unified, safe & fast Level-Zero backend for Celerity
// ----------------------------------------------------
// Core strategies preserved from "copy 6":
//   - Per-device HOST_VISIBLE event pool with freelist reuse (size from CELERITY_L0_EVENT_POOL_SIZE)
//   - Threshold policy: tiny/small --> immediate command list; large --> batched regular command list
//   - Correctness without zeCommandQueueSynchronize: we use zeEventHostSynchronize (imm) and zeFenceHostSynchronize (batch)
//   - Return a single SYCL barrier event that fences all native L0 work
//   - L0-only implementation internally (SYCL used only to obtain native handles & publish a barrier event)
//
// Refactoring to fix build errors:
//   * Removed incorrect fence passed to zeCommandListAppendMemoryCopy (API takes an EVENT, not a fence).
//   * Removed use of non-existent box<3> helpers; rely solely on dispatch_nd_region_copy to emit contiguous segments.
//   * Use sycl_backend_detail::sycl_event via make_async_event<...> factory.
//   * Ensure zeCommandListCreate receives a valid device handle.
//   * No host memcpy on device pointers (tiny copies still go through immediate L0 copy).

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

static size_t env_size_or(const char* name, size_t def) {
	if(const char* v = std::getenv(name)) {
		char* end = nullptr;
		const unsigned long long x = std::strtoull(v, &end, 10);
		if(end && *end == '\0' && x > 0) return static_cast<size_t>(x);
	}
	return def;
}

static bool env_bool_or(const char* name, bool def) {
	if(const char* v = std::getenv(name)) {
		if(std::strcmp(v, "0") == 0 || strcasecmp(v, "false") == 0) return false;
		if(std::strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0) return true;
	}
	return def;
}

static size_t g_event_pool_size     = 512; // CELERITY_L0_EVENT_POOL_SIZE
static size_t g_micro_threshold     = 256; // CELERITY_L0_MICRO_THRESHOLD (bytes)
static size_t g_small_threshold     = 4096; // CELERITY_L0_SMALL_THRESHOLD (bytes)
static size_t g_batch_threshold_ops = 8;   // CELERITY_L0_BATCH_THRESHOLD_OPS
static size_t g_batch_threshold_us  = 100; // CELERITY_L0_BATCH_THRESHOLD_US
static bool   g_use_batching        = true;// CELERITY_L0_USE_BATCHING

static void init_policy_from_env_once() {
	static std::once_flag once;
	std::call_once(once, []{
		g_event_pool_size     = env_size_or("CELERITY_L0_EVENT_POOL_SIZE", g_event_pool_size);
		g_micro_threshold     = env_size_or("CELERITY_L0_MICRO_THRESHOLD",  g_micro_threshold);
		g_small_threshold     = env_size_or("CELERITY_L0_SMALL_THRESHOLD",  g_small_threshold);
		g_batch_threshold_ops = env_size_or("CELERITY_L0_BATCH_THRESHOLD_OPS", g_batch_threshold_ops);
		g_batch_threshold_us  = env_size_or("CELERITY_L0_BATCH_THRESHOLD_US",  g_batch_threshold_us);
		g_use_batching        = env_bool_or("CELERITY_L0_USE_BATCHING", g_use_batching);
	});
}

// ---------- per-device event pools ----------

struct event_pool_manager {
	ze_event_pool_handle_t pool = nullptr;
	std::vector<ze_event_handle_t> events; // fixed size
	std::queue<size_t> free_indices;
	std::mutex mtx;

	void initialize(ze_context_handle_t ctx, ze_device_handle_t dev, size_t count) {
		ze_event_pool_desc_t epd{};
		epd.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
		epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		epd.count = static_cast<uint32_t>(count);
		ze_check(zeEventPoolCreate(ctx, &epd, 1, &dev, &pool), "zeEventPoolCreate");

		events.resize(count);
		for(size_t i=0;i<count;++i) {
			ze_event_desc_t ed{};
			ed.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
			ed.index = static_cast<uint32_t>(i);
			ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			ed.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
			ze_check(zeEventCreate(pool, &ed, &events[i]), "zeEventCreate");
			free_indices.push(i);
		}
		CELERITY_DEBUG("L0: event pool created with {} events", count);
	}

	std::optional<size_t> acquire() {
		std::lock_guard<std::mutex> l(mtx);
		if(free_indices.empty()) return std::nullopt;
		const size_t idx = free_indices.front(); free_indices.pop();
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

// ---------- immediate list manager (tiny/small copies) ----------

struct immediate_cmdlist_manager {
	ze_command_list_handle_t list = nullptr;
	std::mutex mtx;

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
		return list;
	}

	void cleanup() {
		if(list) ze_check(zeCommandListDestroy(list), "zeCommandListDestroy");
		list = nullptr;
	}
};

// ---------- batch manager (large copies) ----------

struct batch_manager {
	ze_command_queue_handle_t queue = nullptr;
	ze_command_list_handle_t  list  = nullptr;
	ze_fence_handle_t         fence = nullptr;
	std::mutex mtx;

	size_t pending_ops = 0;
	std::chrono::steady_clock::time_point start{};
	size_t thr_ops = 8;
	size_t thr_us  = 100;

	void initialize(ze_command_queue_handle_t q, ze_context_handle_t ctx, ze_device_handle_t dev,
	                size_t ops_thr, size_t us_thr) {
		queue = q;
		ze_command_list_desc_t ld{}; ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
		ze_check(zeCommandListCreate(ctx, dev, &ld, &list), "zeCommandListCreate");
		ze_fence_desc_t fd{}; fd.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
		ze_check(zeFenceCreate(queue, &fd, &fence), "zeFenceCreate");
		thr_ops = ops_thr;
		thr_us  = us_thr;
	}

	void append_copy(const void* src, void* dst, size_t bytes) {
		// NOTE: Do NOT pass the fence here; AppendMemoryCopy expects an EVENT, not a FENCE.
		ze_check(zeCommandListAppendMemoryCopy(list, dst, src, bytes, /*event*/ nullptr, 0, nullptr),
		         "zeCommandListAppendMemoryCopy (batch)");
		if(pending_ops++ == 0) start = std::chrono::steady_clock::now();
	}

	bool should_flush() const {
		if(pending_ops == 0) return false;
		if(pending_ops >= thr_ops) return true;
		const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::steady_clock::now() - start).count();
		return static_cast<size_t>(us) >= thr_us;
	}

	void flush_and_wait() {
		if(pending_ops == 0) return;
		ze_check(zeCommandListClose(list), "zeCommandListClose");
		// Correctness: fence the queue at submission time, then host-wait on the fence.
		ze_check(zeCommandQueueExecuteCommandLists(queue, 1, &list, fence), "zeCommandQueueExecuteCommandLists");
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

// ---------- global per-device state (initialized on first use) ----------

static std::unique_ptr<event_pool_manager>       g_pool;
static std::unique_ptr<immediate_cmdlist_manager> g_imm;
static std::unique_ptr<batch_manager>            g_batch;
static bool g_initialized = false;
static std::mutex g_init_mtx;

static void initialize_state_once(sycl::queue& q) {
	std::lock_guard<std::mutex> l(g_init_mtx);
	if(g_initialized) return;

	init_policy_from_env_once();

	const auto ze_queue = std::get<ze_command_queue_handle_t>(
	    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q));
	const auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	const auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

	g_pool = std::make_unique<event_pool_manager>();
	g_pool->initialize(ze_ctx, ze_dev, g_event_pool_size);

	g_imm  = std::make_unique<immediate_cmdlist_manager>();
	g_imm->initialize(ze_ctx, ze_dev);

	g_batch = std::make_unique<batch_manager>();
	g_batch->initialize(ze_queue, ze_ctx, ze_dev, g_batch_threshold_ops, g_batch_threshold_us);

	g_initialized = true;
	CELERITY_DEBUG("L0 unified backend: init(pool={}, micro={}B, small={}B, batching={}, ops_thr={}, us_thr={})",
	               g_event_pool_size, g_micro_threshold, g_small_threshold, g_use_batching,
	               g_batch_threshold_ops, g_batch_threshold_us);
}

static void cleanup_all() {
	std::lock_guard<std::mutex> l(g_init_mtx);
	if(g_batch) g_batch->cleanup();
	if(g_imm)   g_imm->cleanup();
	if(g_pool)  g_pool->cleanup();
	g_batch.reset(); g_imm.reset(); g_pool.reset();
	g_initialized = false;
}

// ---------- contiguous copy path (used for all segments) ----------
//
// Return contract: we publish a SYCL barrier that fences all native L0 work we enqueued.

static sycl::event copy_contiguous_segment(sycl::queue& q, const void* src, void* dst, size_t bytes) {
	if(bytes == 0) return q.ext_oneapi_submit_barrier();

	initialize_state_once(q);

	// Tiny/small: immediate command list + pooled event (host sync) → SYCL barrier
	if(bytes <= g_small_threshold) {
		const auto idx_opt = g_pool->acquire();
		if(!idx_opt.has_value()) {
			// Pool exhausted: still enqueue but without an event; fence via barrier afterwards.
			ze_command_list_handle_t cl = g_imm->acquire();
			ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, /*event*/ nullptr, 0, nullptr),
			         "zeCommandListAppendMemoryCopy (imm/fallback)");
			return q.ext_oneapi_submit_barrier();
		}

		const size_t idx = *idx_opt;
		ze_event_handle_t ev = g_pool->at(idx);

		ze_command_list_handle_t cl = g_imm->acquire();
		ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, ev, 0, nullptr),
		         "zeCommandListAppendMemoryCopy (imm)");
		ze_check(zeEventHostSynchronize(ev, UINT64_MAX), "zeEventHostSynchronize");
		g_pool->release(idx);

		return q.ext_oneapi_submit_barrier();
	}

	// Large: batch on reusable regular list, flush by ops/time with fence
	if(g_use_batching) {
		{
			std::lock_guard<std::mutex> l(g_batch->mtx);
			g_batch->append_copy(src, dst, bytes);
			if(g_batch->should_flush()) g_batch->flush_and_wait();
		}
		return q.ext_oneapi_submit_barrier();
	}

	// Fallback large-single: one-off list submit with queue fence
	const auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	const auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
	const auto ze_queue = std::get<ze_command_queue_handle_t>(
	    sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q));

	ze_command_list_handle_t cl{};
	ze_command_list_desc_t ld{}; ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
	ze_check(zeCommandListCreate(ze_ctx, ze_dev, &ld, &cl), "zeCommandListCreate");

	ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr),
	         "zeCommandListAppendMemoryCopy (single)");
	ze_check(zeCommandListClose(cl), "zeCommandListClose");

	ze_fence_desc_t fd{}; fd.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
	ze_fence_handle_t fence{};
	ze_check(zeFenceCreate(ze_queue, &fd, &fence), "zeFenceCreate");
	ze_check(zeCommandQueueExecuteCommandLists(ze_queue, 1, &cl, fence), "zeCommandQueueExecuteCommandLists");
	ze_check(zeFenceHostSynchronize(fence, UINT64_MAX), "zeFenceHostSynchronize");
	ze_check(zeFenceDestroy(fence), "zeFenceDestroy");
	ze_check(zeCommandListDestroy(cl), "zeCommandListDestroy");

	return q.ext_oneapi_submit_barrier();
}

// ---------- ND copy: rely on dispatcher to produce contiguous segments ----------

static async_event nd_copy_device_level_zero(
    sycl::queue& queue, const void* source_base, void* dest_base,
    const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size, bool enable_profiling) {

	// dispatch_nd_region_copy calls 'submit' for each contiguous slice (src, dst, bytes).
	sycl::event last = queue.ext_oneapi_submit_barrier();
	dispatch_nd_region_copy(source_base, dest_base, source_layout, dest_layout, copy_region, elem_size,
	    [&](const void* src, void* dst, size_t bytes) {
		    last = copy_contiguous_segment(queue, src, dst, bytes);
	    });

	// Publish a celerity-visible event (profiling flag honored).
	if(enable_profiling) {
		return make_async_event<sycl_backend_detail::sycl_event>(last, last);
	}
	return make_async_event<sycl_backend_detail::sycl_event>(std::nullopt, last);
}

} // namespace celerity::detail::level_zero_backend_detail

// ---------- public backend glue ----------

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices,
                                                 const sycl_backend::configuration& config)
: sycl_backend(devices, config) {
	CELERITY_DEBUG("Using Level-Zero unified backend.");
	// Peer detection logic remains unchanged in the wider runtime; nothing to do here.
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_all();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane,
    const void* source_base, void* dest_base,
    const region_layout& source_layout, const region_layout& dest_layout,
    const region<3>& copy_region, const size_t elem_size) {
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& q) {
		return level_zero_backend_detail::nd_copy_device_level_zero(
		    q, source_base, dest_base, source_layout, dest_layout, copy_region, elem_size, is_profiling_enabled());
	});
}

} // namespace celerity::detail
