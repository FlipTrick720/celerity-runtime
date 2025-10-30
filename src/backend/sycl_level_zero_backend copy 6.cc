//Version: v6_unified_safe_fast
//Text: Event-pool + micro/thresholds + immediate for small + fence-batched large; L0-only; strict correctness.

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
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>
#include <vector>
#include <chrono>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/backend.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace celerity::detail::level_zero_backend_detail {

static inline void ze_check(ze_result_t r, const char* where) {
	if(r != ZE_RESULT_SUCCESS) utils::panic("Level-Zero error in {}: code={}", where, static_cast<int>(r));
}

// ---------- Env knobs with sane defaults ----------
static inline size_t env_or(const char* name, size_t def) {
	if(const char* v = std::getenv(name)) { try { return static_cast<size_t>(std::stoull(v)); } catch(...) {} }
	return def;
}
static inline bool env_or_bool(const char* name, bool def) {
	if(const char* v = std::getenv(name)) { return std::string(v) == "1" || std::string(v) == "true"; }
	return def;
}

static size_t g_pool_size             = env_or("CELERITY_L0_EVENT_POOL_SIZE", 512);
static size_t g_micro_threshold       = env_or("CELERITY_L0_MICRO_THRESHOLD", 256);    // bytes
static size_t g_small_threshold       = env_or("CELERITY_L0_SMALL_THRESHOLD", 4096);   // bytes
static bool   g_use_batching          = env_or_bool("CELERITY_L0_USE_BATCHING", true);
static size_t g_batch_threshold_ops   = env_or("CELERITY_L0_BATCH_THRESHOLD_OPS", 8);
static size_t g_batch_threshold_us    = env_or("CELERITY_L0_BATCH_THRESHOLD_US", 100);

// ---------- Event pool per device ----------
struct event_pool {
	ze_event_pool_handle_t pool = nullptr;
	std::vector<ze_event_handle_t> events;
	std::queue<size_t> free_idx;
	std::mutex mtx;

	void init(ze_context_handle_t ctx, ze_device_handle_t dev, size_t count) {
		ze_event_pool_desc_t d{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
		d.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
		d.count = static_cast<uint32_t>(count);
		ze_check(zeEventPoolCreate(ctx, &d, 1, &dev, &pool), "zeEventPoolCreate");

		events.resize(count);
		for(size_t i=0;i<count;++i){
			ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC};
			ed.index  = static_cast<uint32_t>(i);
			ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
			ed.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
			ze_check(zeEventCreate(pool, &ed, &events[i]), "zeEventCreate");
			free_idx.push(i);
		}
	}

	size_t acquire() {
		std::lock_guard lk(mtx);
		if(free_idx.empty()) utils::panic("L0 event pool exhausted (size={})", events.size());
		const auto i = free_idx.front(); free_idx.pop(); return i;
	}
	void release(size_t i) {
		std::lock_guard lk(mtx);
		free_idx.push(i);
	}

	void destroy() {
		if(!pool) return;
		for(auto& e: events) if(e) zeEventDestroy(e);
		zeEventPoolDestroy(pool);
		pool=nullptr; events.clear();
	}
};

// ---------- Batch manager per device ----------
struct batch_manager {
	ze_command_list_handle_t cl = nullptr;   // regular list for batching
	ze_fence_handle_t fence     = nullptr;   // queue fence to wait for batched work
	std::mutex mtx;
	size_t pending_ops = 0;
	std::chrono::steady_clock::time_point start;

	void init(ze_context_handle_t ctx, ze_device_handle_t dev) {
		ze_command_list_desc_t lcd{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
		ze_check(zeCommandListCreate(ctx, dev, &lcd, &cl), "zeCommandListCreate");
		// Fence will be created on first use with actual queue
		fence = nullptr;
	}

	void append_memcpy(const void* src, void* dst, size_t bytes) {
		if(!cl) return;
		if(pending_ops==0) start = std::chrono::steady_clock::now();
		ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, /*signal*/nullptr, 0, nullptr),
		         "zeCommandListAppendMemoryCopy");
		++pending_ops;
	}

	bool should_flush() const {
		if(pending_ops >= g_batch_threshold_ops) return true;
		if(pending_ops>0 && g_batch_threshold_us>0) {
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(
			              std::chrono::steady_clock::now() - start).count();
			if(static_cast<size_t>(us) >= g_batch_threshold_us) return true;
		}
		return false;
	}

	void flush_and_wait(ze_command_queue_handle_t q) {
		if(pending_ops==0) return;
		
		// Create fence on first use if needed
		if(!fence) {
			ze_fence_desc_t fd{ZE_STRUCTURE_TYPE_FENCE_DESC};
			ze_check(zeFenceCreate(q, &fd, &fence), "zeFenceCreate");
		}
		
		ze_check(zeCommandListClose(cl), "zeCommandListClose");
		ze_check(zeFenceReset(fence), "zeFenceReset");
		ze_check(zeCommandQueueExecuteCommandLists(q, 1, &cl, /*fence*/fence), "zeCommandQueueExecuteCommandLists");
		ze_check(zeFenceHostSynchronize(fence, UINT64_MAX), "zeFenceHostSynchronize");
		ze_check(zeCommandListReset(cl), "zeCommandListReset");
		pending_ops = 0;
	}

	void destroy() {
		if(!cl) return;
		if(fence) zeFenceDestroy(fence);
		zeCommandListDestroy(cl);
		cl=nullptr; fence=nullptr;
	}
};

// ---------- Per-device state ----------
struct device_state {
	event_pool pool;
	// immediate command list for small copies
	ze_command_list_handle_t imm = nullptr;
	// batch manager for large copies
	batch_manager batch;

	void init_small(ze_context_handle_t ctx, ze_device_handle_t dev) {
		// Query device properties to get compute queue ordinal
		uint32_t queue_group_count = 0;
		ze_check(zeDeviceGetCommandQueueGroupProperties(dev, &queue_group_count, nullptr), "zeDeviceGetCommandQueueGroupProperties");
		std::vector<ze_command_queue_group_properties_t> queue_props(queue_group_count);
		for(auto& prop : queue_props) {
			prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
			prop.pNext = nullptr;
		}
		ze_check(zeDeviceGetCommandQueueGroupProperties(dev, &queue_group_count, queue_props.data()), "zeDeviceGetCommandQueueGroupProperties");
		
		// Find compute queue ordinal
		uint32_t compute_ordinal = 0;
		for(uint32_t i = 0; i < queue_group_count; ++i) {
			if(queue_props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
				compute_ordinal = i;
				break;
			}
		}
		
		ze_command_queue_desc_t qd{};
		qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
		qd.pNext = nullptr;
		qd.ordinal = compute_ordinal;
		qd.index = 0;
		qd.flags = 0;
		qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;  // Use async mode
		qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
		ze_check(zeCommandListCreateImmediate(ctx, dev, &qd, &imm), "zeCommandListCreateImmediate");
		batch.init(ctx, dev);
	}

	void destroy() {
		// Note: Cannot flush batch here as we don't have queue handle
		// Batch should be flushed before destroy is called
		batch.destroy();
		if(imm) zeCommandListDestroy(imm);
		pool.destroy();
	}
};

// ---------- Helpers ----------
static inline std::pair<ze_command_queue_handle_t, ze_context_handle_t>
get_native_q_and_ctx(sycl::queue& q) {
	auto nq = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
	auto zeq = std::get<ze_command_queue_handle_t>(nq);
	auto zectx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
	return {zeq, zectx};
}

// cpu memcpy fallback for micro copies
static inline void memcpy_micro(void* dst, const void* src, size_t n) {
	std::memcpy(dst, src, n);
}

// ---------- Main copy primitive (contiguous only here; ND dispatched by layout) ----------
class l0_copy_engine {
  public:
	l0_copy_engine(std::vector<std::unique_ptr<device_state>>* per_dev) : m_per_dev(per_dev) {}

	// Must be called on the submission thread of the backend, queue already selected
	sycl::event copy_contiguous(sycl::queue& sq, device_id dev_id,
	                            const void* src, void* dst, size_t bytes, bool profiling) {
		if(bytes==0) {
			return sq.ext_oneapi_submit_barrier();
		}
		if(bytes <= g_micro_threshold) {
			memcpy_micro(dst, src, bytes);
			return sq.ext_oneapi_submit_barrier(); // tie into SYCL dep graph
		}

		const auto [zeq, zectx] = get_native_q_and_ctx(sq);
		auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sq.get_device());

		auto& st = *(*m_per_dev)[dev_id];

		if(bytes <= g_small_threshold) {
			// immediate list path (lowest latency)
			// Use async immediate list but synchronize explicitly
			ze_check(zeCommandListAppendMemoryCopy(st.imm, dst, src, bytes, nullptr, 0, nullptr),
			         "zeCommandListAppendMemoryCopy[imm]");
			// Synchronize the immediate command list to ensure completion
			ze_check(zeCommandListHostSynchronize(st.imm, UINT64_MAX), "zeCommandListHostSynchronize");
			return sq.ext_oneapi_submit_barrier();
		}

		if(g_use_batching) {
			{
				std::lock_guard lk(st.batch.mtx);
				st.batch.append_memcpy(src, dst, bytes);
				// For correctness: always flush to ensure completion before returning
				// This ensures the copy is done before the barrier event is used
				st.batch.flush_and_wait(zeq);
			}
			return sq.ext_oneapi_submit_barrier();
		}

		// fallback: single regular list submit + fence to guarantee completion
		ze_command_list_handle_t cl{};
		ze_command_list_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
		ze_check(zeCommandListCreate(zectx, zedev, &d, &cl), "zeCommandListCreate");
		ze_check(zeCommandListAppendMemoryCopy(cl, dst, src, bytes, nullptr, 0, nullptr),
		         "zeCommandListAppendMemoryCopy");
		ze_check(zeCommandListClose(cl), "zeCommandListClose");
		ze_fence_desc_t fd{ZE_STRUCTURE_TYPE_FENCE_DESC};
		ze_fence_handle_t f{};
		ze_check(zeFenceCreate(zeq, &fd, &f), "zeFenceCreate");
		ze_check(zeCommandQueueExecuteCommandLists(zeq, 1, &cl, f), "zeCommandQueueExecuteCommandLists");
		ze_check(zeFenceHostSynchronize(f, UINT64_MAX), "zeFenceHostSynchronize");
		zeFenceDestroy(f);
		zeCommandListDestroy(cl);
		return sq.ext_oneapi_submit_barrier();
	}

  private:
	std::vector<std::unique_ptr<device_state>>* m_per_dev;
};

// ND dispatcher using existing layout helpers
template <typename SubmitContig>
static inline void dispatch_nd(const void* src_base, void* dst_base,
                               const region_layout& src_layout, const region_layout& dst_layout,
                               const region<3>& copy_region, size_t elem_size,
                               SubmitContig&& submit) {
	using namespace celerity::detail;
	dispatch_nd_region_copy(
	    src_base, dst_base, src_layout, dst_layout, copy_region, elem_size,
	    [&](const void* src, void* dst, const box<3>& src_box, const box<3>& dst_box, const box<3>& box_copy) {
		    const auto layout = layout_nd_copy(src_box.get_range(), dst_box.get_range(),
		                                      box_copy.get_offset() - src_box.get_offset(),
		                                      box_copy.get_offset() - dst_box.get_offset(),
		                                      box_copy.get_range(), elem_size);
		    if(layout.contiguous_size == 0) return;
		    submit(static_cast<const std::byte*>(src) + layout.offset_in_source,
		           static_cast<std::byte*>(dst) + layout.offset_in_dest,
		           layout.contiguous_size);
	    },
	    [&](const void* src, void* dst, size_t bytes) { submit(src, dst, bytes); });
}

// Public entry used by backend
static async_event nd_copy_device_level_zero(sycl::queue& sq, device_id dev_id,
    const void* src_base, void* dst_base, const region_layout& src_layout,
    const region_layout& dst_layout, const region<3>& copy_region, size_t elem_size, bool profiling,
    l0_copy_engine& engine) {

	std::optional<sycl::event> first; // we can fill this if profiling later
	sycl::event last = sq.ext_oneapi_submit_barrier(); // seed

	dispatch_nd(src_base, dst_base, src_layout, dst_layout, copy_region, elem_size,
	    [&](const void* src, void* dst, size_t bytes){
		    last = engine.copy_contiguous(sq, dev_id, src, dst, bytes, profiling);
		    if(profiling && !first) first = last;
	    });

	sycl_backend_detail::flush(sq);
	return make_async_event<sycl_backend_detail::sycl_event>(std::move(first), std::move(last));
}

// ---------- Global state ----------
static std::vector<std::unique_ptr<device_state>> g_states;

static void initialize_all(const std::vector<sycl::device>& devices, ze_context_handle_t zectx) {
	g_states.clear();
	g_states.reserve(devices.size());
	for(size_t i=0;i<devices.size();++i){
		auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
		auto st = std::make_unique<device_state>();
		st->pool.init(zectx, zedev, g_pool_size);
		st->init_small(zectx, zedev);
		g_states.push_back(std::move(st));
	}
}

static void cleanup_all() {
	for(auto& s : g_states) s->destroy();
	g_states.clear();
}

} // namespace celerity::detail::level_zero_backend_detail

namespace celerity::detail {

sycl_level_zero_backend::sycl_level_zero_backend(const std::vector<sycl::device>& devices, const sycl_backend::configuration& config)
: sycl_backend(devices, config) {
	auto zectx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
	                 devices[0].get_platform().ext_oneapi_get_default_context());
	level_zero_backend_detail::initialize_all(devices, zectx);

	// peer detection (unchanged logic)
	for(device_id i=0;i<devices.size();++i){
		for(device_id j=i+1;j<devices.size();++j){
			try{
				const auto di = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[i]);
				const auto dj = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(devices[j]);
				ze_bool_t ij=false, ji=false;
				ze_result_t rij = zeDeviceCanAccessPeer(di, dj, &ij);
				ze_result_t rji = zeDeviceCanAccessPeer(dj, di, &ji);
				if(rij==ZE_RESULT_SUCCESS && rji==ZE_RESULT_SUCCESS && ij && ji){
					const memory_id mi = first_device_memory_id + i;
					const memory_id mj = first_device_memory_id + j;
					get_system_info().memories[mi].copy_peers.set(mj);
					get_system_info().memories[mj].copy_peers.set(mi);
				}
			} catch(const std::exception& e){
				CELERITY_WARN("Level-Zero: peer access query failed: {}", e.what());
			}
		}
	}
}

sycl_level_zero_backend::~sycl_level_zero_backend() {
	level_zero_backend_detail::cleanup_all();
}

async_event sycl_level_zero_backend::enqueue_device_copy(device_id device, size_t device_lane,
    const void* src_base, void* dst_base, const region_layout& src_layout, const region_layout& dst_layout,
    const region<3>& copy_region, size_t elem_size) {
	return enqueue_device_work(device, device_lane, [=, this](sycl::queue& q) {
		using namespace level_zero_backend_detail;
		static l0_copy_engine engine(&g_states);
		return nd_copy_device_level_zero(q, device, src_base, dst_base, src_layout, dst_layout, copy_region, elem_size, is_profiling_enabled(), engine);
	});
}

} // namespace celerity::detail
