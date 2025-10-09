# Level Zero Event Strategy - 3D Copy Fix

## The Question

**"In the 3D case, where am I adding the ze_event - at the beginning, at the end, or some other case?"**

## The Answer

**At the END (last chunk only)!** ✅

## Why This Matters

Level Zero events signal **completion**. We want to know when **ALL** copy operations are done, not when individual chunks complete.

## The Three Cases

### 1️⃣ **Contiguous Copy (0 strides)**
```cpp
// Single operation - signal event immediately
ze_check(zeCommandListAppendMemoryCopy(
    cmd_list, dst_ptr, src_ptr, layout.contiguous_size, 
    ze_event,  // ✅ Signal on this single operation
    0, nullptr
));
```

**Why**: Only one operation, so signal when it completes.

---

### 2️⃣ **2D Strided Copy (1 stride)**
```cpp
// Single 2D region copy - signal event immediately
ze_check(zeCommandListAppendMemoryCopyRegion(
    cmd_list, dst_ptr, &dst_region, dst_pitch, 0,
    src_ptr, &src_region, src_pitch, 0, 
    ze_event,  // ✅ Signal on this single 2D operation
    0, nullptr
));
```

**Why**: Hardware-optimized 2D copy is a single operation, so signal when it completes.

---

### 3️⃣ **3D Complex Copy (2 strides)** - THE TRICKY ONE

#### ❌ **WRONG Approach (Original Bug)**
```cpp
bool first = true;
for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
    ze_event_handle_t event_to_use = first ? nullptr : ze_event;  // ❌ WRONG!
    zeCommandListAppendMemoryCopy(cmd_list, dst, src, size, event_to_use, 0, nullptr);
    first = false;
});
```

**Problem**: This signals the event on **every chunk EXCEPT the first**!

**Example with 4 chunks:**
```
Chunk 0: event = nullptr  ← No signal
Chunk 1: event = ze_event ← Signals! (but not done yet)
Chunk 2: event = ze_event ← Signals! (but not done yet)
Chunk 3: event = ze_event ← Signals! (finally done)
```

**Why this is bad:**
- Event signals **multiple times** (undefined behavior!)
- Event signals **before all chunks complete**
- Wastes GPU cycles signaling unnecessarily

---

#### ✅ **CORRECT Approach (Fixed)**
```cpp
// Collect all chunks first
std::vector<std::tuple<size_t, size_t, size_t>> chunks;
for_each_contiguous_chunk(layout, [&](size_t src_off, size_t dst_off, size_t size) {
    chunks.emplace_back(src_off, dst_off, size);
});

// Append all copies, signal event ONLY on the last one
for(size_t i = 0; i < chunks.size(); ++i) {
    const auto& [src_off, dst_off, size] = chunks[i];
    const void* src_ptr = static_cast<const char*>(source_base) + src_off;
    void* dst_ptr = static_cast<char*>(dest_base) + dst_off;
    
    // Signal event ONLY on the last chunk
    const bool is_last = (i == chunks.size() - 1);
    ze_event_handle_t event_to_use = is_last ? ze_event : nullptr;  // ✅ CORRECT!
    
    zeCommandListAppendMemoryCopy(cmd_list, dst_ptr, src_ptr, size, event_to_use, 0, nullptr);
}
```

**Example with 4 chunks:**
```
Chunk 0: event = nullptr  ← No signal
Chunk 1: event = nullptr  ← No signal
Chunk 2: event = nullptr  ← No signal
Chunk 3: event = ze_event ← Signals! (all done) ✅
```

**Why this is correct:**
- Event signals **exactly once**
- Event signals **only when ALL chunks complete**
- Minimal overhead - only one signal operation

---

## The Key Insight

### Level Zero Command Lists Execute In Order

When you append commands to a Level Zero command list:
```cpp
zeCommandListAppendMemoryCopy(cmd_list, ...);  // Command 1
zeCommandListAppendMemoryCopy(cmd_list, ...);  // Command 2
zeCommandListAppendMemoryCopy(cmd_list, ...);  // Command 3
zeCommandListAppendMemoryCopy(cmd_list, ..., ze_event, ...);  // Command 4 - signals event
```

The commands execute **in order**:
1. Command 1 completes
2. Command 2 completes
3. Command 3 completes
4. Command 4 completes **→ signals event** ✅

So if you signal the event on the **last command**, you know **all previous commands completed**!

---

## Why Not Signal on Every Chunk?

### Performance Cost
```cpp
// BAD: Signal on every chunk
for(each chunk) {
    zeCommandListAppendMemoryCopy(..., ze_event, ...);  // ❌ Expensive!
}
```

**Problems:**
- Each event signal has overhead
- GPU must synchronize after each chunk
- Prevents parallel execution of chunks
- Event can only be signaled once (undefined behavior to signal multiple times)

### Correct Approach
```cpp
// GOOD: Signal only on last chunk
for(each chunk except last) {
    zeCommandListAppendMemoryCopy(..., nullptr, ...);  // ✅ No overhead
}
zeCommandListAppendMemoryCopy(..., ze_event, ...);  // ✅ Signal once at end
```

**Benefits:**
- Minimal overhead (one signal)
- GPU can pipeline chunk operations
- Event signals exactly once
- Correct completion semantics

---

## Real-World Example

### 3D Copy: 4x4x4 cube
```
Total chunks: 16 (4 planes × 4 rows)

Chunk  0: Copy row 0 of plane 0  → event = nullptr
Chunk  1: Copy row 1 of plane 0  → event = nullptr
Chunk  2: Copy row 2 of plane 0  → event = nullptr
Chunk  3: Copy row 3 of plane 0  → event = nullptr
Chunk  4: Copy row 0 of plane 1  → event = nullptr
...
Chunk 14: Copy row 2 of plane 3  → event = nullptr
Chunk 15: Copy row 3 of plane 3  → event = ze_event ✅ (ALL DONE!)
```

**When the event signals**, you know:
- All 16 chunks copied successfully
- All data is in destination memory
- Safe to proceed with next operation

---

## Summary

| Case | Operations | Event Placement | Reason |
|------|-----------|-----------------|--------|
| **1D Contiguous** | 1 memcpy | On the single operation | Only one operation |
| **2D Strided** | 1 region copy | On the single operation | Hardware-optimized single op |
| **3D Complex** | N memcpy | On the **LAST** operation only | Signals when ALL complete |

**Golden Rule**: Signal the event when **ALL operations complete**, not on individual operations.

**For 3D copies**: Signal on the **last chunk** because that's when all chunks are done! 🎯
