### Breaking Changes

- **`PlanarPtrAdapterMut::from_ptr` now takes `*const *mut S`** instead of `*mut *mut S`. The array of channel pointers is only read, so exclusive access to it is no longer required. Callers passing a `*mut *mut S` still compile via pointer coercion; those naming the type explicitly need updating.
- **`PlanarViewMut` now panics above `MAX_PLANAR_CHANNELS` (64) channels**, where previously there was no limit. Use the owned `Planar` type for higher channel counts.

### Bug Fixes

- **Unsound mutable frame iteration on planar blocks fixed**: `frames_iter_mut` on `Planar` / `PlanarViewMut` previously handed out frame iterators that aliased the same underlying storage, so holding two live frames at once was undefined behavior. Each yielded frame now borrows disjoint samples via a cached base-pointer table, so frames can be advanced, held, or collected freely regardless of layout. Verified sound under Miri (stacked borrows and tree borrows).
- **`enumerate_allocated` produced wrong channel/frame indices for interleaved blocks**: the channel/frame split was computed using `num_frames` instead of `num_channels`, so indices were wrong for any block where channels and frames differed. Fixed, with a regression test added.
- **`PlanarPtrAdapterMut::planar_view_mut` could only be called once per adapter.** It returned a view tied to the adapter's data lifetime rather than to the `&mut self` borrow, which left the adapter mutably borrowed for the rest of its life. Views can now be created repeatedly.
- **`PlanarPtrAdapterMut::from_ptr` documented safety requirements were incomplete**: overlapping channel pointers produce aliasing `&mut [S]` and are undefined behavior, which the docs did not state. Now documented.
- **`PlanarPtrAdapterMut::from_ptr` now rejects channel counts above `MAX_PLANAR_CHANNELS` immediately**, instead of accepting them and panicking later in `planar_view_mut`.

### New Features

- **`PlanarFrameIterMut` and `MAX_PLANAR_CHANNELS`**: new public items backing the sound mutable frame iteration above. `PlanarViewMut` now caps channels at `MAX_PLANAR_CHANNELS` (64) and panics if constructed with more; the owned `Planar` type keeps its table on the heap and has no such limit.

### Memory Use

- **`PlanarViewMut` is much larger than the other view types** (552 vs 40 bytes on 64-bit), because it caches one pointer per channel. If memory is tight, use `PlanarView` or a non-planar layout.

### Other Changes

- Fixed several doc comments that incorrectly referenced `SequentialView` in `InterleavedView` / `InterleavedViewMut` / `SequentialViewMut` safety docs.
- README and doc cleanup.
- CI: refreshed workflows, bumped actions, added `cargo-shear` and RTSan jobs, and improved Miri checks (stacked borrows and tree borrows).
