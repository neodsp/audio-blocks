<!-- cargo-rdme start -->

# audio-blocks

Real-time safe abstractions over audio data with support for all common layouts.

## Quick Start

Install:
```sh
cargo add audio-blocks
```

Basic planar usage (most common for DSP):
```rust
use audio_blocks::*;

// Create a planar block - each channel gets its own buffer
let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames

// Process per channel
for channel in block.channels_mut() {
    for sample in channel {
        *sample *= 0.5;
    }
}
```

Generic function that accepts any layout:
```rust
fn process(block: &mut impl AudioBlockMut<f32>) {
    for channel in block.channels_iter_mut() {
        for sample in channel {
            *sample *= 0.5;
        }
    }
}
```

## Block Types

Three multi-channel layouts supported:

**Planar** - `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
Each channel has its own separate buffer. Standard for real-time DSP. Optimal for SIMD/vectorization.

**Sequential** - `[ch0, ch0, ch0, ch1, ch1, ch1]`
Single contiguous buffer with all samples for channel 0, then all samples for channel 1. Channel-contiguous in one allocation.

**Interleaved** - `[ch0, ch1, ch0, ch1, ch0, ch1]`
Channels alternate sample-by-sample. Common in audio APIs and hardware interfaces.

Plus a dedicated mono type:

**Mono** - `[sample0, sample1, sample2, ...]`
Simplified single-channel block with a streamlined API that doesn't require channel indexing.

## Core Traits

Write functions that accept any layout:

```rust
fn process(block: &mut impl AudioBlockMut<f32>) {
    // Works with planar, sequential, or interleaved
}
```

Generic across float types:

```rust
fn process<F: Copy + 'static + std::ops::MulAssign>(block: &mut impl AudioBlockMut<F>) {
    let gain: F = todo!();
    for channel in block.channels_iter_mut() {
        for sample in channel {
            *sample *= gain;
        }
    }
}
```

## Creating Blocks

All block types provide `new(channels, frames)` for owned allocation and
`from_slice()` to copy from existing data. Views borrow data without allocation.

**Owned** (allocates):
- [`Planar::new`], [`Sequential::new`], [`Interleaved::new`], [`Mono::new`]
- [`Planar::from_slice`], [`Sequential::from_slice`], [`Interleaved::from_slice`], [`Mono::from_slice`]

**Views** (zero-allocation, borrows data):
- [`PlanarView::from_slice`], [`SequentialView::from_slice`], [`InterleavedView::from_slice`], [`MonoView::from_slice`]
- `from_ptr` for raw pointer access, [`PlanarPtrAdapter`] for planar pointers

## Trait API

### [`AudioBlock`] — read-only access

- `num_channels()`, `num_frames()`, `layout()`
- `sample(channel, frame)` — direct sample access
- `channel_iter(ch)`, `channels_iter()` — per-channel iteration
- `frame_iter(fr)`, `frames_iter()` — per-frame iteration
- `as_view()` — zero-allocation view
- `as_interleaved_view()`, `as_planar_view()`, `as_sequential_view()` — downcast to concrete type

### [`AudioBlockMut`] — mutable access

Everything from `AudioBlock` plus:
- `sample_mut(channel, frame)` — mutable sample access
- `channel_iter_mut(ch)`, `channels_iter_mut()` — mutable per-channel iteration
- `frame_iter_mut(fr)`, `frames_iter_mut()` — mutable per-frame iteration
- `set_visible(channels, frames)`, `set_num_channels_visible()`, `set_num_frames_visible()` — resize without reallocation
- `as_view_mut()` — zero-allocation mutable view
- `as_interleaved_view_mut()`, `as_planar_view_mut()`, `as_sequential_view_mut()`

### [`AudioBlockOps`] — read-only operations (extension trait)

- `mix_to_mono()` / `mix_to_mono_exact()` — average all channels to mono
- `copy_channel_to_mono()` / `copy_channel_to_mono_exact()` — extract a single channel

### [`AudioBlockOpsMut`] — mutable operations (extension trait)

- `copy_from_block()` / `copy_from_block_exact()` — copy from another block
- `copy_mono_to_all_channels()` / `copy_mono_to_all_channels_exact()` — broadcast mono
- `for_each(fn)`, `enumerate(fn)` — per-sample processing
- `for_each_allocated(fn)`, `enumerate_allocated(fn)` — process all allocated samples (including non-visible)
- `fill_with(value)`, `clear()`, `gain(factor)` — bulk operations

## Variable Buffer Sizes

Blocks separate allocated capacity from visible size. Resize the visible portion
without reallocation via `set_num_frames_visible()` / `set_num_channels_visible()`.
Views support this too via `from_slice_limited()`.

## Performance

- Sequential/Planar: channel iteration is faster
- Interleaved: frame iteration is faster
- `raw_data()` gives direct slice access (includes non-visible samples)
- `fill_with`, `clear`, and `gain` operate on the entire allocated buffer

## `no_std` Support

Disable default features. Owned blocks require `alloc` or `std` feature.

<!-- cargo-rdme end -->
