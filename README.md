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

## Creating Blocks

Each block type provides `new(channels, frames)` for owned allocation and
`from_slice()` to copy from existing data. Allocation only happens when creating
owned blocks — never do this in real-time contexts.

| Owned (allocates) | View (borrows data) |
|---|---|
| [`Planar`] | [`PlanarView`] / [`PlanarViewMut`] |
| [`Sequential`] | [`SequentialView`] / [`SequentialViewMut`] |
| [`Interleaved`] | [`InterleavedView`] / [`InterleavedViewMut`] |
| [`Mono`] | [`MonoView`] / [`MonoViewMut`] |

Views can also be created from raw pointers (`from_ptr`). For planar pointer data,
use [`PlanarPtrAdapter`].

## Traits

Use `impl AudioBlock<f32>` / `impl AudioBlockMut<f32>` to write layout-generic functions
(as shown above). These traits are also generic over the sample type (`f32`, `f64`, `i16`, etc.).

| Trait | Purpose |
|---|---|
| [`AudioBlock`] | Read-only access: sample access, channel/frame iteration, layout info |
| [`AudioBlockMut`] | Mutable access: sample mutation, resizing visible region, mutable iteration |
| [`AudioBlockOps`] | Read-only operations: mono mixdown, channel extraction |
| [`AudioBlockOpsMut`] | Mutable operations: block copy, gain, clear, fill, per-sample processing |

Blocks also separate allocated capacity from visible size — see [`AudioBlockMut::set_num_frames_visible`]
for real-time safe buffer resizing without reallocation.

## `no_std` Support

Disable default features. Owned blocks require `alloc` or `std` feature.

<!-- cargo-rdme end -->
