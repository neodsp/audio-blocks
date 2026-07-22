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

Owned blocks allocate via `new()` — never do this in real-time contexts.
Views borrow existing data via `from_slice()` or `from_ptr()` and are always real-time safe.

| Owned (allocates) | View (borrows data) |
|---|---|
| [`Planar`](https://docs.rs/audio-blocks/latest/audio_blocks/planar/owned/struct.Planar.html) | [`PlanarView`](https://docs.rs/audio-blocks/latest/audio_blocks/planar/view/struct.PlanarView.html) / [`PlanarViewMut`](https://docs.rs/audio-blocks/latest/audio_blocks/planar/view_mut/struct.PlanarViewMut.html) |
| [`Sequential`](https://docs.rs/audio-blocks/latest/audio_blocks/sequential/owned/struct.Sequential.html) | [`SequentialView`](https://docs.rs/audio-blocks/latest/audio_blocks/sequential/view/struct.SequentialView.html) / [`SequentialViewMut`](https://docs.rs/audio-blocks/latest/audio_blocks/sequential/view_mut/struct.SequentialViewMut.html) |
| [`Interleaved`](https://docs.rs/audio-blocks/latest/audio_blocks/interleaved/owned/struct.Interleaved.html) | [`InterleavedView`](https://docs.rs/audio-blocks/latest/audio_blocks/interleaved/view/struct.InterleavedView.html) / [`InterleavedViewMut`](https://docs.rs/audio-blocks/latest/audio_blocks/interleaved/view_mut/struct.InterleavedViewMut.html) |
| [`Mono`](https://docs.rs/audio-blocks/latest/audio_blocks/mono/owned/struct.Mono.html) | [`MonoView`](https://docs.rs/audio-blocks/latest/audio_blocks/mono/view/struct.MonoView.html) / [`MonoViewMut`](https://docs.rs/audio-blocks/latest/audio_blocks/mono/view_mut/struct.MonoViewMut.html) |

Views can also be created from raw pointers (`from_ptr`). For planar pointer data,
use [`PlanarPtrAdapter`](https://docs.rs/audio-blocks/latest/audio_blocks/planar/view/struct.PlanarPtrAdapter.html).

## Traits

Use `impl AudioBlock<f32>` / `impl AudioBlockMut<f32>` to write layout-generic functions
(as shown above). These traits are also generic over the sample type (`f32`, `f64`, `i16`, etc.).

| Trait | Purpose |
|---|---|
| [`AudioBlock`](https://docs.rs/audio-blocks/latest/audio_blocks/trait.AudioBlock.html) | Read-only access: sample access, channel/frame iteration, layout info |
| [`AudioBlockMut`](https://docs.rs/audio-blocks/latest/audio_blocks/trait.AudioBlockMut.html) | Mutable access: sample mutation, resizing visible region, mutable iteration |
| [`AudioBlockOps`](https://docs.rs/audio-blocks/latest/audio_blocks/ops/trait.AudioBlockOps.html) | Read-only operations: mono mixdown, channel extraction |
| [`AudioBlockOpsMut`](https://docs.rs/audio-blocks/latest/audio_blocks/ops/trait.AudioBlockOpsMut.html) | Mutable operations: block copy, gain, clear, fill, per-sample processing |

Blocks also separate allocated capacity from visible size — see [`AudioBlockMut::set_num_frames_visible`](https://docs.rs/audio-blocks/latest/audio_blocks/trait.AudioBlockMut.html#tymethod.set_num_frames_visible)
for real-time safe buffer resizing without reallocation.

## `no_std` Support

Disable default features. Owned blocks require `alloc` or `std` feature.

<!-- cargo-rdme end -->
