<!-- cargo-rdme start -->

# audio-block

Real-time safe abstractions over audio data with support for all common layouts.

## Quick Start

Install:
```sh
cargo add audio-block
```

Basic planar usage (most common for DSP):
```rust
use audio_block::*;

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

### Owned Blocks

```rust
use audio_block::*;

// Allocate with default values (zero)
let mut block = Planar::<f32>::new(2, 512);       // 2 channels, 512 frames
let mut block = Sequential::<f32>::new(2, 512);   // 2 channels, 512 frames
let mut block = Interleaved::<f32>::new(2, 512);  // 2 channels, 512 frames
let mut block = Mono::<f32>::new(512);            // 512 frames

// Copy from existing data
let channel_data = vec![[0.0f32; 512], [0.0f32; 512]];
let data = vec![0.0f32; 1024];
let mut block = Planar::from_slice(&channel_data);   // channels derived from slice
let mut block = Sequential::from_slice(&data, 2);   // 2 channels
let mut block = Interleaved::from_slice(&data, 2);  // 2 channels
let mut block = Mono::from_slice(&data);
```

Allocation only happens when creating owned blocks. Never do that in real-time contexts.

### Views (zero-allocation, borrows data)

```rust
use audio_block::*;

let channel_data = vec![[0.0f32; 512], [0.0f32; 512]];
let data = vec![0.0f32; 1024];

let block = PlanarView::from_slice(&channel_data);   // channels derived from slice
let block = SequentialView::from_slice(&data, 2);   // 2 channels
let block = InterleavedView::from_slice(&data, 2);  // 2 channels
let block = MonoView::from_slice(&data);
```

From raw pointers:
```rust
let data = vec![0.0f32; 1024];
let block = unsafe { InterleavedView::from_ptr(ptr, 2, 512) }; // 2 channels, 512 frames
```

Planar requires adapter:
```rust
let mut adapter = unsafe { PlanarPtrAdapter::<_, 16>::from_ptr(data, 2, 512) }; // 2 channels, 512 frames
let block = adapter.planar_view();
```

## Common Operations

Import the extension traits for additional operations:

```rust
use audio_block::{AudioBlockOps, AudioBlockOpsMut};
```

### Copying and Clearing

```rust
let other_block = Planar::<f32>::new(2, 512);
let mut block = Planar::<f32>::new(2, 512);

// Copy from another block (flexible - copies min of both sizes)
let result = block.copy_from_block(&other_block);
// Returns None if exact match, Some((channels, frames)) if partial

// Copy with exact size requirement (panics on mismatch)
block.copy_from_block_exact(&other_block);

// Fill all samples with a value
block.fill_with(0.5);

// Clear to zero
block.clear();
```

### Per-Sample Processing

```rust
let mut block = Planar::<f32>::new(2, 512);

// Process each sample
block.for_each(|sample| *sample *= 0.5);

// Process with channel/frame indices
block.enumerate(|channel, frame, sample| {
    *sample *= 0.5;
});

// Apply gain to all samples
block.gain(0.5);
```

### Mono Conversions

```rust
let mut block = Planar::<f32>::new(2, 512);
let mut mono_data = vec![0.0f32; 512];
let mut mono_view = MonoViewMut::from_slice(&mut mono_data);

// Mix all channels to mono (averages channels)
let result = block.mix_to_mono(&mut mono_view);
// Returns None if exact match, Some(frames_processed) if partial

// Or with exact size requirement
block.mix_to_mono_exact(&mut mono_view);

// Copy a specific channel to mono
block.copy_channel_to_mono(&mut mono_view, 0); // channel 0

// Copy mono to all channels of a block
let mono_ro = MonoView::from_slice(&mono_data);
block.copy_mono_to_all_channels(&mono_ro);
```

## Working with Slices

Convert generic blocks to concrete types for slice access:

```rust
fn process(block: &mut impl AudioBlockMut<f32>) {
    if block.layout() == BlockLayout::Planar {
        let mut view = block.as_planar_view_mut().unwrap();
        let ch0: &mut [f32] = view.channel_mut(0);
        let ch1: &mut [f32] = view.channel_mut(1);
    }
}
```

Direct slice access on concrete types:

```rust
let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames
let channel: &[f32] = block.channel(0);
let raw_data: &[Box<[f32]>] = block.raw_data();

let mut block = Interleaved::<f32>::new(2, 512); // 2 channels, 512 frames
let frame: &[f32] = block.frame(0);
let raw_data: &[f32] = block.raw_data();
```

## Trait API Reference

### `AudioBlock`

Size and layout:
```rust
let channels: u16 = audio.num_channels();
let frames: usize = audio.num_frames();
let layout: BlockLayout = audio.layout();
```

Sample access:
```rust
let s: f32 = audio.sample(0, 0);
```

Iteration:
```rust
for s in audio.channel_iter(0) { let _: &f32 = s; }
for ch in audio.channels_iter() { for s in ch { let _: &f32 = s; } }
for s in audio.frame_iter(0) { let _: &f32 = s; }
for fr in audio.frames_iter() { for s in fr { let _: &f32 = s; } }
```

Generic view (zero-allocation):
```rust
let view = audio.as_view();
```

Downcast to concrete type:
```rust
let interleaved: Option<InterleavedView<f32>> = audio.as_interleaved_view();
let sequential: Option<SequentialView<f32>> = audio.as_sequential_view();
```

### `AudioBlockMut`

Everything from `AudioBlock` plus:

Resizing:
```rust
audio.set_visible(2, 1024);
audio.set_num_channels_visible(2);
audio.set_num_frames_visible(1024);
```

Mutable access:
```rust
let s: &mut f32 = audio.sample_mut(0, 0);
for s in audio.channel_iter_mut(0) { let _: &mut f32 = s; }
for ch in audio.channels_iter_mut() { for s in ch { let _: &mut f32 = s; } }
for s in audio.frame_iter_mut(0) { let _: &mut f32 = s; }
for fr in audio.frames_iter_mut() { for s in fr { let _: &mut f32 = s; } }
```

Generic view (zero-allocation):
```rust
let view = audio.as_view_mut();
```

Downcast to concrete type:
```rust
let interleaved: Option<InterleavedViewMut<f32>> = audio.as_interleaved_view_mut();
let sequential: Option<SequentialViewMut<f32>> = audio.as_sequential_view_mut();
```

### `AudioBlockOps` (extension trait)

Read-only operations on audio blocks:
```rust
let _: Option<usize> = block.mix_to_mono(dest);
block.mix_to_mono_exact(dest);
let _: Option<usize> = block.copy_channel_to_mono(dest, 0);
block.copy_channel_to_mono_exact(dest, 0);
```

### `AudioBlockOpsMut` (extension trait)

Mutable operations on audio blocks:
```rust
let _: Option<(u16, usize)> = block.copy_from_block(other);
block.copy_from_block_exact(other);
let _: Option<usize> = block.copy_mono_to_all_channels(mono);
block.copy_mono_to_all_channels_exact(mono);
block.for_each(|sample| *sample *= 0.5);
block.enumerate(|_ch, _fr, sample| { *sample *= 0.5; });
block.for_each_allocated(|sample| *sample *= 0.5);
block.fill_with(0.5);
block.clear();
block.gain(0.5);
```

## Advanced: Variable Buffer Sizes

Blocks separate allocated capacity from visible size. Resize visible portion without reallocation:

```rust
let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames
block.set_num_frames_visible(256); // use only 256 frames
```

Create views with limited visibility:
```rust
let block = InterleavedView::from_slice_limited(
    &data,
    2,   // num_channels_visible
    256, // num_frames_visible
    2,   // num_channels_allocated
    512  // num_frames_allocated
);
```

Query allocation:
```rust
let _ = block.num_channels_allocated();
let _ = block.num_frames_allocated();
```

## Advanced: Access Allocated Samples

For operations that process all allocated memory (including non-visible samples):

```rust
use audio_block::AudioBlockOpsMut;

block.for_each_allocated(|sample| *sample *= 0.5);
block.enumerate_allocated(|_ch, _frame, sample| {
    // Process including allocated but non-visible samples
    let _ = sample;
});
```

Note: `fill_with`, `clear`, and `gain` also operate on the entire allocated buffer for efficiency.

Direct memory access:
```rust
let block = Sequential::<f32>::new(2, 512);
let data: &[f32] = block.raw_data();  // Includes non-visible samples
```

## Performance

Iterator performance varies by layout:
- Sequential/Planar: Channel iteration faster
- Interleaved (many channels): Frame iteration faster

`raw_data()` access is fastest but exposes non-visible samples. For simple operations like gain, processing all samples (including non-visible) can be more efficient.

Check layout before optimization:
```rust
match block.layout() {
    BlockLayout::Planar => { /* channel-wise processing */ }
    BlockLayout::Interleaved => { /* frame-wise processing */ }
    BlockLayout::Sequential => { /* channel-wise processing */ }
}
```

## `no_std` Support

Disable default features. Owned blocks require `alloc` or `std` feature.

<!-- cargo-rdme end -->
