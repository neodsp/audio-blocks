# audio-blocks

Real-time safe abstractions over audio data with support for all common layouts.

## Quick Start

Install:
```sh
cargo add audio-blocks
```

Basic planar usage (most common for DSP):
```rust,ignore
use audio_blocks::*;

// Create a planar block - each channel gets its own buffer
let mut block = AudioBlockPlanar::<f32>::new(2, 512);

// Process per channel
for channel in block.channels_mut() {
    for sample in channel {
        *sample *= 0.5;
    }
}
```

Generic function that accepts any layout:
```rust,ignore
fn process(block: &mut impl AudioBlockMut<f32>) {
    for channel in block.channels_iter_mut() {
        for sample in channel {
            *sample *= 0.5;
        }
    }
}
```

## Block Types

Three layouts supported:

**Planar** - `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
Each channel has its own separate buffer. Standard for real-time DSP. Optimal for SIMD/vectorization.

**Sequential** - `[ch0, ch0, ch0, ch1, ch1, ch1]`
Single contiguous buffer with all samples for channel 0, then all samples for channel 1. Channel-contiguous in one allocation.

**Interleaved** - `[ch0, ch1, ch0, ch1, ch0, ch1]`
Channels alternate sample-by-sample. Common in audio APIs and hardware interfaces.

## Core Traits

Write functions that accept any layout:

```rust,ignore
fn process(block: &mut impl AudioBlockMut<f32>) {
    // Works with planar, sequential, or interleaved
}
```

Generic across float types:

```rust,ignore
fn process<F: num::Float + 'static>(block: &mut impl AudioBlockMut<F>) {
    let gain = F::from(0.5).unwrap();
    for channel in block.channels_iter_mut() {
        for sample in channel {
            *sample *= gain;
        }
    }
}
```

## Creating Blocks

### Owned Blocks

```rust,ignore
let mut block = AudioBlockPlanar::new(2, 512);
let mut block = AudioBlockSequential::new(2, 512);
let mut block = AudioBlockInterleaved::new(2, 512);
```

Allocation only happens here. Never create owned blocks in real-time contexts.

### Views (zero-allocation)

```rust,ignore
let block = AudioBlockPlanarView::from_slice(&data, 2, 512);
let block = AudioBlockSequentialView::from_slice(&data, 2, 512);
let block = AudioBlockInterleavedView::from_slice(&data, 2, 512);
```

From raw pointers:
```rust,ignore
let block = unsafe { AudioBlockInterleavedView::from_ptr(ptr, 2, 512) };
```

Planar requires adapter:
```rust,ignore
let mut adapter = unsafe { PlanarPtrAdapter::<_, 16>::from_ptr(data, 2, 512) };
let block = adapter.planar_view();
```

## Common Operations

```rust,ignore
use audio_blocks::AudioBlockOps;

block.copy_from_block(&other_block);
block.fill_with(0.0);
block.clear();
block.for_each(|sample| *sample *= 0.5);
```

## Working with Slices

Convert generic blocks to concrete types for slice access:

```rust,ignore
fn process(block: &mut impl AudioBlockMut<f32>) {
    if block.layout() == BlockLayout::Planar {
        let mut view = block.as_planar_view_mut().unwrap();
        let ch0: &mut [f32] = view.channel_mut(0);
        let ch1: &mut [f32] = view.channel_mut(1);
    }
}
```

Direct slice access on concrete types:

```rust,ignore
let mut block = AudioBlockPlanar::new(2, 512);
let channel: &[f32] = block.channel(0);
let raw_data: &[Box<[f32]>] = block.raw_data();

let mut block = AudioBlockInterleaved::new(2, 512);
let frame: &[f32] = block.frame(0);
let raw_data: &[f32] = block.raw_data();
```

## Trait API Reference

### `AudioBlock`

Size and layout:
```rust,ignore
fn num_channels(&self) -> u16;
fn num_frames(&self) -> usize;
fn layout(&self) -> BlockLayout;
```

Sample access:
```rust,ignore
fn sample(&self, channel: u16, frame: usize) -> S;
```

Iteration:
```rust,ignore
fn channel_iter(&self, channel: u16) -> impl Iterator<Item = &S>;
fn channels_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_>;
fn frame_iter(&self, frame: usize) -> impl Iterator<Item = &S>;
fn frames_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_>;
```

Generic view (zero-allocation):
```rust,ignore
fn as_view(&self) -> impl AudioBlock<S>;
```

Downcast to concrete type:
```rust,ignore
fn as_interleaved_view(&self) -> Option<AudioBlockInterleavedView<'_, S>>;
fn as_planar_view(&self) -> Option<AudioBlockPlanarView<'_, S, Self::PlanarView>>;
fn as_sequential_view(&self) -> Option<AudioBlockSequentialView<'_, S>>;
```

### `AudioBlockMut`

Everything from `AudioBlock` plus:

Resizing:
```rust,ignore
fn set_active_size(&mut self, num_channels: u16, num_frames: usize);
fn set_active_num_channels(&mut self, num_channels: u16);
fn set_active_num_frames(&mut self, num_frames: usize);
```

Mutable access:
```rust,ignore
fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;
fn channel_iter_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
fn channels_iter_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
fn frame_iter_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
fn frames_iter_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
```

Generic view (zero-allocation):
```rust,ignore
fn as_view_mut(&mut self) -> impl AudioBlockMut<S>;
```

Downcast to concrete type:
```rust,ignore
fn as_interleaved_view_mut(&mut self) -> Option<AudioBlockInterleavedViewMut<'_, S>>;
fn as_planar_view_mut(&mut self) -> Option<AudioBlockPlanarViewMut<'_, S, Self::PlanarView>>;
fn as_sequential_view_mut(&mut self) -> Option<AudioBlockSequentialViewMut<'_, S>>;
```

Operations:
```rust,ignore
fn copy_from_block(&mut self, block: &impl AudioBlock<S>);
fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>);
fn for_each(&mut self, f: impl FnMut(&mut S));
fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
fn fill_with(&mut self, sample: S);
fn clear(&mut self);
```

## Advanced: Variable Buffer Sizes

Blocks separate allocated capacity from visible size. Resize visible portion without reallocation:

```rust,ignore
let mut block = AudioBlockPlanar::new(2, 512);  // Allocate 512 frames
block.set_num_frames_visible(256);  // Use only 256
```

Create views with limited visibility:
```rust,ignore
let view = AudioBlockInterleavedView::from_slice_limited(
    data,
    2,    // visible channels
    256,  // visible frames
    2,    // allocated channels
    512   // allocated frames
);
```

Auto-resize when copying:
```rust,ignore
fn process(&mut self, input: &impl AudioBlock<f32>) {
    self.block.copy_from_block_resize(input);  // Adapts to input size
}
```

Query allocation:
```rust,ignore
block.num_channels_allocated();
block.num_frames_allocated();
```

## Advanced: Access Non-Visible Samples

For operations that can safely process all allocated memory:

```rust,ignore
block.for_each_including_non_visible(|sample| *sample *= 0.5);
block.enumerate_including_non_visible(|ch, frame, sample| {
    // Process including allocated but non-visible samples
});
```

Direct memory access:
```rust,ignore
let data: &[f32] = block.raw_data();  // Includes non-visible samples
```

## Performance

Iterator performance varies by layout:
- Sequential/Planar: Channel iteration faster
- Interleaved (many channels): Frame iteration faster

`raw_data()` access is fastest but exposes non-visible samples. For simple operations like gain, processing all samples (including non-visible) can be more efficient.

Check layout before optimization:
```rust,ignore
match block.layout() {
    BlockLayout::Planar => { /* channel-wise processing */ }
    BlockLayout::Interleaved => { /* frame-wise processing */ }
    BlockLayout::Sequential => { /* channel-wise processing */ }
}
```

## `no_std` Support

Disable default features. Owned blocks require `alloc` or `std` feature.
