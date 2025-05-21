# audio-blocks

![image](docs/audio-blocks-logo.png)

This crate offers traits for handling audio data in a generic way, addressing common challenges such as varying channel layouts, conversions between them, and processing different numbers of samples.

It provides `Interleaved`, `Sequential`, and `Stacked` block types, allowing you to choose the underlying data storage: owned data, views, and mutable views. Owned blocks allocate data on the heap, while views offer access to data from slices, raw pointers, or other blocks.

All block types implement the `AudioBlock` and `AudioBlockMut` traits, with mutable blocks providing in-place modification operations.

This crate supports `no_std` environments by disabling default features. Note that owned blocks require either the `alloc` or the `std` feature due to heap allocation.

With the exception of creating new owned blocks, all functionalities within this library are real-time safe.

The core problem this crate solves is the diversity of audio data formats:

* **Interleaved:** `[ch0, ch1, ch0, ch1, ch0, ch1]`
    * **Interpretation:** Consecutive channel samples form a frame. This layout stores frames sequentially.
    * **Terminology:** Often referred to as "packed" or "frames first" because each time step is grouped as a single processing unit (a frame).
    * **Usage:** Frequently used in APIs or hardware interfaces where synchronized playback across channels is essential.

* **Sequential:** `[ch0, ch0, ch0, ch1, ch1, ch1]`
    * **Interpretation:** All samples for channel 0 are stored first, followed by all samples for channel 1, and so on.
    * **Terminology:** Described as "planar" or "channels first," emphasizing that all data for one channel precedes the data for the next.
    * **Usage:** Common in Digital Signal Processing (DSP) pipelines where per-channel processing is more straightforward and efficient.

* **Stacked:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
    * **Interpretation:** Each channel has its own distinct buffer or array.
    * **Terminology:** Also known as "planar" or "channels first," but more specifically refers to channel-isolated buffers.
    * **Usage:** Highly prevalent in real-time DSP due to simplified memory access and potential for improved SIMD (Single Instruction, Multiple Data) or vectorization efficiency.

By designing your processor functions to accept an `impl AudioBlock<S>`, your code can seamlessly handle audio data regardless of the underlying layout used by the audio API. `AudioBlock`s can hold any sample type `S` that implements `Copy`, `Zero`, and has a `'static` lifetime, which includes all primitive number types.

For specialized processing requiring a specific sample type, such as `f32`, you can define functions that expect `impl AudioBlockMut<f32>`.

```rust,ignore
fn process(block: &mut impl AudioBlockMut<f32>) {
    for channel in block.channels_mut() {
        for sample in channel {
            *sample *= 0.5;
        }
    }
}
```

Alternatively, you can create generic processing blocks that work with various floating-point types (`f32`, `f64`, and optionally `half::f16`) by leveraging the `Float` trait from the `num` or `num-traits` crate:

```rust,ignore
use num_traits::Float;

fn process<F: Float + 'static>(block: &mut impl AudioBlockMut<F>) {
    let gain = F::from(0.5).unwrap();
    for channel in block.channels_mut() {
        for sample in channel {
            *sample *= gain;
        }
    }
}
```

Accessing audio data is facilitated through iterators like `channels()` and `frames()`. You can also access specific channels or frames using `channel(u16)` and `frame(usize)`, or individual samples with `sample(u16, usize)`. Iterating over frames can be more efficient for interleaved data, while iterating over channels is generally faster for sequential or stacked layouts.

## All Trait Functions

### `AudioBlock`

```rust,ignore
/// Size and layout information
fn num_channels(&self) -> u16;
fn num_frames(&self) -> usize;
fn num_channels_allocated(&self) -> u16;
fn num_frames_allocated(&self) -> usize;
fn layout(&self) -> BlockLayout;

/// Individual sample access
fn sample(&self, channel: u16, frame: usize) -> S;

/// Channel-based access
fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
fn channel_slice(&self, channel: u16) -> Option<&[S]>;

/// Frame-based access
fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
fn frame_slice(&self, frame: usize) -> Option<&[S]>;

/// Views and raw data access
fn view(&self) -> impl AudioBlock<S>;
fn raw_data(&self, stacked_ch: Option<u16>) -> &[S];
```

### `AudioBlockMut`

Includes all functions from `AudioBlock` plus:

```rust,ignore
/// Resize within allocated bounds
fn set_active_size(&mut self, num_channels: u16, num_frames: usize);
fn set_active_num_channels(&mut self, num_channels: u16);
fn set_active_num_frames(&mut self, num_frames: usize);

/// Individual sample access
fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;

/// Channel-based access
fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
fn channel_slice_mut(&mut self, channel: u16) -> Option<&mut [T]>;

/// Frame-based access
fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
fn frame_slice_mut(&mut self, frame: usize) -> Option<&mut [T]>;

/// Views and raw data access
fn view_mut(&mut self) -> impl AudioBlockMut<S>;
fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [S];
```

## Operations

Several operations are defined for audio blocks, enabling data copying between them and applying functions to each sample.

```rust,ignore
fn copy_from_block(&mut self, block: &impl AudioBlock<S>);
fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>);
fn for_each(&mut self, f: impl FnMut(&mut S));
fn for_each_including_non_visible(&mut self, f: impl FnMut(&mut S));
fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
fn enumerate_including_non_visible(&mut self, f: impl FnMut(u16, usize, &mut S));
fn fill_with(&mut self, sample: S);
fn clear(&mut self);
```

## Creating Audio Blocks

### Owned

Available types:

* `Interleaved`
* `Sequential`
* `Stacked`

```rust,ignore
fn new(num_channels: u16, num_frames: usize) -> Self;
fn from_block(block: &impl AudioBlock<S>) -> Self;
```

> **Warning:** Avoid creating owned blocks in real-time contexts! `new` and `from_block` are the only functions in this crate that perform memory allocation.

### Views

Available types:

* `InterleavedView` / `InterleavedViewMut`
* `SequentialView` / `SequentialViewMut`
* `StackedView` / `StackedViewMut`

```rust,ignore
fn from_slice(data: &'a [S], num_channels: u16, num_frames: usize) -> Self;
fn from_slice_limited(data: &'a [S], num_channels_visible: u16, num_frames_visible: usize, num_channels_allocated: u16, num_frames_allocated: usize) -> Self;
```

Interleaved and sequential blocks can be created directly from raw pointers:

```rust,ignore
unsafe fn from_ptr(data: *const S, num_channels: u16, num_frames: usize) -> Self;
unsafe fn from_ptr_limited(data: *const S, num_channels_visible: u16, num_frames_visible: usize, num_channels_allocated: u16, num_frames_allocated: usize) -> Self;
```

Stacked blocks can only be created from raw pointers using `StackedPtrAdapter`:

```rust,ignore
let mut adapter = unsafe { StackedPtrAdapter::<_, 16>::from_ptr(data, num_channels, num_frames) };
let block = adapter.stacked_view();
```

## Handling Varying Number of Frames

> **Note:** This is primarily useful when you need to copy data to another block. In typical audio API usage, you can usually create a new view with the size of the incoming data in each callback, eliminating the need for resizing.

Audio buffers from audio APIs can have a varying number of samples per processing call, often with only the maximum number of frames specified. To address this, all block types distinguish between the number of **allocated** frames and channels (the underlying storage capacity) and the number of **visible** frames and channels (the portion of data currently being used).

The `resize` function allows you to adjust the number of visible frames and channels, provided they do not exceed the allocated capacity. Resizing is always real-time safe.

The `copy_from_block_resize` function automatically adapts the size of the destination block to match the visible size of the source block.

For views, the `from_slice_limited` and `from_ptr_limited` functions enable you to directly specify the visible portion of the underlying memory.

Here's an example of how to adapt your block size to incoming blocks with changing sizes when copying data is necessary:

```rust,ignore
fn process(&mut self, other_block: &mut impl AudioBlock<f32>) {
    self.block.copy_from_block_resize(other_block);
}
```

> **Warning:** Accessing `raw_data` can be unsafe because it provides access to all contained samples, including those that are not intended to be visible.

## Performance Considerations

When iterating using channels or frames, performance is influenced by the block's memory layout.

* For Sequential and Stacked layouts, iterating over channels is generally faster.
* For Interleaved layouts, especially with a high number of channels, iterating over frames might offer better performance.

Accessing data via `channel_slice` and `frame_slice` isn't significantly faster than direct slice access but can be convenient for SIMD operations or functions requiring slice inputs.

The most performant way to access data is through `raw_data`. However, this method carries risks for blocks that have allocated more memory than they expose, as it grants access to non-visible samples. You'll also need to manually manage the data layout. For simple, sample-independent operations (e.g., applying gain), processing all samples (including non-visible ones, provided their count isn't excessive) can be more efficient. The `Ops` trait offers `for_each`, `for_each_including_non_visible`, `enumerate`, and `enumerate_including_non_visible` for these scenarios.

To determine a block's memory organization, use the `layout()` function.
