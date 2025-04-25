# neodsp audio-blocks

![image](docs/audio-blocks-logo.png)

This crate provides traits for audio blocks to generalize common problems in handling audio data, like different channel layouts, adapting between them and receiving varying number of samples.
You will get `Interleaved`, `Sequential` and `Stacked` blocks and you can select where the data is stored by choosing between owned data, views and mutable views.
Owned blocks will store the data on the heap, while views can be created over slices, raw pointers or from any other block.
All of them implement the `AudioBlock` / `AudioBlockMut` traits and the mutable blocks implement operations like `clear`, `fill_with`, `for_each`, and `enumerate`.

This crate can be used in `no_std` contexts when disabling the default features. Owned blocks are stored on the heap and thus need either the `alloc` or the `std` feature.
Everything in this library, except for generating new owned blocks, is real-time safe.

The main problem this crate is solving is that audio data can have different formats:

* Interleaved: `[ch0, ch1, ch0, ch1, ch0, ch1]`
  * Layout: frames first in one single buffer
  * Often used in system APIs, as the hardware most often operates on frame-by-frame basis (each frame needs to be played at the same time)

* Sequential: `[ch0, ch0, ch0, ch1, ch1, ch1]`
  * Layout: channels first in one single buffer
  * Often used in DSP code, as effects often operate on channel-by-channel basis

* Stacked: `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
  * Layout: channels first in individual buffers
  * Often used in DSP code, as effects often operate on channel-by-channel basis
  * Packing each channel in an individual buffer can have performance improvements

So if you write your processor functions expecting an `impl AudioBlock<S>` you can receive any kind of audio data, no matter which layout the audio API is using.
AudioBlocks can contain any type of sample that is `Copy`, `Default` and `'static` which is true for all kinds of numbers.

As you mostly don't want your process function to work with any kind of number, you can write a specialized process block, expecting only `f32` samples.

```rust
fn process(block: &mut impl AudioBlockMut<f32>) {
    for channel in block.channels_mut() {
        for sample in channel {
            *sample *= 0.5;
        }
    }
}
```

or you can write a generic process block which works on all floating point values (`f32`, `f64` and optionally `half::f16`) by using the `Float` trait from the `num` or `num_traits` crate:

```rust
use num_traits::Float;

fn process<F: Float>(block: &mut impl AudioBlockMut<F>) {
    let gain = F::from(0.5).unwrap();
    for channel in block.channels_mut() {
        for sample in channel {
            *sample *= gain;
        }
    }
}
```

Access to the audio data can be achieved with the iterators `channels()` or `frames()`, by accessing a specific one with `channel(u16)` or `frame(usize)` and accessing only one value with `sample(u16, usize)`.
Iterating over frames can be faster for interleaved data, while iterating over channels is always faster for sequential or stacked data.

## All Trait Functions

### AudioBlock

```rust
fn num_channels(&self) -> u16;
fn num_frames(&self) -> usize;
fn num_channels_allocated(&self) -> u16;
fn num_frames_allocated(&self) -> usize;
fn sample(&self, channel: u16, frame: usize) -> S;
fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
fn view(&self) -> impl AudioBlock<S>;
fn layout(&self) -> BlockLayout;
fn raw_data(&self, stacked_ch: Option<u16>) -> &[S];
```
### AudioBlockMut

contains all of the non-mutable functions plus:

```Rust
fn resize(&mut self, num_channels: u16, num_frames: usize);
fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;
fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
fn view_mut(&mut self) -> impl AudioBlockMut<S>;
fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [S];
```

## Operations

There are multiple operations defined on audio blocks, which allow copying data between them and applying an operation on each sample.

```rust
fn copy_from_block(&mut self, block: &impl AudioBlock<S>);
fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>);
fn for_each(&mut self, f: impl FnMut(&mut S));
fn for_each_including_non_visible(&mut self, f: impl FnMut(&mut S));
fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
fn enumerate_including_non_visible(&mut self, f: impl FnMut(u16, usize, &mut S));
fn fill_with(&mut self, sample: S);
fn clear(&mut self);
```

## Handling Varying Number of Frames

> [!NOTE]
> This is mostly needed if you need to copy the data to another block. Usually in audio APIs as you can generate a new view on every callback
> directly with the size of the incoming data and you never have to resize anything.

The number of samples in audio buffers coming from audio APIs can vary with each process call and often only the maximum number of frames is given.
This is the reason why all blocks have a number of **allocated** frames and channels and **visible** frames and channels.

With the function `resize` the buffers can be resized as long as they do not grow larger than the allocated memory. Resize is always real-time safe!
When using the `copy_from_block_resize` function, the destination block will automatically adapt the size of the source block.
For views the `from_slice_limited` or `from_ptr_limited` functions will provide you with a way to directly limit the visible data of the underlying memory.

Here you see how you adapt your block size to incoming blocks with changing sizes if you need to copy the data for any reason:

```rust
fn process(&mut self, other_block: &mut impl AudioBlock<f32>) {
    self.block.copy_from_block_resize(other_block);
}
```

> [!WARNING]
> This is the reason why accessing `raw_data` can be dangerous. It will give you access to all contained samples, even the ones that should not be visible!

## Generate Audio blocks

### Owned Blocks

The new function will pre-fill the block with the default value, which will be zero for most numeric values:

```rust
let block = Interleaved::<f32>::new(2, 64);
let block = Sequential::<f32>::new(2, 64);
let block = Stacked::<f32>::new(2, 64);
```

Other than that you can generate an owned block from any kind of existing block:

```rust
let block = Interleaved::from_block(old_block);
let block = Sequential::from_block(old_block);
let block = Stacked::from_block(old_block);
```

> [!WARNING]
> Never generate an owned block in a real-time context! `new` and `from_block` are the only functions that allocate memory in this crate.

### Block Views

> [!NOTE]
> All views exist as mutable or non-mutable version, depending if the underlying data should be mutable or not.
> `InterleavedView`, `SequentialView` and `StackedView` are non-mutable, `InterleavedViewMut`, `SequentialViewMut` and `StackedViewMut` are mutable.
> All mentioned constructor functions exist for the mutable and non-mutable versions and are real-time safe!

Views can be generated from slices, from pointers and any block can generate a new view.
Views are extremely cheap to create, so you can do this in every audio callback.

This is how you generate views from existing blocks (no matter if they are owned or already a view):

```rust
let view = block.view();
let view_mut = block.view_mut();
```

This is how you generate a new view from existing data:

```rust
let data = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
let block = InterleavedView::from_slice(&data, 2, 3);

let data = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
let block = SequentialView::from_slice(&data, 2, 3);
```

This even works directly on pointers, although this is mostly used when handling data from a C-API:

```rust
fn c_api_callback_interleaved(data: *mut f32, num_channels: u16, num_frames: usize) {
    let block = unsafe { InterleavedViewMut::from_ptr(data, num_channels, num_frames) };
}
```

There also exist `limited` functions that can be handy when the underlying data stores more channels and/or frames than you want to make visible in the block.

```rust
let block = InterleavedViewMut::from_slice_limited(&mut data, /* visible */ 2, 3, /* allocated */ 4, 6);
let block = unsafe { InterleavedView::from_ptr_limited(ptr, /* visible */ 2, 3, /* allocated */ 4, 6) };
```

## Stacked Block Views

For stacked views only the functions `from_slices` and `from_slices_limited` are available to create the views.
The functions accept any kind of collection over samples that implement `AsRef<[S]>` or `AsMut<[S]>`.

A `StackedView` or `StackedViewMut` can be generated over vectors:

```rust
let slices = vec![vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]];
let block = StackedView::from_slices(&slices);
```

over arrays:

```rust
let slices = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
let block = StackedView::from_slices(&slices);
```

and over slices:

```rust
let slices = [&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]];
let block = StackedView::from_slices(&slices);
```

To generate stacked views from pointers, a special adapter is needed to avoid allocations.
Note that in this case you have to provide the maximum number of channels you want to support as a constant value.
In this case we limit this adapter to 16 channels.

```rust
fn c_api_stacked(data: *mut *mut f32, num_channels: u16, num_frames: usize) {
    let mut adapter = unsafe { StackedPtrAdapterMut::<_, 16>::from_ptr(data, num_channels, num_frames) };
    let block = adapter.stacked_view_mut();
}
```

Specifically the Juce API will give you a `*const *mut f32` in this case you can just cast the pointer with `as *mut *mut f32`:

```rust
fn c_api_stacked(data: *const *mut f32, num_channels: u16, num_frames: usize) {
    let mut adapter = unsafe { StackedPtrAdapterMut::<_, 16>::from_ptr(data as *mut *mut f32, num_channels, num_frames) };
    let block = adapter.stacked_view_mut();
}
```

## Performance Optimizations

The most performant way to iterate over blocks will be by accessing the raw data. But this can be dangerous because in limited blocks it will retrieve samples that
are not meant to be visible and you have to figure out the data layout. For simple operations that do not depend on other samples like applying a gain, doing so for all samples,
can be faster (if the amount of invisible samples is not exceptionally high). In the `Ops` you will find `for_each` and `for_each_including_non_visible` or `enumerate` and `enumerate_including_non_visible` for this reason.

If you use the iterators `channels` or `frames` it depends on the layout which one will be more performant.
For `Sequential` and `Stacked` it will be always faster to iterate over the channels, for `Interleaved` with higher channel counts it can be faster to iterate over frames.

The layout of a block can be retrieved using the `layout` function.
