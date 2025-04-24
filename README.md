# Audio Blocks

This crate provides traits for audio blocks to generalize common problems in handling audio data, like different channel layouts and varying number of samples.

Audio data can have different formats:

* Interleaved: `[ch0, ch1, ch0, ch1, ch0, ch1]`
  * Layout: frames first in one single buffer
  * often used in system APIs, as the hardware most often operates on frame-by-frame basis (each frame needs to be played at the same time)

* Sequential: `[ch0, ch0, ch0, ch1, ch1, ch1]`
  * Layout: channels first in one single buffer
  * often used in dsp code, as effects often operate on channel-by-channel basis

* Stacked: `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
  * Layout: channels first in individual buffers
  * often used in dsp code, as effects often operate on channel-by-channel basis
  * packing each channel in an individual buffer can have performance improvements

This crate provides the two traits `AudioBlock` and `AudioBlockMut` which provide common functions for any of those layouts.

```rust
pub fn process(mut block: impl AudioBlockMut<f32>) {
    for channel in block.channels_mut() {
        for sample in channel {
            *sample *= 0.5;
        }
    }
}
```

## Handling Varying Number of Frames

The number of samples in audio buffers coming from audio APIs can vary with each process call and often only the maximum number of frames is given.
This is the reason why all blocks have a number of **allocated** frames and channels and **visible** frames and channels.
With the function `resize` the buffers can be resized as long as they do not grow larger than the allocated memory. Resize is always real-time safe!
When using the `copy_from_block_resize` function, the destination block will autoamtically adapt the size of the source block.

> [!WARNING]
> This is the reason why accessing `raw_data` can be dangerous as it will give access to all contained samples, even the one that should not be visible.

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
pub fn process(mut block: impl AudioBlockMut<f32>) {
    block.apply_gain(0.5);
}
```

```rust
pub fn process(mut block: impl AudioBlockMut<f32>) {
    block.for_each(|_channel, _frame, value| {
        *value *= 0.5;
    });
}
```

```rust
pub fn process(mut block: impl AudioBlockMut<f32>) {
    block.clear();
}
```

```rust
pub fn process(mut block: impl AudioBlockMut<f32>, other_block: impl AudioBlock<f32>) {
    block.copy_from_block(other_block);
}
```
