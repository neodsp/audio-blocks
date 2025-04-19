# Audio Blocks

This crate provides a trait to abstract interleaved, planar and stacked audio data in real-time processes.

By using the traits `AudioBlock` or `AudioBlockMut`, your dsp code will work on any common layout of audio data.

## Example

```rust
pub fn process(mut data: impl AudioBlockMut<f32>) {
    data.apply_gain(0.5);
}
```

## Creating Blocks

```rust

```

## Access to Channels

```rust
channels_mut!(block, channel, {
    for sample in channel {
        *sample *= 0.5;
    }
});
```

Similar to the `channels_mut!` macro, there is `channels!`, `frames!` and `frames_mut!`.

The macro is just syntactic sugar for the following

```rust
for c in 0..block.num_channels() {
    for sample in block.channel_mut(c) {
        *sample *= 0.5;
    }
}
```
