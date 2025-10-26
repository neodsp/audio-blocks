#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)] // enable std library when feature std is provided
#![cfg_attr(not(test), no_std)] // activate std library only for tests

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate std;

pub use num::Zero;
pub use ops::AudioBlockOps;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use interleaved::AudioBlockInterleaved;
pub use interleaved::AudioBlockInterleavedView;
pub use interleaved::AudioBlockInterleavedViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use sequential::AudioBlockSequential;
pub use sequential::AudioBlockSequentialView;
pub use sequential::AudioBlockSequentialViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use planar::AudioBlockPlanar;
pub use planar::AudioBlockPlanarView;
pub use planar::AudioBlockPlanarViewMut;
pub use planar::PlanarPtrAdapter;
pub use planar::PlanarPtrAdapterMut;

pub mod interleaved;
mod iter;
pub mod ops;
pub mod planar;
pub mod sequential;

/// Represents the memory layout of audio data returned by [`AudioBlock::layout`].
///
/// This enum allows consumers to determine the underlying data layout, which is essential for:
/// - Direct raw data access
/// - Performance optimizations
/// - Efficient interfacing with external audio APIs
///
/// # Examples of layouts
///
/// Each variant represents a common pattern used in audio processing.
#[derive(PartialEq, Debug)]
pub enum BlockLayout {
    /// Samples from different channels alternate in sequence.
    ///
    /// Format: `[ch0, ch1, ..., ch0, ch1, ..., ch0, ch1, ...]`
    ///
    /// This layout is common in consumer audio formats and some APIs.
    Interleaved,

    /// Channels are separated into discrete chunks of memory.
    ///
    /// Format: `[[ch0, ch0, ch0, ...], [ch1, ch1, ch1, ...]]`
    ///
    /// Useful for operations that work on one channel at a time.
    Planar,

    /// All samples from one channel appear consecutively before the next channel.
    ///
    /// Format: `[ch0, ch0, ch0, ..., ch1, ch1, ch1, ...]`
    ///
    /// Also known as "planar" format in some audio libraries.
    Sequential,
}

/// Represents a sample type that can be stored and processed in audio blocks.
///
/// This trait is automatically implemented for any type that meets the following requirements:
/// - `Copy`: The type can be copied by value efficiently
/// - `Zero`: The type has a zero value
/// - `'static`: The type doesn't contain any non-static references
///
/// All numeric types (f32, f64, i16, i32, etc.) automatically implement this trait,
/// as well as any custom types that satisfy these bounds.
pub trait Sample: Copy + Zero + 'static {}
impl<T> Sample for T where T: Copy + Zero + 'static {}

/// Core trait for audio data access operations across various memory layouts.
///
/// [`AudioBlock`] provides a unified interface for interacting with audio data regardless of its
/// underlying memory representation ([`BlockLayout::Interleaved`], [`BlockLayout::Sequential`], or [`BlockLayout::Planar`]). It supports operations
/// on both owned audio blocks and temporary views.
///
/// # Usage
///
/// This trait gives you multiple ways to access audio data:
/// - Direct sample access via indices
/// - Channel and frame iterators for processing data streams
/// - Raw data access for optimized operations
/// - Layout information for specialized handling
///
/// # Example
///
/// ```
/// use audio_blocks::AudioBlock;
///
/// fn example(audio: &impl AudioBlock<f32>) {
///     // Get number of channels and frames
///     let channels = audio.num_channels();
///     let frames = audio.num_frames();
///
///     // Access individual samples
///     let first_sample = audio.sample(0, 0);
///
///     // Process one channel
///     for sample in audio.channel_iter(0) {
///         // work with each sample
///     }
///
///     // Process all channels
///     for channel in audio.channel_iters() {
///         for sample in channel {
///             // work with each sample
///         }
///     }
/// }
/// ```
pub trait AudioBlock<S: Sample> {
    type PlanarView: AsRef<[S]>;

    /// Returns the number of active audio channels.
    fn num_channels(&self) -> u16;

    /// Returns the number of audio frames (samples per channel).
    fn num_frames(&self) -> usize;

    /// Returns the total number of channels allocated in memory.
    ///
    /// This may be greater than `num_channels()` if the buffer has reserved capacity.
    fn num_channels_allocated(&self) -> u16;

    /// Returns the total number of frames allocated in memory.
    ///
    /// This may be greater than `num_frames()` if the buffer has reserved capacity.
    fn num_frames_allocated(&self) -> usize;

    /// Returns the memory layout of this audio block (interleaved, sequential, or planar).
    fn layout(&self) -> BlockLayout;

    /// Returns the sample value at the specified channel and frame position.
    ///
    /// # Panics
    ///
    /// Panics if channel or frame indices are out of bounds.
    fn sample(&self, channel: u16, frame: usize) -> S;

    /// Returns an iterator over all samples in the specified channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn channel_iter(&self, channel: u16) -> impl Iterator<Item = &S>;

    /// Returns an iterator that yields an iterator for each channel.
    fn channel_iters(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;

    /// Returns an iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_iter(&self, frame: usize) -> impl Iterator<Item = &S>;

    /// Returns an iterator that yields an iterator for each frame.
    fn frame_iters(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;

    /// Creates a non-owning view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn as_view(&self) -> impl AudioBlock<S>;

    fn as_interleaved_view(&self) -> Option<AudioBlockInterleavedView<'_, S>> {
        None
    }

    fn as_planar_view(&self) -> Option<AudioBlockPlanarView<'_, S, Self::PlanarView>> {
        None
    }

    fn as_sequential_view(&self) -> Option<AudioBlockSequentialView<'_, S>> {
        None
    }
}

/// Extends the [`AudioBlock`] trait with mutable access operations.
///
/// [`AudioBlockMut`] provides methods for modifying audio data across different memory layouts.
/// It enables in-place processing, buffer resizing, and direct mutable access to the underlying data.
///
/// # Usage
///
/// This trait gives you multiple ways to modify audio data:
/// - Change individual samples at specific positions
/// - Iterate through and modify channels or frames
/// - Resize the buffer to accommodate different audio requirements
/// - Access raw data for optimized processing
///
/// # Example
///
/// ```
/// use audio_blocks::{AudioBlock, AudioBlockMut};
///
/// fn process_audio(audio: &mut impl AudioBlockMut<f32>) {
///     // Resize to 2 channels, 1024 frames
///     audio.set_active_size(2, 1024);
///
///     // Modify individual samples
///     *audio.sample_mut(0, 0) = 0.5;
///
///     // Process one channel with mutable access
///     for sample in audio.channel_iter_mut(0) {
///         *sample *= 0.8; // Apply gain reduction
///     }
///
///     // Process all channels
///     for mut channel in audio.channel_iters_mut() {
///         for sample in channel {
///             // Apply processing to each sample
///         }
///     }
/// }
/// ```
pub trait AudioBlockMut<S: Sample>: AudioBlock<S> {
    type PlanarViewMut: AsRef<[S]> + AsMut<[S]>;

    /// Sets the active size of the audio block to the specified number of channels and frames.
    ///
    /// # Panics
    ///
    /// When `num_channels` exceeds [`AudioBlock::num_channels_allocated`] or `num_frames` exceeds [`AudioBlock::num_frames_allocated`].
    fn set_active_size(&mut self, num_channels: u16, num_frames: usize) {
        self.set_active_num_channels(num_channels);
        self.set_active_num_frames(num_frames);
    }

    /// Sets the active size of the audio block to the specified number of channels.
    ///
    /// This operation is real-time safe but only works up to [`AudioBlock::num_channels_allocated`].
    ///
    /// # Panics
    ///
    /// When `num_channels` exceeds [`AudioBlock::num_channels_allocated`].
    fn set_active_num_channels(&mut self, num_channels: u16);

    /// Sets the active size of the audio block to the specified number of frames.
    ///
    /// # Panics
    ///
    ///  When `num_frames` exceeds [`AudioBlock::num_frames_allocated`].
    fn set_active_num_frames(&mut self, num_frames: usize);

    /// Returns a mutable reference to the sample at the specified channel and frame position.
    ///
    /// # Panics
    ///
    /// Panics if channel or frame indices are out of bounds.
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;

    /// Returns a mutable iterator over all samples in the specified channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn channel_iter_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;

    /// Returns a mutable iterator that yields mutable iterators for each channel.
    fn channel_iters_mut(&mut self)
    -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;

    /// Returns a mutable iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_iter_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;

    /// Returns a mutable iterator that yields mutable iterators for each frame.
    fn frame_iters_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;

    /// Creates a non-owning mutable view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S>;

    fn as_interleaved_view_mut(&mut self) -> Option<AudioBlockInterleavedViewMut<'_, S>> {
        None
    }

    fn as_planar_view_mut(
        &mut self,
    ) -> Option<AudioBlockPlanarViewMut<'_, S, Self::PlanarViewMut>> {
        None
    }

    fn as_sequential_view_mut(&mut self) -> Option<AudioBlockSequentialViewMut<'_, S>> {
        None
    }
}
