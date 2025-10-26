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
///     for sample in audio.channel(0) {
///         // work with each sample
///     }
///
///     // Process all channels
///     for channel in audio.channels() {
///         for sample in channel {
///             // Apply processing to each sample
///         }
///     }
/// }
/// ```
pub trait AudioBlock<S: Sample> {
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
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;

    /// Returns an iterator that yields iterators for each channel.
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;

    /// Returns a slice for a single channel in case of sequential or planar layout.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn try_channel_slice(&self, channel: u16) -> Option<&[S]> {
        let _ = channel;
        None
    }

    /// Returns an iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;

    /// Returns an iterator that yields iterators for each frame.
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;

    /// Returns a slice for a single frame in case of interleaved memory layout.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn try_frame_slice(&self, frame: usize) -> Option<&[S]> {
        let _ = frame;
        None
    }

    /// Creates a non-owning view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn view(&self) -> impl AudioBlock<S>;

    /// Provides direct access to the underlying memory as an interleaved slice.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[S])` if the block's layout is [`BlockLayout::Interleaved`], containing
    /// interleaved samples across all allocated channels. Returns `None` if the layout is
    /// not interleaved. Check `block.layout() == BlockLayout::Interleaved` before calling.
    fn try_raw_data_interleaved(&self) -> Option<&[S]> {
        None
    }

    /// Provides direct access to the underlying memory as a planar slice for a specific channel.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Parameters
    ///
    /// * `ch` - Specifies which channel to access.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[S])` if the block's layout is [`BlockLayout::Planar`], containing
    /// data for the specified channel only. Returns `None` if the layout is
    /// not planar. Check `block.layout() == BlockLayout::Planar` before calling.
    fn try_raw_channel_planar(&self, ch: u16) -> Option<&[S]> {
        let _ = ch;
        None
    }

    /// Provides direct access to the underlying memory as a sequential slice.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[S])` if the block's layout is [`BlockLayout::Sequential`], containing
    /// sequential data with all allocated channels. Returns `None` if the layout is
    /// not sequential. Check `block.layout() == BlockLayout::Sequential` before calling.
    fn try_raw_data_sequential(&self) -> Option<&[S]> {
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
///     for sample in audio.channel_mut(0) {
///         *sample *= 0.8; // Apply gain reduction
///     }
///
///     // Process all channels
///     for mut channel in audio.channels_mut() {
///         for sample in channel {
///             // Apply processing to each sample
///         }
///     }
/// }
/// ```
pub trait AudioBlockMut<S: Sample>: AudioBlock<S> {
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
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;

    /// Returns a mutable iterator that yields mutable iterators for each channel.
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;

    /// Returns a mutable slice for a single channel in case of sequential or planar layout.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn try_channel_slice_mut(&mut self, channel: u16) -> Option<&mut [S]> {
        let _ = channel;
        None
    }

    /// Returns a mutable iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;

    /// Returns a mutable iterator that yields mutable iterators for each frame.
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;

    /// Returns a mutable slice for a single frame in case of interleaved memory layout.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn try_frame_slice_mut(&mut self, frame: usize) -> Option<&mut [S]> {
        let _ = frame;
        None
    }

    /// Creates a non-owning mutable view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn view_mut(&mut self) -> impl AudioBlockMut<S>;

    /// Provides direct mutable access to the underlying memory as an interleaved slice.
    ///
    /// This function gives mutable access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut [S])` if the block's layout is [`BlockLayout::Interleaved`], containing
    /// interleaved samples across all allocated channels. Returns `None` if the layout is
    /// not interleaved. Check `block.layout() == BlockLayout::Interleaved` before calling.
    fn try_raw_data_interleaved_mut(&mut self) -> Option<&mut [S]> {
        None
    }

    /// Provides direct mutable access to the underlying memory as a planar slice for a specific channel.
    ///
    /// This function gives mutable access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Parameters
    ///
    /// * `ch` - Specifies which channel to access.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut [S])` if the block's layout is [`BlockLayout::Planar`], containing
    /// data for the specified channel only. Returns `None` if the layout is
    /// not planar. Check `block.layout() == BlockLayout::Planar` before calling.
    fn try_raw_channel_planar_mut(&mut self, ch: u16) -> Option<&mut [S]> {
        let _ = ch;
        None
    }

    /// Provides direct mutable access to the underlying memory as a sequential slice.
    ///
    /// This function gives mutable access to all allocated data, including any reserved capacity
    /// beyond the active range.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut [S])` if the block's layout is [`BlockLayout::Sequential`], containing
    /// sequential data with all allocated channels. Returns `None` if the layout is
    /// not sequential. Check `block.layout() == BlockLayout::Sequential` before calling.
    fn try_raw_data_sequential_mut(&mut self) -> Option<&mut [S]> {
        None
    }
}
