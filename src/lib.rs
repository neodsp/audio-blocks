//! # audio-blocks
//!
//! This crate provides traits for audio blocks to generalize common problems in handling audio data, like different channel layouts, adapting between them and receiving varying number of samples.
//! You will get [`Interleaved`], [`Sequential`] and [`Stacked`] blocks and you can select where the data is stored by choosing between owned data, views and mutable views.
//! Owned blocks will store the data on the heap, while views can be created over slices, raw pointers or from any other block.
//! All of them implement the [`AudioBlock`] / [`AudioBlockMut`] traits and the mutable blocks implement operations.
//!
//! This crate can be used in `no_std` contexts when disabling the default features. Owned blocks are stored on the heap and thus need either the `alloc` or the `std` feature.
//! Everything in this library, except for generating new owned blocks, is real-time safe.
//!
//! The main problem this crate is solving is that audio data can have different formats:
//!
//! * **Interleaved:** `[ch0, ch1, ch0, ch1, ch0, ch1]`
//!     * **Interpretation:** Each group of channel samples represents a frame. So, this layout stores frames one after another.
//!     * **Terminology:** Described as “packed” or “frames first” because each time step is grouped and processed as a unit (a frame).
//!     * **Usage:** Often used in APIs or hardware-level interfaces, where synchronized playback across channels is crucial.
//!
//! * **Sequential:** `[ch0, ch0, ch0, ch1, ch1, ch1]`
//!     * **Interpretation:** All samples from `ch0` are stored first, followed by all from `ch1`, etc.
//!     * **Terminology:** Described as “planar” or “channels first” in the sense that all data for one channel appears before any data for the next.
//!     * **Usage:** Used in DSP pipelines where per-channel processing is easier and more efficient.
//!
//! * **Stacked:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
//!     * **Interpretation:** Each channel has its own separate buffer or array.
//!     * **Terminology:** Also described as “planar” or “channels first” though more specifically it’s channel-isolated buffers.
//!     * **Usage:** Very common in real-time DSP, as it simplifies memory access and can improve SIMD/vectorization efficiency.
//!
//! So if you write your processor functions expecting an `impl [`AudioBlock`]<S>` you can receive any kind of audio data, no matter which layout the audio API is using.
//! [`AudioBlock`]s can contain any type of sample that is [`Copy`], [`Default`] and `'static` which is true for all kinds of numbers.
//!
//! As you mostly don't want your process function to work with any kind of number type, you can write a specialized process block, expecting only `f32` samples.
//!
//! ```ignore
//! fn process(block: &mut impl AudioBlockMut<f32>) {
//!     for channel in block.channels_mut() {
//!         for sample in channel {
//!             *sample *= 0.5;
//!         }
//!     }
//! }
//! ```
//!
//! or you can write a generic process block which works on all floating point values (`f32`, `f64` and optionally `half::f16`) by using the `Float` trait from the `num` or `num_traits` crate:
//!
//! ```ignore
//! use num_traits::Float;
//!
//! fn process<F: Float>(block: &mut impl AudioBlockMut<F>) {
//!     let gain = F::from(0.5).unwrap();
//!     for channel in block.channels_mut() {
//!         for sample in channel {
//!             *sample *= gain;
//!         }
//!     }
//! }
//! ```
//!
//! Access to the audio data can be achieved with the iterators [`AudioBlock::channels()`] or [`AudioBlock::frames()`], by accessing a specific one with [`AudioBlock::channel()`] or [`AudioBlock::frame()`] and accessing only one value with [`AudioBlock::sample()`].
//! Iterating over frames can be faster for interleaved data, while iterating over channels is always faster for sequential or stacked data.
//!
//! ## All Trait Functions
//!
//! ### [`AudioBlock`]
//!
//! ```ignore
//! fn num_channels(&self) -> u16;
//! fn num_frames(&self) -> usize;
//! fn num_channels_allocated(&self) -> u16;
//! fn num_frames_allocated(&self) -> usize;
//! fn sample(&self, channel: u16, frame: usize) -> S;
//! fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
//! fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
//! fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
//! fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
//! fn view(&self) -> impl AudioBlock<S>;
//! fn layout(&self) -> BlockLayout;
//! fn raw_data(&self, stacked_ch: Option<u16>) -> &[S];
//! ```
//! ### [`AudioBlockMut`]
//!
//! contains all of the non-mutable functions plus:
//!
//! ```ignore
//! fn resize(&mut self, num_channels: u16, num_frames: usize);
//! fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;
//! fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
//! fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
//! fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
//! fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
//! fn view_mut(&mut self) -> impl AudioBlockMut<S>;
//! fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [S];
//! ```
//!
//! ## Operations
//!
//! There are multiple operations defined on audio blocks, which allow copying data between them and applying an operation on each sample.
//!
//! ```ignore
//! fn copy_from_block(&mut self, block: &impl AudioBlock<S>);
//! fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>);
//! fn for_each(&mut self, f: impl FnMut(&mut S));
//! fn for_each_including_non_visible(&mut self, f: impl FnMut(&mut S));
//! fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
//! fn enumerate_including_non_visible(&mut self, f: impl FnMut(u16, usize, &mut S));
//! fn fill_with(&mut self, sample: S);
//! fn clear(&mut self);
//! ```
//!
//! ## Create Audio Blocks
//!
//! ### Owned
//!
//! Types:
//! * [`Interleaved`]
//! * [`Sequential`]
//! * [`Stacked`]
//!
//! ```ignore
//! fn new(num_channels: u16, num_frames: usize) -> Self;
//! fn from_block(block: &impl AudioBlock<S>) -> Self;
//! ```
//!
//! ### Views
//!
//! Types:
//! * [`InterleavedView`] / [`InterleavedViewMut`]
//! * [`SequentialView`] / [`SequentialViewMut`]
//! * [`StackedView`] / [`StackedViewMut`]
//!
//! ```ignore
//! fn from_slice(data: &'a [S], num_channels: u16, num_frames: usize) -> Self;
//! fn from_slice_limited(data: &'a [S], num_channels_visible: u16, num_frames_visible: usize, num_channels_allocated: u16, num_frames_allocated: usize) -> Self;
//! ```
//!
//! Interleaved and sequential blocks can be directly generated from pointers:
//!
//! ```ignore
//! unsafe fn from_ptr(data: *const S, num_channels: u16, num_frames: usize) -> Self;
//! unsafe fn from_ptr_limited(data: *const S, num_channels_visible: u16, num_frames_visible: usize, num_channels_allocated: u16, num_frames_allocated: usize) -> Self;
//! ```
//!
//! Stacked blocks can only be generated from pointers using [`StackedPtrAdapter`]:
//!
//! ```ignore
//! let mut adapter = unsafe { StackedPtrAdapter::<_, 16>::from_ptr(data, num_channels, num_frames) };
//! let block = adapter.stacked_view();
//! ```
//!
//! ## Handling Varying Number of Frames
//!
//! The number of samples in audio buffers coming from audio APIs can vary with each process call and often only the maximum number of frames is given.
//! This is the reason why all blocks have a number of **allocated** frames and channels and **visible** frames and channels.
//!
//! With the function [`AudioBlockMut::resize()`] the buffers can be resized as long as they do not grow larger than the allocated memory. Resize is always real-time safe!
//! When using the [`Ops::copy_from_block_resize()`] function, the destination block will automatically adapt the size of the source block.
//! For views the `from_slice_limited` or `from_ptr_limited` functions will provide you with a way to directly limit the visible data of the underlying memory.
//!
//! Here you see how you adapt your block size to incoming blocks with changing sizes if you need to copy the data for any reason:
//!
//! ```ignore
//! fn process(&mut self, other_block: &mut impl AudioBlock<f32>) {
//!     self.block.copy_from_block_resize(other_block);
//! }
//! ```
//!
//! ## Performance Optimizations
//!
//! The most performant way to iterate over blocks will be by accessing the raw data. But this can be dangerous because in limited blocks it will retrieve samples that
//! are not meant to be visible and you have to figure out the data layout. For simple operations that do not depend on other samples like applying a gain, doing so for all samples,
//! can be faster (if the amount of invisible samples is not exceptionally high). In the [`AudioBlockMut`] trait you will find [`Ops::for_each()`] and [`Ops::for_each_including_non_visible()`] or [`Ops::enumerate()`] and [`Ops::enumerate_including_non_visible()`] for this reason.
//!
//! If you use the iterators [`AudioBlock::channels()`] or [`AudioBlock::frames()`] it depends on the layout which one will be more performant.
//! For [`Sequential`] and [`Stacked`] it will be always faster to iterate over the channels, for [`Interleaved`] with higher channel counts it can be faster to iterate over frames.
//!
//! The layout of a block can be retrieved using the [`AudioBlock::layout()`] function.

#![cfg_attr(not(feature = "std"), no_std)] // enable std library when feature std is provided
#![cfg_attr(not(test), no_std)] // activate std library only for tests

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate std;

pub use ops::Ops;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use interleaved::Interleaved;
pub use interleaved::InterleavedView;
pub use interleaved::InterleavedViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use sequential::Sequential;
pub use sequential::SequentialView;
pub use sequential::SequentialViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use stacked::Stacked;
pub use stacked::StackedPtrAdapter;
pub use stacked::StackedPtrAdapterMut;
pub use stacked::StackedView;
pub use stacked::StackedViewMut;

pub mod interleaved;
mod iter;
pub mod ops;
pub mod sequential;
pub mod stacked;

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

    /// All samples from one channel appear consecutively before the next channel.
    ///
    /// Format: `[ch0, ch0, ch0, ..., ch1, ch1, ch1, ...]`
    ///
    /// Also known as "planar" format in some audio libraries.
    Sequential,

    /// Channels are separated into discrete chunks of memory.
    ///
    /// Format: `[[ch0, ch0, ch0, ...], [ch1, ch1, ch1, ...]]`
    ///
    /// Useful for operations that work on one channel at a time.
    Stacked,
}

/// Represents a sample type that can be stored and processed in audio blocks.
///
/// This trait is automatically implemented for any type that meets the following requirements:
/// - `Copy`: The type can be copied by value efficiently
/// - `Default`: The type has a reasonable default/zero value
/// - `'static`: The type doesn't contain any non-static references
///
/// All numeric types (f32, f64, i16, i32, etc.) automatically implement this trait,
/// as well as any custom types that satisfy these bounds.
pub trait Sample: Copy + Default + 'static {}
impl<T> Sample for T where T: Copy + Default + 'static {}

/// Core trait for audio data access operations across various memory layouts.
///
/// [`AudioBlock`] provides a unified interface for interacting with audio data regardless of its
/// underlying memory representation ([`BlockLayout::Interleaved`], [`BlockLayout::Sequential`], or [`BlockLayout::Stacked`]). It supports operations
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
pub trait AudioBlock<T: Sample> {
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

    /// Returns the sample value at the specified channel and frame position.
    ///
    /// # Panics
    ///
    /// Panics if channel or frame indices are out of bounds.
    fn sample(&self, channel: u16, frame: usize) -> T;

    /// Returns an iterator over all samples in the specified channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn channel(&self, channel: u16) -> impl Iterator<Item = &T>;

    /// Returns an iterator that yields iterators for each channel.
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &T> + '_> + '_;

    /// Returns an iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame(&self, frame: usize) -> impl Iterator<Item = &T>;

    /// Returns an iterator that yields iterators for each frame.
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &T> + '_> + '_;

    /// Creates a non-owning view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn view(&self) -> impl AudioBlock<T>;

    /// Returns the memory layout of this audio block (interleaved, sequential, or stacked).
    fn layout(&self) -> BlockLayout;

    /// Provides direct access to the underlying memory as a slice.
    ///
    /// # Parameters
    ///
    /// * `stacked_ch` - For `Layout::Stacked`, specifies which channel to access (required).
    ///   For other layouts, this parameter is ignored.
    ///
    /// # Returns
    ///
    /// A slice containing all allocated data, including any reserved capacity beyond
    /// the visible/active range. The data format follows the block's layout:
    /// - For `Interleaved`: returns interleaved samples across all channels
    /// - For `Sequential`: returns planar data with all channels
    /// - For `Stacked`: returns data for the specified channel only
    fn raw_data(&self, stacked_ch: Option<u16>) -> &[T];
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
///     audio.resize(2, 1024);
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
pub trait AudioBlockMut<T: Sample>: AudioBlock<T> {
    /// Resizes the audio block to the specified number of channels and frames.
    ///
    /// This operation is real-time safe but only works up to [`AudioBlock::num_channels_allocated`]
    /// and [`AudioBlock::num_frames_allocated`]. Attempting to resize beyond the allocated capacity
    /// will have implementation-dependent behavior.
    ///
    /// # Panics
    ///
    /// This function may panic when attempting to resize beyond the allocated capacity
    /// (`num_channels_allocated` and `num_frames_allocated`).
    fn resize(&mut self, num_channels: u16, num_frames: usize);

    /// Returns a mutable reference to the sample at the specified channel and frame position.
    ///
    /// # Panics
    ///
    /// Panics if channel or frame indices are out of bounds.
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut T;

    /// Returns a mutable iterator over all samples in the specified channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut T>;

    /// Returns a mutable iterator that yields mutable iterators for each channel.
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut T> + '_> + '_;

    /// Returns a mutable iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut T>;

    /// Returns a mutable iterator that yields mutable iterators for each frame.
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut T> + '_> + '_;

    /// Creates a non-owning mutable view of this audio block.
    ///
    /// This operation is zero-cost (no allocation or copying) and real-time safe,
    /// as it returns a lightweight wrapper around the original data.
    fn view_mut(&mut self) -> impl AudioBlockMut<T>;

    /// Provides direct mutable access to the underlying memory as a slice.
    ///
    /// # Parameters
    ///
    /// * `stacked_ch` - For `BlockLayout::Stacked`, specifies which channel to access (required).
    ///   For other layouts, this parameter is ignored.
    ///
    /// # Returns
    ///
    /// A mutable slice containing all allocated data, including any reserved capacity beyond
    /// the visible/active range. The data format follows the block's layout:
    /// - For `Interleaved`: returns interleaved samples across all channels
    /// - For `Sequential`: returns planar data with all channels
    /// - For `Stacked`: returns data for the specified channel only
    fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [T];
}
