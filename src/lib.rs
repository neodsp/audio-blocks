//! # audio-blocks
//!
//! Real-time safe abstractions over audio data with support for all common layouts.
//!
//! ## Quick Start
//!
//! Install:
//! ```sh
//! cargo add audio-blocks
//! ```
//!
//! Basic planar usage (most common for DSP):
//! ```
//! use audio_blocks::*;
//!
//! // Create a planar block - each channel gets its own buffer
//! let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames
//!
//! // Process per channel
//! for channel in block.channels_mut() {
//!     for sample in channel {
//!         *sample *= 0.5;
//!     }
//! }
//! ```
//!
//! Generic function that accepts any layout:
//! ```
//! # use audio_blocks::*;
//! fn process(block: &mut impl AudioBlockMut<f32>) {
//!     for channel in block.channels_iter_mut() {
//!         for sample in channel {
//!             *sample *= 0.5;
//!         }
//!     }
//! }
//! ```
//!
//! ## Block Types
//!
//! Three multi-channel layouts supported:
//!
//! **Planar** - `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
//! Each channel has its own separate buffer. Standard for real-time DSP. Optimal for SIMD/vectorization.
//!
//! **Sequential** - `[ch0, ch0, ch0, ch1, ch1, ch1]`
//! Single contiguous buffer with all samples for channel 0, then all samples for channel 1. Channel-contiguous in one allocation.
//!
//! **Interleaved** - `[ch0, ch1, ch0, ch1, ch0, ch1]`
//! Channels alternate sample-by-sample. Common in audio APIs and hardware interfaces.
//!
//! Plus a dedicated mono type:
//!
//! **Mono** - `[sample0, sample1, sample2, ...]`
//! Simplified single-channel block with a streamlined API that doesn't require channel indexing.
//!
//! ## Creating Blocks
//!
//! Owned blocks allocate via `new()` — never do this in real-time contexts.
//! Views borrow existing data via `from_slice()` or `from_ptr()` and are always real-time safe.
//!
//! | Owned (allocates) | View (borrows data) |
//! |---|---|
//! | [`Planar`] | [`PlanarView`] / [`PlanarViewMut`] |
//! | [`Sequential`] | [`SequentialView`] / [`SequentialViewMut`] |
//! | [`Interleaved`] | [`InterleavedView`] / [`InterleavedViewMut`] |
//! | [`Mono`] | [`MonoView`] / [`MonoViewMut`] |
//!
//! Views can also be created from raw pointers (`from_ptr`). For planar pointer data,
//! use [`PlanarPtrs`] / [`PlanarPtrsMut`], which borrow the caller's array of channel pointers.
//!
//! ## Traits
//!
//! Use `impl AudioBlock<f32>` / `impl AudioBlockMut<f32>` to write layout-generic functions
//! (as shown above). These traits are also generic over the sample type (`f32`, `f64`, `i16`, etc.).
//!
//! | Trait | Purpose |
//! |---|---|
//! | [`AudioBlock`] | Read-only access: sample access, channel/frame iteration, layout info |
//! | [`AudioBlockMut`] | Mutable access: sample mutation, resizing, per-sample iteration |
//! | [`AudioBlockOps`] | Read-only operations: mono mixdown, channel extraction |
//! | [`AudioBlockOpsMut`] | Composite operations: block copy, mono fan-out |
//!
//! Two further traits describe what a layout can do, so generic code states its
//! requirement in the signature instead of inspecting the layout at run time:
//!
//! | Trait | Implemented by | Gives you |
//! |---|---|---|
//! | [`Contiguous`] / [`ContiguousMut`] | interleaved, sequential, mono | the flat sample slice |
//! | [`FramesMut`] | interleaved, sequential, mono | mutable frame-major iteration |
//!
//! Planar blocks implement neither: each channel is a separate allocation, so
//! there is no flat slice, and independent mutable frames would need a cached
//! pointer per channel. Iterate them channel-major, or a frame at a time with
//! [`AudioBlockMut::frame_iter_mut`].
//!
//! Blocks also separate allocated capacity from visible size — see [`AudioBlockMut::set_num_frames_visible`]
//! for real-time safe buffer resizing without reallocation.
//!
//! ## `no_std` Support
//!
//! Disable default features. Owned blocks require `alloc` or `std` feature.
#![cfg_attr(all(not(test), not(feature = "std")), no_std)] // enable std library when feature std is provided

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate std;

pub use ops::AudioBlockOps;
pub use ops::AudioBlockOpsMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use interleaved::Interleaved;
pub use interleaved::InterleavedView;
pub use interleaved::InterleavedViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use sequential::Sequential;
pub use sequential::SequentialView;
pub use sequential::SequentialViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use planar::Planar;
pub use planar::PlanarPtrs;
pub use planar::PlanarPtrsMut;
pub use planar::PlanarView;
pub use planar::PlanarViewMut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use mono::Mono;
pub use mono::MonoView;
pub use mono::MonoViewMut;

pub mod interleaved;
mod iter;
pub mod mono;
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
    /// Note: Unlike `Planar`, this uses a single contiguous buffer rather than separate buffers per channel.
    Sequential,
}

/// Represents a sample type that can be stored and processed in audio blocks.
///
/// This trait is automatically implemented for any type that meets the following requirements:
/// - `Copy`: The type can be copied by value efficiently
/// - `'static`: The type doesn't contain any non-static references
///
/// All numeric types (f32, f64, i16, i32, etc.) automatically implement this trait,
/// as well as any custom types that satisfy these bounds.
pub trait Sample: Copy + 'static {}
impl<T> Sample for T where T: Copy + 'static {}

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
///     for channel in audio.channels_iter() {
///         for sample in channel {
///             // work with each sample
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
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S>;

    /// Returns an iterator that yields an iterator for each channel.
    fn channels_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>>;

    /// Returns an iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S>;

    /// Returns an iterator that yields an iterator for each frame.
    fn frames_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>>;

    /// Creates a non-owning view of this audio block.
    ///
    /// This operation is real-time safe, as it returns a lightweight
    /// wrapper around the original data.
    fn as_view(&self) -> impl AudioBlock<S>;
}

/// Blocks whose samples all live in one contiguous slice.
///
/// Implemented by interleaved, sequential and mono blocks. Planar blocks store
/// each channel in its own allocation and so cannot implement it.
///
/// Take this bound when you need the flat buffer, for example to hand it to a C
/// API. It replaces asking a block what layout it has and then downcasting: the
/// requirement is expressed in the signature and checked at compile time.
///
/// ```
/// # use audio_blocks::*;
/// fn to_c_api(block: &impl Contiguous<f32>) -> &[f32] {
///     block.raw_data()
/// }
/// ```
pub trait Contiguous<S: Sample>: AudioBlock<S> {
    /// Returns every allocated sample as one slice, in memory order.
    ///
    /// Matches the inherent `raw_data` of each contiguous block type.
    fn raw_data(&self) -> &[S];
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
///     audio.set_visible(2, 1024);
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
///     for mut channel in audio.channels_iter_mut() {
///         for sample in channel {
///             // Apply processing to each sample
///         }
///     }
/// }
/// ```
pub trait AudioBlockMut<S: Sample>: AudioBlock<S> {
    /// Sets the visible size of the audio block to the specified number of channels and frames.
    ///
    /// # Panics
    ///
    /// When `num_channels` exceeds [`AudioBlock::num_channels_allocated`] or `num_frames` exceeds [`AudioBlock::num_frames_allocated`].
    fn set_visible(&mut self, num_channels: u16, num_frames: usize) {
        self.set_num_channels_visible(num_channels);
        self.set_num_frames_visible(num_frames);
    }

    /// Sets the visible size of the audio block to the specified number of channels.
    ///
    /// This operation is real-time safe but only works up to [`AudioBlock::num_channels_allocated`].
    ///
    /// # Panics
    ///
    /// When `num_channels` exceeds [`AudioBlock::num_channels_allocated`].
    fn set_num_channels_visible(&mut self, num_channels: u16);

    /// Sets the visible size of the audio block to the specified number of frames.
    ///
    /// # Panics
    ///
    ///  When `num_frames` exceeds [`AudioBlock::num_frames_allocated`].
    fn set_num_frames_visible(&mut self, num_frames: usize);

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
    fn channel_iter_mut(&mut self, channel: u16) -> impl ExactSizeIterator<Item = &mut S>;

    /// Returns a mutable iterator that yields mutable iterators for each channel.
    fn channels_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>>;

    /// Returns a mutable iterator over all samples in the specified frame (across all channels).
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S>;

    /// Creates a non-owning mutable view of this audio block.
    ///
    /// This operation is real-time safe, as it returns a lightweight
    /// wrapper around the original data.
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S>;

    /// Visits every sample in the visible region.
    ///
    /// The default walks channel by channel. Layouts with a faster traversal
    /// override it.
    fn for_each(&mut self, mut f: impl FnMut(&mut S)) {
        for channel in self.channels_iter_mut() {
            channel.for_each(&mut f);
        }
    }

    /// Visits every sample in the visible region with its channel and frame index.
    ///
    /// The default walks channel by channel. Layouts with a faster traversal
    /// override it.
    fn enumerate(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        for (channel, samples) in self.channels_iter_mut().enumerate() {
            for (frame, sample) in samples.enumerate() {
                f(channel as u16, frame, sample);
            }
        }
    }

    /// Visits every *allocated* sample in memory order, including samples
    /// outside the visible region.
    ///
    /// Linear traversal of the underlying storage, so this is faster than
    /// [`for_each`](AudioBlockMut::for_each) when the visible region covers most
    /// of the allocation. There is no layout-independent way to reach samples
    /// outside the visible region, so every layout implements this itself.
    fn for_each_allocated(&mut self, f: impl FnMut(&mut S));

    /// Visits every *allocated* sample with its channel and frame index.
    ///
    /// See [`for_each_allocated`](AudioBlockMut::for_each_allocated).
    fn enumerate_allocated(&mut self, f: impl FnMut(u16, usize, &mut S));

    /// Sets every allocated sample to `sample`.
    fn fill_with(&mut self, sample: S) {
        self.for_each_allocated(|v| *v = sample);
    }

    /// Sets every allocated sample to the default value (zero for numeric types).
    fn clear(&mut self)
    where
        S: Default,
    {
        self.fill_with(S::default());
    }

    /// Multiplies every allocated sample by `gain`.
    fn gain(&mut self, gain: S)
    where
        S: core::ops::Mul<Output = S>,
    {
        self.for_each_allocated(|v| *v = *v * gain);
    }
}

/// Mutable counterpart to [`Contiguous`].
pub trait ContiguousMut<S: Sample>: Contiguous<S> + AudioBlockMut<S> {
    /// Returns every allocated sample as one mutable slice, in memory order.
    ///
    /// Matches the inherent `raw_data_mut` of each contiguous block type.
    fn raw_data_mut(&mut self) -> &mut [S];
}

/// Mutable frame-major iteration, for layouts that can reach a frame without
/// side storage.
///
/// Interleaved, sequential and mono blocks implement this. Planar blocks do not:
/// each channel is a separate allocation, so handing out independent frames would
/// require caching one pointer per channel. Iterate planar blocks with
/// [`AudioBlockMut::frame_iter_mut`] one frame at a time, or with
/// [`AudioBlockMut::for_each`] / [`AudioBlockMut::enumerate`].
///
/// ```
/// # use audio_blocks::*;
/// let mut block = Interleaved::<f32>::new(2, 4);
/// for frame in block.frames_iter_mut() {
///     for sample in frame { *sample *= 0.5; }
/// }
/// ```
pub trait FramesMut<S: Sample>: AudioBlockMut<S> {
    /// Returns a mutable iterator that yields a mutable iterator per frame.
    fn frames_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>>;
}
