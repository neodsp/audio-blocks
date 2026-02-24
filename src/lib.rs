//! # audio-block
//!
//! Real-time safe abstractions over audio data with support for all common layouts.
//!
//! ## Quick Start
//!
//! Install:
//! ```sh
//! cargo add audio-block
//! ```
//!
//! Basic planar usage (most common for DSP):
//! ```
//! use audio_block::*;
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
//! # use audio_block::*;
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
//! ## Core Traits
//!
//! Write functions that accept any layout:
//!
//! ```
//! # use audio_block::*;
//! fn process(block: &mut impl AudioBlockMut<f32>) {
//!     // Works with planar, sequential, or interleaved
//! }
//! ```
//!
//! Generic across float types:
//!
//! ```
//! # use audio_block::*;
//! fn process<F: Copy + 'static + std::ops::MulAssign>(block: &mut impl AudioBlockMut<F>) {
//!     let gain: F = todo!();
//!     for channel in block.channels_iter_mut() {
//!         for sample in channel {
//!             *sample *= gain;
//!         }
//!     }
//! }
//! ```
//!
//! ## Creating Blocks
//!
//! ### Owned Blocks
//!
//! ```
//! use audio_block::*;
//!
//! // Allocate with default values (zero)
//! let mut block = Planar::<f32>::new(2, 512);       // 2 channels, 512 frames
//! let mut block = Sequential::<f32>::new(2, 512);   // 2 channels, 512 frames
//! let mut block = Interleaved::<f32>::new(2, 512);  // 2 channels, 512 frames
//! let mut block = Mono::<f32>::new(512);            // 512 frames
//!
//! // Copy from existing data
//! let channel_data = vec![[0.0f32; 512], [0.0f32; 512]];
//! let data = vec![0.0f32; 1024];
//! let mut block = Planar::from_slice(&channel_data);   // channels derived from slice
//! let mut block = Sequential::from_slice(&data, 2);   // 2 channels
//! let mut block = Interleaved::from_slice(&data, 2);  // 2 channels
//! let mut block = Mono::from_slice(&data);
//! ```
//!
//! Allocation only happens when creating owned blocks. Never do that in real-time contexts.
//!
//! ### Views (zero-allocation, borrows data)
//!
//! ```
//! use audio_block::*;
//!
//! let channel_data = vec![[0.0f32; 512], [0.0f32; 512]];
//! let data = vec![0.0f32; 1024];
//!
//! let block = PlanarView::from_slice(&channel_data);   // channels derived from slice
//! let block = SequentialView::from_slice(&data, 2);   // 2 channels
//! let block = InterleavedView::from_slice(&data, 2);  // 2 channels
//! let block = MonoView::from_slice(&data);
//! ```
//!
//! From raw pointers:
//! ```
//! # use audio_block::*;
//! let data = vec![0.0f32; 1024];
//! # let ptr = data.as_ptr();
//! let block = unsafe { InterleavedView::from_ptr(ptr, 2, 512) }; // 2 channels, 512 frames
//! ```
//!
//! Planar requires adapter:
//! ```
//! # use audio_block::*;
//! # let ch0 = vec![0.0f32; 512];
//! # let ch1 = vec![0.0f32; 512];
//! # let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
//! # let data = ptrs.as_ptr();
//! let mut adapter = unsafe { PlanarPtrAdapter::<_, 16>::from_ptr(data, 2, 512) }; // 2 channels, 512 frames
//! let block = adapter.planar_view();
//! ```
//!
//! ## Common Operations
//!
//! Import the extension traits for additional operations:
//!
//! ```
//! use audio_block::{AudioBlockOps, AudioBlockOpsMut};
//! ```
//!
//! ### Copying and Clearing
//!
//! ```
//! # use audio_block::*;
//! let other_block = Planar::<f32>::new(2, 512);
//! let mut block = Planar::<f32>::new(2, 512);
//!
//! // Copy from another block (flexible - copies min of both sizes)
//! let result = block.copy_from_block(&other_block);
//! // Returns None if exact match, Some((channels, frames)) if partial
//!
//! // Copy with exact size requirement (panics on mismatch)
//! block.copy_from_block_exact(&other_block);
//!
//! // Fill all samples with a value
//! block.fill_with(0.5);
//!
//! // Clear to zero
//! block.clear();
//! ```
//!
//! ### Per-Sample Processing
//!
//! ```
//! # use audio_block::*;
//! let mut block = Planar::<f32>::new(2, 512);
//!
//! // Process each sample
//! block.for_each(|sample| *sample *= 0.5);
//!
//! // Process with channel/frame indices
//! block.enumerate(|channel, frame, sample| {
//!     *sample *= 0.5;
//! });
//!
//! // Apply gain to all samples
//! block.gain(0.5);
//! ```
//!
//! ### Mono Conversions
//!
//! ```
//! # use audio_block::*;
//! let mut block = Planar::<f32>::new(2, 512);
//! let mut mono_data = vec![0.0f32; 512];
//! let mut mono_view = MonoViewMut::from_slice(&mut mono_data);
//!
//! // Mix all channels to mono (averages channels)
//! let result = block.mix_to_mono(&mut mono_view);
//! // Returns None if exact match, Some(frames_processed) if partial
//!
//! // Or with exact size requirement
//! block.mix_to_mono_exact(&mut mono_view);
//!
//! // Copy a specific channel to mono
//! block.copy_channel_to_mono(&mut mono_view, 0); // channel 0
//!
//! // Copy mono to all channels of a block
//! let mono_ro = MonoView::from_slice(&mono_data);
//! block.copy_mono_to_all_channels(&mono_ro);
//! ```
//!
//! ## Working with Slices
//!
//! Convert generic blocks to concrete types for slice access:
//!
//! ```
//! # use audio_block::*;
//! fn process(block: &mut impl AudioBlockMut<f32>) {
//!     if block.layout() == BlockLayout::Planar {
//!         let mut view = block.as_planar_view_mut().unwrap();
//!         let ch0: &mut [f32] = view.channel_mut(0);
//!         let ch1: &mut [f32] = view.channel_mut(1);
//!     }
//! }
//! ```
//!
//! Direct slice access on concrete types:
//!
//! ```
//! # use audio_block::*;
//! let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames
//! let channel: &[f32] = block.channel(0);
//! let raw_data: &[Box<[f32]>] = block.raw_data();
//!
//! let mut block = Interleaved::<f32>::new(2, 512); // 2 channels, 512 frames
//! let frame: &[f32] = block.frame(0);
//! let raw_data: &[f32] = block.raw_data();
//! ```
//!
//! ## Trait API Reference
//!
//! ### `AudioBlock`
//!
//! Size and layout:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &impl AudioBlock<f32>) {
//! let channels: u16 = audio.num_channels();
//! let frames: usize = audio.num_frames();
//! let layout: BlockLayout = audio.layout();
//! # }
//! ```
//!
//! Sample access:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &impl AudioBlock<f32>) {
//! let s: f32 = audio.sample(0, 0);
//! # }
//! ```
//!
//! Iteration:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &impl AudioBlock<f32>) {
//! for s in audio.channel_iter(0) { let _: &f32 = s; }
//! for ch in audio.channels_iter() { for s in ch { let _: &f32 = s; } }
//! for s in audio.frame_iter(0) { let _: &f32 = s; }
//! for fr in audio.frames_iter() { for s in fr { let _: &f32 = s; } }
//! # }
//! ```
//!
//! Generic view (zero-allocation):
//! ```
//! # use audio_block::*;
//! # fn example(audio: &impl AudioBlock<f32>) {
//! let view = audio.as_view();
//! # }
//! ```
//!
//! Downcast to concrete type:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &impl AudioBlock<f32>) {
//! let interleaved: Option<InterleavedView<f32>> = audio.as_interleaved_view();
//! let sequential: Option<SequentialView<f32>> = audio.as_sequential_view();
//! # }
//! ```
//!
//! ### `AudioBlockMut`
//!
//! Everything from `AudioBlock` plus:
//!
//! Resizing:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &mut impl AudioBlockMut<f32>) {
//! audio.set_visible(2, 1024);
//! audio.set_num_channels_visible(2);
//! audio.set_num_frames_visible(1024);
//! # }
//! ```
//!
//! Mutable access:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &mut impl AudioBlockMut<f32>) {
//! let s: &mut f32 = audio.sample_mut(0, 0);
//! for s in audio.channel_iter_mut(0) { let _: &mut f32 = s; }
//! for ch in audio.channels_iter_mut() { for s in ch { let _: &mut f32 = s; } }
//! for s in audio.frame_iter_mut(0) { let _: &mut f32 = s; }
//! for fr in audio.frames_iter_mut() { for s in fr { let _: &mut f32 = s; } }
//! # }
//! ```
//!
//! Generic view (zero-allocation):
//! ```
//! # use audio_block::*;
//! # fn example(audio: &mut impl AudioBlockMut<f32>) {
//! let view = audio.as_view_mut();
//! # }
//! ```
//!
//! Downcast to concrete type:
//! ```
//! # use audio_block::*;
//! # fn example(audio: &mut impl AudioBlockMut<f32>) {
//! let interleaved: Option<InterleavedViewMut<f32>> = audio.as_interleaved_view_mut();
//! let sequential: Option<SequentialViewMut<f32>> = audio.as_sequential_view_mut();
//! # }
//! ```
//!
//! ### `AudioBlockOps` (extension trait)
//!
//! Read-only operations on audio blocks:
//! ```
//! # use audio_block::*;
//! # fn example(block: &impl AudioBlock<f32>, dest: &mut MonoViewMut<f32>) {
//! let _: Option<usize> = block.mix_to_mono(dest);
//! block.mix_to_mono_exact(dest);
//! let _: Option<usize> = block.copy_channel_to_mono(dest, 0);
//! block.copy_channel_to_mono_exact(dest, 0);
//! # }
//! ```
//!
//! ### `AudioBlockOpsMut` (extension trait)
//!
//! Mutable operations on audio blocks:
//! ```
//! # use audio_block::*;
//! # fn example(block: &mut impl AudioBlockMut<f32>, other: &impl AudioBlock<f32>, mono: &MonoView<f32>) {
//! let _: Option<(u16, usize)> = block.copy_from_block(other);
//! block.copy_from_block_exact(other);
//! let _: Option<usize> = block.copy_mono_to_all_channels(mono);
//! block.copy_mono_to_all_channels_exact(mono);
//! block.for_each(|sample| *sample *= 0.5);
//! block.enumerate(|_ch, _fr, sample| { *sample *= 0.5; });
//! block.for_each_allocated(|sample| *sample *= 0.5);
//! block.fill_with(0.5);
//! block.clear();
//! block.gain(0.5);
//! # }
//! ```
//!
//! ## Advanced: Variable Buffer Sizes
//!
//! Blocks separate allocated capacity from visible size. Resize visible portion without reallocation:
//!
//! ```
//! # use audio_block::*;
//! let mut block = Planar::<f32>::new(2, 512); // 2 channels, 512 frames
//! block.set_num_frames_visible(256); // use only 256 frames
//! ```
//!
//! Create views with limited visibility:
//! ```
//! # use audio_block::*;
//! # let data = vec![0.0f32; 1024];
//! let block = InterleavedView::from_slice_limited(
//!     &data,
//!     2,   // num_channels_visible
//!     256, // num_frames_visible
//!     2,   // num_channels_allocated
//!     512  // num_frames_allocated
//! );
//! ```
//!
//! Query allocation:
//! ```
//! # use audio_block::*;
//! # let block = Planar::<f32>::new(2, 512);
//! let _ = block.num_channels_allocated();
//! let _ = block.num_frames_allocated();
//! ```
//!
//! ## Advanced: Access Allocated Samples
//!
//! For operations that process all allocated memory (including non-visible samples):
//!
//! ```
//! use audio_block::AudioBlockOpsMut;
//! # use audio_block::*;
//! # let mut block = Planar::<f32>::new(2, 512);
//!
//! block.for_each_allocated(|sample| *sample *= 0.5);
//! block.enumerate_allocated(|_ch, _frame, sample| {
//!     // Process including allocated but non-visible samples
//!     let _ = sample;
//! });
//! ```
//!
//! Note: `fill_with`, `clear`, and `gain` also operate on the entire allocated buffer for efficiency.
//!
//! Direct memory access:
//! ```
//! # use audio_block::*;
//! let block = Sequential::<f32>::new(2, 512);
//! let data: &[f32] = block.raw_data();  // Includes non-visible samples
//! ```
//!
//! ## Performance
//!
//! Iterator performance varies by layout:
//! - Sequential/Planar: Channel iteration faster
//! - Interleaved (many channels): Frame iteration faster
//!
//! `raw_data()` access is fastest but exposes non-visible samples. For simple operations like gain, processing all samples (including non-visible) can be more efficient.
//!
//! Check layout before optimization:
//! ```
//! # use audio_block::*;
//! # fn example(block: &impl AudioBlock<f32>) {
//! match block.layout() {
//!     BlockLayout::Planar => { /* channel-wise processing */ }
//!     BlockLayout::Interleaved => { /* frame-wise processing */ }
//!     BlockLayout::Sequential => { /* channel-wise processing */ }
//! }
//! # }
//! ```
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
pub use planar::PlanarPtrAdapter;
pub use planar::PlanarPtrAdapterMut;
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
/// - `Zero`: The type has a zero value
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
/// use audio_block::AudioBlock;
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

    /// Attempts to downcast this generic audio block to a concrete interleaved view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in interleaved format,
    /// otherwise returns `None`.
    fn as_interleaved_view(&self) -> Option<InterleavedView<'_, S>> {
        None
    }

    /// Attempts to downcast this generic audio block to a concrete planar view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in planar format,
    /// otherwise returns `None`.
    fn as_planar_view(&self) -> Option<PlanarView<'_, S, Self::PlanarView>> {
        None
    }

    /// Attempts to downcast this generic audio block to a concrete sequential view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in sequential format,
    /// otherwise returns `None`.
    fn as_sequential_view(&self) -> Option<SequentialView<'_, S>> {
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
/// use audio_block::{AudioBlock, AudioBlockMut};
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
    type PlanarViewMut: AsRef<[S]> + AsMut<[S]>;

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

    /// Returns a mutable iterator that yields mutable iterators for each frame.
    fn frames_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>>;

    /// Creates a non-owning mutable view of this audio block.
    ///
    /// This operation is real-time safe, as it returns a lightweight
    /// wrapper around the original data.
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S>;

    /// Attempts to downcast this generic audio block to a concrete interleaved view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in interleaved format,
    /// otherwise returns `None`.
    fn as_interleaved_view_mut(&mut self) -> Option<InterleavedViewMut<'_, S>> {
        None
    }

    /// Attempts to downcast this generic audio block to a concrete planar view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in planar format,
    /// otherwise returns `None`.
    fn as_planar_view_mut(&mut self) -> Option<PlanarViewMut<'_, S, Self::PlanarViewMut>> {
        None
    }

    /// Attempts to downcast this generic audio block to a concrete sequential view.
    /// This enables access to frame slices and the underlying raw data.
    ///
    /// Returns `Some` if the underlying data is stored in sequential format,
    /// otherwise returns `None`.
    fn as_sequential_view_mut(&mut self) -> Option<SequentialViewMut<'_, S>> {
        None
    }
}
