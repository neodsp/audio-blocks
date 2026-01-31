use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

use super::{view::AudioBlockMonoView, view_mut::AudioBlockMonoViewMut};
use crate::{AudioBlock, AudioBlockMut, Sample};

/// A mono (single-channel) audio block that owns its data.
///
/// This is a simplified version of the multi-channel audio blocks, optimized for
/// mono audio processing with a streamlined API that doesn't require channel indexing.
///
/// * **Layout:** `[sample0, sample1, sample2, ...]`
/// * **Interpretation:** A simple sequence of samples representing a single audio channel.
/// * **Usage:** Ideal for mono audio processing, side-chain signals, or any single-channel audio data.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let mut block = AudioBlockMono::new(512);
///
/// // Fill with a simple ramp
/// for (i, sample) in block.samples_mut().iter_mut().enumerate() {
///     *sample = i as f32;
/// }
///
/// assert_eq!(block.sample(0), 0.0);
/// assert_eq!(block.sample(511), 511.0);
/// ```
pub struct AudioBlockMono<S: Sample> {
    data: Box<[S]>,
    num_frames: usize,
    num_frames_allocated: usize,
}

impl<S: Sample + Default> AudioBlockMono<S> {
    /// Creates a new mono audio block with the specified number of frames.
    ///
    /// Allocates memory for a new mono audio block with exactly the specified
    /// number of frames. The block is initialized with the default value (zero)
    /// for the sample type.
    ///
    /// Do not use in real-time processes!
    ///
    /// # Arguments
    ///
    /// * `num_frames` - The number of frames (samples)
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::AudioBlockMono, AudioBlock};
    ///
    /// let block = AudioBlockMono::<f32>::new(1024);
    /// assert_eq!(block.num_frames(), 1024);
    /// ```
    #[blocking]
    pub fn new(num_frames: usize) -> Self {
        Self {
            data: vec![S::default(); num_frames].into_boxed_slice(),
            num_frames,
            num_frames_allocated: num_frames,
        }
    }
}

impl<S: Sample> AudioBlockMono<S> {
    /// Creates a new mono audio block from a slice of samples.
    ///
    /// Copies the provided slice into a new owned mono block.
    ///
    /// # Warning
    ///
    /// This function allocates memory and should not be used in real-time audio processing contexts.
    ///
    /// # Arguments
    ///
    /// * `samples` - The slice of samples to copy
    #[blocking]
    pub fn from_slice(samples: &[S]) -> Self {
        Self {
            data: samples.to_vec().into_boxed_slice(),
            num_frames: samples.len(),
            num_frames_allocated: samples.len(),
        }
    }

    /// Creates a new mono audio block from a slice of samples.
    ///
    /// Copies the provided slice into a new owned mono block.
    ///
    /// # Warning
    ///
    /// This function allocates memory and should not be used in real-time audio processing contexts.
    ///
    /// # Arguments
    ///
    /// * `samples` - The slice of samples to copy
    /// * `num_frames_visible` - Number of audio frames to expose
    #[blocking]
    pub fn from_slice_limited(samples: &[S], num_frames_visible: usize) -> Self {
        assert!(num_frames_visible <= samples.len());
        Self {
            data: samples.to_vec().into_boxed_slice(),
            num_frames: num_frames_visible,
            num_frames_allocated: samples.len(),
        }
    }

    /// Creates a new mono audio block by copying data from another [`AudioBlock`].
    ///
    /// Extracts the first channel from any [`AudioBlock`] implementation.
    /// If the source block has no channels, creates an empty mono block.
    ///
    /// # Warning
    ///
    /// This function allocates memory and should not be used in real-time audio processing contexts.
    ///
    /// # Arguments
    ///
    /// * `block` - The source audio block to copy data from (first channel will be used)
    ///
    /// # Panics
    ///
    /// Panics if the source block has zero channels.
    #[blocking]
    pub fn from_block(block: &impl AudioBlock<S>) -> Self {
        assert!(
            block.num_channels() > 0,
            "Cannot create mono block from block with zero channels"
        );

        let mut data = Vec::with_capacity(block.num_frames());
        block.channel_iter(0).for_each(|&v| data.push(v));

        Self {
            data: data.into_boxed_slice(),
            num_frames: block.num_frames(),
            num_frames_allocated: block.num_frames(),
        }
    }

    /// Returns the sample at the specified frame index.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    #[nonblocking]
    pub fn sample(&self, frame: usize) -> S {
        assert!(frame < self.num_frames);
        unsafe { *self.data.get_unchecked(frame) }
    }

    /// Returns a mutable reference to the sample at the specified frame index.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    #[nonblocking]
    pub fn sample_mut(&mut self, frame: usize) -> &mut S {
        assert!(frame < self.num_frames);
        unsafe { self.data.get_unchecked_mut(frame) }
    }

    /// Provides direct access to the underlying samples as a slice.
    ///
    /// Returns only the visible samples (up to `num_frames`).
    #[nonblocking]
    pub fn samples(&self) -> &[S] {
        &self.data[..self.num_frames]
    }

    /// Provides direct mutable access to the underlying samples as a slice.
    ///
    /// Returns only the visible samples (up to `num_frames`).
    #[nonblocking]
    pub fn samples_mut(&mut self) -> &mut [S] {
        let num_frames = self.num_frames;
        &mut self.data[..num_frames]
    }

    /// Provides direct access to all allocated memory, including reserved capacity.
    ///
    /// This gives access to the full allocated buffer, including any frames
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data(&self) -> &[S] {
        &self.data
    }

    /// Provides direct mutable access to all allocated memory, including reserved capacity.
    ///
    /// This gives mutable access to the full allocated buffer, including any frames
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Creates a view of this mono audio block.
    #[nonblocking]
    pub fn view(&self) -> AudioBlockMonoView<'_, S> {
        AudioBlockMonoView::from_slice_limited(
            self.raw_data(),
            self.num_frames,
            self.num_frames_allocated,
        )
    }

    /// Creates a mutable view of this mono audio block.
    #[nonblocking]
    pub fn view_mut(&mut self) -> AudioBlockMonoViewMut<'_, S> {
        let num_frames = self.num_frames;
        let num_frames_allocated = self.num_frames_allocated;
        AudioBlockMonoViewMut::from_slice_limited(
            self.raw_data_mut(),
            num_frames,
            num_frames_allocated,
        )
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockMono<S> {
    type PlanarView = [S; 0];

    #[nonblocking]
    fn num_channels(&self) -> u16 {
        1
    }

    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn num_channels_allocated(&self) -> u16 {
        1
    }

    #[nonblocking]
    fn num_frames_allocated(&self) -> usize {
        self.num_frames_allocated
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Sequential
    }

    #[nonblocking]
    fn sample(&self, channel: u16, frame: usize) -> S {
        assert_eq!(channel, 0, "AudioBlockMono only has channel 0");
        self.sample(frame)
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
        assert_eq!(channel, 0, "AudioBlockMono only has channel 0");
        self.samples().iter()
    }

    #[nonblocking]
    fn channels_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &S>>
    {
        core::iter::once(self.samples().iter())
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        core::iter::once(&self.data[frame])
    }

    #[nonblocking]
    fn frames_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &S>>
    {
        self.data.iter().take(self.num_frames).map(core::iter::once)
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        AudioBlockMonoView::from_slice_limited(
            self.raw_data(),
            self.num_frames,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> AudioBlockMut<S> for AudioBlockMono<S> {
    type PlanarViewMut = [S; 0];

    #[nonblocking]
    fn set_num_channels_visible(&mut self, num_channels: u16) {
        assert_eq!(
            num_channels, 1,
            "AudioBlockMono can only have 1 channel, got {}",
            num_channels
        );
    }

    #[nonblocking]
    fn set_num_frames_visible(&mut self, num_frames: usize) {
        assert!(
            num_frames <= self.num_frames_allocated,
            "Cannot set visible frames ({}) beyond allocated frames ({})",
            num_frames,
            self.num_frames_allocated
        );
        self.num_frames = num_frames;
    }

    #[nonblocking]
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S {
        assert_eq!(channel, 0, "AudioBlockMono only has channel 0");
        self.sample_mut(frame)
    }

    #[nonblocking]
    fn channel_iter_mut(
        &mut self,
        channel: u16,
    ) -> impl ExactSizeIterator<Item = &mut S> {
        assert_eq!(channel, 0, "AudioBlockMono only has channel 0");
        self.samples_mut().iter_mut()
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl '_
    + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &mut S>> {
        core::iter::once(self.samples_mut().iter_mut())
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        let ptr = &mut self.data[frame] as *mut S;
        // Safety: We're creating a single-item iterator from a valid mutable reference
        core::iter::once(unsafe { &mut *ptr })
    }

    #[nonblocking]
    fn frames_iter_mut(
        &mut self,
    ) -> impl '_
    + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &mut S>> {
        let num_frames = self.num_frames;
        self.data.iter_mut().take(num_frames).map(core::iter::once)
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        let num_frames = self.num_frames;
        let num_frames_allocated = self.num_frames_allocated;
        AudioBlockMonoViewMut::from_slice_limited(
            self.raw_data_mut(),
            num_frames,
            num_frames_allocated,
        )
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for AudioBlockMono<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "AudioBlockMono {{")?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  samples: {:?}", self.samples())?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_new() {
        let block = AudioBlockMono::<f32>::new(1024);
        assert_eq!(block.num_frames(), 1024);
        assert_eq!(block.num_frames_allocated(), 1024);
        assert!(block.samples().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_from_slice() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMono::from_slice(&samples);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.samples(), &samples);
    }

    #[test]
    fn test_sample_access() {
        let mut block = AudioBlockMono::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(block.sample(0), 1.0);
        assert_eq!(block.sample(2), 3.0);
        assert_eq!(block.sample(4), 5.0);

        *block.sample_mut(2) = 10.0;
        assert_eq!(block.sample(2), 10.0);
    }

    #[test]
    fn test_iterators() {
        let mut block = AudioBlockMono::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let sum: f32 = block.samples().iter().sum();
        assert_eq!(sum, 15.0);

        for sample in block.samples_mut() {
            *sample *= 2.0;
        }

        assert_eq!(block.samples(), &[2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_resize_beyond_allocated() {
        let mut block = AudioBlockMono::<f32>::new(10);
        block.set_num_frames_visible(11);
    }

    #[test]
    fn test_audio_block_trait() {
        let block = AudioBlockMono::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(block.num_channels(), 1);
        assert_eq!(block.num_frames(), 5);

        // Test channel_iter
        let channel: Vec<f32> = block.channel_iter(0).copied().collect();
        assert_eq!(channel, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test frame_iter
        let frame: Vec<f32> = block.frame_iter(2).copied().collect();
        assert_eq!(frame, vec![3.0]);
    }

    #[test]
    fn test_audio_block_mut_trait() {
        let mut block = AudioBlockMono::<f32>::new(5);

        for (i, sample) in block.channel_iter_mut(0).enumerate() {
            *sample = i as f32;
        }

        assert_eq!(block.samples(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let block = AudioBlockMono::<f32>::new(10);
        let _ = block.channel_iter(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_sample_out_of_bounds() {
        let block = AudioBlockMono::<f32>::new(10);
        let _ = block.sample(10);
    }

    #[test]
    fn test_from_block() {
        use crate::AudioBlockInterleaved;

        let mut multi = AudioBlockInterleaved::<f32>::new(2, 5);
        for (i, sample) in multi.channel_iter_mut(0).enumerate() {
            *sample = i as f32;
        }

        let mono = AudioBlockMono::from_block(&multi);
        assert_eq!(mono.samples(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_views() {
        let mut block = AudioBlockMono::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test immutable view
        {
            let view = block.as_view();
            assert_eq!(view.num_frames(), 5);
            assert_eq!(view.sample(0, 2), 3.0);
        }

        // Test mutable view
        {
            let mut view_mut = block.as_view_mut();
            *view_mut.sample_mut(0, 2) = 10.0;
        }

        assert_eq!(block.sample(2), 10.0);
    }
}
