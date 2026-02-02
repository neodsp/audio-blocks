use rtsan_standalone::nonblocking;

use super::view::MonoView;
use crate::{AudioBlock, AudioBlockMut, Sample};

/// A mutable view of mono (single-channel) audio data.
///
/// This provides a lightweight, non-owning mutable reference to a slice of mono audio samples.
///
/// * **Layout:** `[sample0, sample1, sample2, ...]`
/// * **Interpretation:** A simple sequence of samples representing a single audio channel.
/// * **Usage:** Ideal for mono audio processing, side-chain signals, or any single-channel audio data.
///
/// # Example
///
/// ```
/// use audio_blocks::{mono::MonoViewMut, AudioBlock};
///
/// let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let mut block = MonoViewMut::from_slice(&mut data);
///
/// *block.sample_mut(0) = 10.0;
/// assert_eq!(block.sample(0), 10.0);
/// ```
pub struct MonoViewMut<'a, S: Sample> {
    data: &'a mut [S],
    num_frames: usize,
    num_frames_allocated: usize,
}

impl<'a, S: Sample> MonoViewMut<'a, S> {
    /// Creates a new mono audio block view from a mutable slice of audio samples.
    ///
    /// # Parameters
    /// * `data` - The mutable slice containing mono audio samples
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::MonoViewMut, AudioBlock};
    ///
    /// let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut block = MonoViewMut::from_slice(&mut samples);
    /// assert_eq!(block.num_frames(), 5);
    /// ```
    #[nonblocking]
    pub fn from_slice(data: &'a mut [S]) -> Self {
        let num_frames = data.len();
        Self {
            data,
            num_frames,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new mono audio block view from a mutable slice with limited visibility.
    ///
    /// This function allows creating a view that exposes only a subset of the allocated frames,
    /// which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The mutable slice containing mono audio samples
    /// * `num_frames_visible` - Number of audio frames to expose in the view
    /// * `num_frames_allocated` - Total number of frames allocated in the data buffer
    ///
    /// # Panics
    /// * Panics if the length of `data` doesn't equal `num_frames_allocated`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::MonoViewMut, AudioBlock};
    ///
    /// let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut block = MonoViewMut::from_slice_limited(&mut samples, 3, 5);
    /// assert_eq!(block.num_frames(), 3);
    /// assert_eq!(block.num_frames_allocated(), 5);
    /// ```
    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a mut [S],
        num_frames_visible: usize,
        num_frames_allocated: usize,
    ) -> Self {
        assert_eq!(data.len(), num_frames_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data,
            num_frames: num_frames_visible,
            num_frames_allocated,
        }
    }

    /// Creates a new mono audio block view from a mutable pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_frames` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned view
    /// - No other references (mutable or immutable) exist to the same memory
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::MonoViewMut, AudioBlock};
    ///
    /// let mut samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut block = unsafe { MonoViewMut::from_ptr(samples.as_mut_ptr(), 5) };
    /// assert_eq!(block.num_frames(), 5);
    /// ```
    #[nonblocking]
    pub unsafe fn from_ptr(ptr: *mut S, num_frames: usize) -> Self {
        Self {
            data: unsafe { std::slice::from_raw_parts_mut(ptr, num_frames) },
            num_frames,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new mono audio block view from a mutable pointer with limited visibility.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_frames_allocated` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned view
    /// - No other references (mutable or immutable) exist to the same memory
    #[nonblocking]
    pub unsafe fn from_ptr_limited(
        ptr: *mut S,
        num_frames_visible: usize,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: unsafe { std::slice::from_raw_parts_mut(ptr, num_frames_allocated) },
            num_frames: num_frames_visible,
            num_frames_allocated,
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
    #[nonblocking]
    pub fn raw_data(&self) -> &[S] {
        self.data
    }

    /// Provides direct mutable access to all allocated memory, including reserved capacity.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [S] {
        self.data
    }

    #[nonblocking]
    pub fn view(&self) -> MonoView<'_, S> {
        MonoView::from_slice_limited(self.data, self.num_frames, self.num_frames_allocated)
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> MonoViewMut<'_, S> {
        MonoViewMut::from_slice_limited(self.data, self.num_frames, self.num_frames_allocated)
    }
}

impl<S: Sample> AudioBlock<S> for MonoViewMut<'_, S> {
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
        assert_eq!(channel, 0, "MonoViewMut only has channel 0");
        self.sample(frame)
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
        assert_eq!(channel, 0, "MonoViewMut only has channel 0");
        self.samples().iter()
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        core::iter::once(self.samples().iter())
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        core::iter::once(&self.data[frame])
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        self.data.iter().take(self.num_frames).map(core::iter::once)
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }
}

impl<S: Sample> AudioBlockMut<S> for MonoViewMut<'_, S> {
    type PlanarViewMut = [S; 0];

    #[nonblocking]
    fn set_num_channels_visible(&mut self, num_channels: u16) {
        assert_eq!(
            num_channels, 1,
            "audio_block::MonoViewMut can only have 1 channel, got {}",
            num_channels
        );
    }

    #[nonblocking]
    fn set_num_frames_visible(&mut self, num_frames: usize) {
        assert!(num_frames <= self.num_frames_allocated);
        self.num_frames = num_frames;
    }

    #[nonblocking]
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S {
        assert_eq!(channel, 0, "audio_block::MonoViewMut only has channel 0");
        self.sample_mut(frame)
    }

    #[nonblocking]
    fn channel_iter_mut(&mut self, channel: u16) -> impl ExactSizeIterator<Item = &mut S> {
        assert_eq!(channel, 0, "audio_block::MonoViewMut only has channel 0");
        self.samples_mut().iter_mut()
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>> {
        core::iter::once(self.samples_mut().iter_mut())
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        let ptr = &mut self.data[frame] as *mut S;
        core::iter::once(unsafe { &mut *ptr })
    }

    #[nonblocking]
    fn frames_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>> {
        let num_frames = self.num_frames;
        self.data.iter_mut().take(num_frames).map(core::iter::once)
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for MonoViewMut<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "audio_block::MonoViewMut {{")?;
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
    fn test_from_slice() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = MonoViewMut::from_slice(&mut data);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_slice_limited() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = MonoViewMut::from_slice_limited(&mut data, 3, 5);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0]);
        assert_eq!(block.raw_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_ptr() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = unsafe { MonoViewMut::from_ptr(data.as_mut_ptr(), 5) };
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_ptr_limited() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = unsafe { MonoViewMut::from_ptr_limited(data.as_mut_ptr(), 3, 5) };
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sample_access() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut block = MonoViewMut::from_slice(&mut data);
        assert_eq!(block.sample(0), 1.0);
        assert_eq!(block.sample(2), 3.0);
        assert_eq!(block.sample(4), 5.0);

        *block.sample_mut(2) = 10.0;
        assert_eq!(block.sample(2), 10.0);
    }

    #[test]
    fn test_samples_iter() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut block = MonoViewMut::from_slice(&mut data);

        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        for sample in block.samples_mut() {
            *sample *= 2.0;
        }

        assert_eq!(block.samples(), &[2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_fill_and_clear() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut block = MonoViewMut::from_slice(&mut data);

        block.samples_mut().fill(0.5);
        assert_eq!(block.samples(), &[0.5, 0.5, 0.5, 0.5, 0.5]);

        for s in block.samples_mut() {
            *s = 0.0;
        }
        assert_eq!(block.samples(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_audio_block_trait() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = MonoViewMut::from_slice(&mut data);

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
        let mut data = [0.0, 0.0, 0.0, 0.0, 0.0];
        let mut block = MonoViewMut::from_slice(&mut data);

        for (i, sample) in block.channel_iter_mut(0).enumerate() {
            *sample = i as f32;
        }

        assert_eq!(block.samples(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_set_num_frames_visible() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut block = MonoViewMut::from_slice(&mut data);

        block.set_num_frames_visible(3);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_views() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut block = MonoViewMut::from_slice(&mut data);

        // Test immutable view
        {
            let view = block.view();
            assert_eq!(view.num_frames(), 5);
            assert_eq!(view.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        }

        // Test mutable view
        {
            let mut view_mut = block.view_mut();
            *view_mut.sample_mut(2) = 10.0;
        }

        assert_eq!(block.sample(2), 10.0);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_sample_out_of_bounds() {
        let mut data = [1.0, 2.0, 3.0];
        let block = MonoViewMut::from_slice(&mut data);
        let _ = block.sample(10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_sample_mut_out_of_bounds() {
        let mut data = [1.0, 2.0, 3.0];
        let mut block = MonoViewMut::from_slice(&mut data);
        let _ = block.sample_mut(10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut data = [1.0, 2.0, 3.0];
        let block = MonoViewMut::from_slice(&mut data);
        let _ = block.channel_iter(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_set_num_frames_beyond_allocated() {
        let mut data = [1.0, 2.0, 3.0];
        let mut block = MonoViewMut::from_slice(&mut data);
        block.set_num_frames_visible(10);
    }
}
