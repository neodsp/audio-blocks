use rtsan_standalone::nonblocking;

use crate::{AudioBlock, Sample};

/// A read-only view of mono (single-channel) audio data.
///
/// This provides a lightweight, non-owning reference to a slice of mono audio samples.
///
/// * **Layout:** `[sample0, sample1, sample2, ...]`
/// * **Interpretation:** A simple sequence of samples representing a single audio channel.
/// * **Usage:** Ideal for mono audio processing, side-chain signals, or any single-channel audio data.
///
/// # Example
///
/// ```
/// use audio_blocks::{mono::AudioBlockMonoView, AudioBlock};
///
/// let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let block = AudioBlockMonoView::from_slice(&data);
///
/// assert_eq!(block.sample(0), 0.0);
/// assert_eq!(block.sample(4), 4.0);
/// assert_eq!(block.num_frames(), 5);
/// ```
pub struct AudioBlockMonoView<'a, S: Sample> {
    data: &'a [S],
    num_frames: usize,
    num_frames_allocated: usize,
}

impl<'a, S: Sample> AudioBlockMonoView<'a, S> {
    /// Creates a new mono audio block view from a slice of audio samples.
    ///
    /// # Parameters
    /// * `data` - The slice containing mono audio samples
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::AudioBlockMonoView, AudioBlock};
    ///
    /// let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let block = AudioBlockMonoView::from_slice(&samples);
    /// assert_eq!(block.num_frames(), 5);
    /// ```
    #[nonblocking]
    pub fn from_slice(data: &'a [S]) -> Self {
        let num_frames = data.len();
        Self {
            data,
            num_frames,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new mono audio block view from a slice with limited visibility.
    ///
    /// This function allows creating a view that exposes only a subset of the allocated frames,
    /// which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The slice containing mono audio samples
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
    /// use audio_blocks::{mono::AudioBlockMonoView, AudioBlock};
    ///
    /// let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let block = AudioBlockMonoView::from_slice_limited(&samples, 3, 5);
    /// assert_eq!(block.num_frames(), 3);
    /// assert_eq!(block.num_frames_allocated(), 5);
    /// ```
    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a [S],
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

    /// Creates a new mono audio block view from a pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_frames` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned view
    /// - The memory must not be mutated through other pointers while this view exists
    ///
    /// # Example
    ///
    /// ```
    /// use audio_blocks::{mono::AudioBlockMonoView, AudioBlock};
    ///
    /// let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let block = unsafe { AudioBlockMonoView::from_ptr(samples.as_ptr(), 5) };
    /// assert_eq!(block.num_frames(), 5);
    /// ```
    #[nonblocking]
    pub unsafe fn from_ptr(ptr: *const S, num_frames: usize) -> Self {
        Self {
            data: unsafe { std::slice::from_raw_parts(ptr, num_frames) },
            num_frames,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new mono audio block view from a pointer with limited visibility.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_frames_allocated` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned view
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_ptr_limited(
        ptr: *const S,
        num_frames_visible: usize,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: unsafe { std::slice::from_raw_parts(ptr, num_frames_allocated) },
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

    /// Provides direct access to the underlying samples as a slice.
    ///
    /// Returns only the visible samples (up to `num_frames`).
    #[nonblocking]
    pub fn samples(&self) -> &[S] {
        &self.data[..self.num_frames]
    }

    /// Provides direct access to all allocated memory, including reserved capacity.
    #[nonblocking]
    pub fn raw_data(&self) -> &[S] {
        self.data
    }

    #[nonblocking]
    pub fn view(&self) -> AudioBlockMonoView<'_, S> {
        AudioBlockMonoView::from_slice_limited(
            self.data,
            self.num_frames,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockMonoView<'_, S> {
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
        assert_eq!(channel, 0, "AudioBlockMonoView only has channel 0");
        self.sample(frame)
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert_eq!(channel, 0, "AudioBlockMonoView only has channel 0");
        self.samples().iter()
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        core::iter::once(self.samples().iter())
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        core::iter::once(&self.data[frame])
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        self.data.iter().take(self.num_frames).map(core::iter::once)
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for AudioBlockMonoView<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "AudioBlockMonoView {{")?;
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
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice(&data);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_slice_limited() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice_limited(&data, 3, 5);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0]);
        assert_eq!(block.raw_data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_ptr() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = unsafe { AudioBlockMonoView::from_ptr(data.as_ptr(), 5) };
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_from_ptr_limited() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = unsafe { AudioBlockMonoView::from_ptr_limited(data.as_ptr(), 3, 5) };
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sample_access() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice(&data);
        assert_eq!(block.sample(0), 1.0);
        assert_eq!(block.sample(2), 3.0);
        assert_eq!(block.sample(4), 5.0);
    }

    #[test]
    fn test_samples_iter() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice(&data);
        assert_eq!(block.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_audio_block_trait() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice(&data);

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
    fn test_view() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let block = AudioBlockMonoView::from_slice(&data);
        let view = block.view();
        assert_eq!(view.num_frames(), 5);
        assert_eq!(view.samples(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_sample_out_of_bounds() {
        let data = [1.0, 2.0, 3.0];
        let block = AudioBlockMonoView::from_slice(&data);
        let _ = block.sample(10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let data = [1.0, 2.0, 3.0];
        let block = AudioBlockMonoView::from_slice(&data);
        let _ = block.channel_iter(1);
    }
}
