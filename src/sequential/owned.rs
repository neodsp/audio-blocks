use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
use core::{marker::PhantomData, ptr::NonNull};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

use super::{view::AudioBlockSequentialView, view_mut::AudioBlockSequentialViewMut};
use crate::{
    AudioBlock, AudioBlockMut, Sample,
    iter::{StridedSampleIter, StridedSampleIterMut},
};

/// A sequential audio block that owns its data.
///
/// * **Layout:** `[ch0, ch0, ch0, ch1, ch1, ch1]`
/// * **Interpretation:** All samples from `ch0` are stored first, followed by all from `ch1`, etc.
/// * **Terminology:** Described as “channels first” in the sense that all data for one channel appears before any data for the next.
/// * **Usage:** Used in DSP pipelines where per-channel processing is easier and more efficient.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let block = AudioBlockSequential::new(2, 3);
/// let mut block = AudioBlockSequential::from_block(&block);
///
/// block.channel_mut(0).fill(0.0);
/// block.channel_mut(1).fill(1.0);
///
/// assert_eq!(block.channel(0), &[0.0, 0.0, 0.0]);
/// assert_eq!(block.channel(1), &[1.0, 1.0, 1.0]);
/// ```
pub struct AudioBlockSequential<S: Sample> {
    data: Box<[S]>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample + Default> AudioBlockSequential<S> {
    /// Creates a new audio block with the specified dimensions.
    ///
    /// Allocates memory with exactly the specified number of channels and
    /// frames. The block is initialized with zeros.
    ///
    /// Do not use in real-time processes!
    ///
    /// # Arguments
    ///
    /// * `num_channels` - The number of audio channels
    /// * `num_frames` - The number of frames per channel
    ///
    /// # Panics
    ///
    /// Panics if the multiplication of `num_channels` and `num_frames` would overflow a usize.
    #[blocking]
    pub fn new(num_channels: u16, num_frames: usize) -> Self {
        let total_samples = (num_channels as usize)
            .checked_mul(num_frames)
            .expect("Multiplication overflow: num_channels * num_frames is too large");
        Self {
            data: vec![S::default(); total_samples].into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }
}

impl<S: Sample> AudioBlockSequential<S> {
    /// Creates a new sequential audio block by copying the data from a slice of sequential audio data.
    ///
    /// # Parameters
    /// * `data` - The slice containing sequential audio samples
    /// * `num_channels` - Number of audio channels in the data
    ///
    /// # Panics
    /// Panics if the length of `data` is not evenly divisible by `num_channels`.
    #[blocking]
    pub fn from_slice(data: &[S], num_channels: u16) -> Self {
        assert!(
            num_channels > 0 && data.len() % num_channels as usize == 0,
            "data length {} must be divisible by num_channels {}",
            data.len(),
            num_channels
        );
        let num_frames = data.len() / num_channels as usize;
        Self {
            data: data.to_vec().into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new sequential audio block by copying the data from a slice of sequential audio data with limited visibility.
    ///
    /// This function allows creating a block that exposes only a subset of the allocated channels
    /// and frames, which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The slice containing sequential audio samples
    /// * `num_channels_visible` - Number of audio channels to expose
    /// * `num_frames_visible` - Number of audio frames to expose
    /// * `num_channels_allocated` - Total number of channels allocated in the data buffer
    /// * `num_frames_allocated` - Total number of frames allocated in the data buffer
    ///
    /// # Panics
    /// * Panics if the length of `data` doesn't equal `num_channels_allocated * num_frames_allocated`
    /// * Panics if `num_channels_visible` exceeds `num_channels_allocated`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    #[blocking]
    pub fn from_slice_limited(
        data: &[S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            num_channels_allocated as usize * num_frames_allocated
        );
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: data.to_vec().into_boxed_slice(),
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }

    /// Creates a new audio block by copying data from another [`AudioBlock`].
    ///
    /// Converts any [`AudioBlock`] implementation to a sequential format by iterating
    /// through each channel of the source block and copying its samples. The new block
    /// will have the same dimensions as the source block.
    ///
    /// # Warning
    ///
    /// This function allocates memory and should not be used in real-time audio processing contexts.
    ///
    /// # Arguments
    ///
    /// * `block` - The source audio block to copy data from
    #[blocking]
    pub fn from_block(block: &impl AudioBlock<S>) -> Self {
        let mut data = Vec::with_capacity(block.num_channels() as usize * block.num_frames());
        block
            .channels_iter()
            .for_each(|c| c.for_each(|&v| data.push(v)));
        Self {
            data: data.into_boxed_slice(),
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }

    /// Returns a slice for a single channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    #[nonblocking]
    pub fn channel(&self, channel: u16) -> &[S] {
        assert!(channel < self.num_channels);
        let start = channel as usize * self.num_frames_allocated;
        let end = start + self.num_frames;
        &self.data[start..end]
    }

    /// Returns a mutable slice for a single channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    #[nonblocking]
    pub fn channel_mut(&mut self, channel: u16) -> &mut [S] {
        assert!(channel < self.num_channels);
        let start = channel as usize * self.num_frames_allocated;
        let end = start + self.num_frames;
        &mut self.data[start..end]
    }

    /// Returns an iterator over all channels in the block.
    ///
    /// Each channel is represented as a slice of samples.
    #[nonblocking]
    pub fn channels(&self) -> impl ExactSizeIterator<Item = &[S]> {
        self.data
            .chunks(self.num_frames_allocated)
            .take(self.num_channels as usize)
            .map(|frame| &frame[..self.num_frames])
    }

    /// Returns a mutable iterator over all channels in the block.
    ///
    /// Each channel is represented as a mutable slice of samples.
    #[nonblocking]
    pub fn channels_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [S]> {
        self.data
            .chunks_mut(self.num_frames_allocated)
            .take(self.num_channels as usize)
            .map(|frame| &mut frame[..self.num_frames])
    }

    /// Provides direct access to the underlying memory as a sequential slice.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data(&self) -> &[S] {
        &self.data
    }

    /// Provides direct mutable access to the underlying memory as a sequential slice.
    ///
    /// This function gives mutable access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [S] {
        &mut self.data
    }

    #[nonblocking]
    pub fn view(&self) -> AudioBlockSequentialView<'_, S> {
        AudioBlockSequentialView::from_slice_limited(
            &self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> AudioBlockSequentialViewMut<'_, S> {
        AudioBlockSequentialViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockSequential<S> {
    type PlanarView = [S; 0];

    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn num_channels_allocated(&self) -> u16 {
        self.num_channels_allocated
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
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            *self
                .data
                .get_unchecked(channel as usize * self.num_frames_allocated + frame)
        }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &S>>
    {
        let num_frames = self.num_frames; // Visible frames per channel
        let num_frames_allocated = self.num_frames_allocated; // Allocated frames per channel (chunk size)

        self.data
            .chunks(num_frames_allocated)
            .take(self.num_channels as usize)
            .map(move |channel_chunk| channel_chunk.iter().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_iter(
        &self,
    ) -> impl '_ + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &S>>
    {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_frames_allocated;
        let data_ptr = self.data.as_ptr();

        (0..num_frames).map(move |frame_idx| {
            // Safety check: Ensure data isn't empty if we calculate a start_ptr.
            // If num_frames or num_channels is 0, remaining will be 0, iterator is safe.
            // If data is empty, ptr is dangling, but add(0) is okay. add(>0) is UB.
            // But if data is empty, num_channels or num_frames must be 0.
            let start_ptr = if self.data.is_empty() {
                NonNull::dangling().as_ptr() // Use dangling pointer if slice is empty
            } else {
                // Safety: channel_idx is < num_channels <= num_channels_allocated.
                // Adding it to a valid data_ptr is safe within slice bounds.
                unsafe { data_ptr.add(frame_idx) }
            };

            StridedSampleIter::<'_, S> {
                // Note: '_ lifetime from &self borrow
                // Safety: Pointer is either dangling (if empty) or valid start pointer.
                // NonNull::new is safe if start_ptr is non-null (i.e., data not empty).
                ptr: NonNull::new(start_ptr as *mut S).unwrap_or(NonNull::dangling()), // Use dangling on null/empty
                stride,
                remaining: num_channels, // If 0, iterator yields None immediately
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }

    #[nonblocking]
    fn as_sequential_view(&self) -> Option<AudioBlockSequentialView<'_, S>> {
        Some(self.view())
    }
}

impl<S: Sample> AudioBlockMut<S> for AudioBlockSequential<S> {
    type PlanarViewMut = [S; 0];

    #[nonblocking]
    fn set_num_channels_visible(&mut self, num_channels: u16) {
        assert!(num_channels <= self.num_channels_allocated);
        self.num_channels = num_channels;
    }

    #[nonblocking]
    fn set_num_frames_visible(&mut self, num_frames: usize) {
        assert!(num_frames <= self.num_frames_allocated);
        self.num_frames = num_frames;
    }

    #[nonblocking]
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            self.data
                .get_unchecked_mut(channel as usize * self.num_frames_allocated + frame)
        }
    }

    #[nonblocking]
    fn channel_iter_mut(
        &mut self,
        channel: u16,
    ) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data
            .iter_mut()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl '_
    + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &mut S>> {
        let num_frames = self.num_frames;
        let num_frames_allocated = self.num_frames_allocated;
        self.data
            .chunks_mut(num_frames_allocated)
            .take(self.num_channels as usize)
            .map(move |channel_chunk| channel_chunk.iter_mut().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_iter_mut(
        &mut self,
    ) -> impl '_
    + ExactSizeIterator<Item = impl '_ + ExactSizeIterator<Item = &mut S>> {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_frames_allocated;
        let data_ptr = self.data.as_mut_ptr();

        (0..num_frames).map(move |frame_idx| {
            // Safety check: Ensure data isn't empty if we calculate a start_ptr.
            // If num_frames or num_channels is 0, remaining will be 0, iterator is safe.
            // If data is empty, ptr is dangling, but add(0) is okay. add(>0) is UB.
            // But if data is empty, num_channels or num_frames must be 0.
            let start_ptr = if self.data.is_empty() {
                NonNull::dangling().as_ptr() // Use dangling pointer if slice is empty
            } else {
                // Safety: channel_idx is < num_channels <= num_channels_allocated.
                // Adding it to a valid data_ptr is safe within slice bounds.
                unsafe { data_ptr.add(frame_idx) }
            };

            StridedSampleIterMut::<'_, S> {
                // Note: '_ lifetime from &self borrow
                // Safety: Pointer is either dangling (if empty) or valid start pointer.
                // NonNull::new is safe if start_ptr is non-null (i.e., data not empty).
                ptr: NonNull::new(start_ptr).unwrap_or(NonNull::dangling()), // Use dangling on null/empty
                stride,
                remaining: num_channels, // If 0, iterator yields None immediately
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }

    #[nonblocking]
    fn as_sequential_view_mut(&mut self) -> Option<AudioBlockSequentialViewMut<'_, S>> {
        Some(self.view_mut())
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for AudioBlockSequential<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "AudioBlockSequential {{")?;
        writeln!(f, "  num_channels: {}", self.num_channels)?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(
            f,
            "  num_channels_allocated: {}",
            self.num_channels_allocated
        )?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  channels:")?;

        for (i, channel) in self.channels().enumerate() {
            writeln!(f, "    {}: {:?}", i, channel)?;
        }

        writeln!(f, "  raw_data: {:?}", self.raw_data())?;
        writeln!(f, "}}")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleaved::AudioBlockInterleavedView;
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_member_functions() {
        let mut block = AudioBlockSequential::<f32>::new(3, 4);
        block.channel_mut(0).copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        block.channel_mut(1).copy_from_slice(&[4.0, 5.0, 6.0, 7.0]);
        block.set_visible(2, 3);

        // single frame
        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[4.0, 5.0, 6.0]);

        assert_eq!(block.channel_mut(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel_mut(1), &[4.0, 5.0, 6.0]);

        // all frames
        let mut channels = block.channels();
        assert_eq!(channels.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(channels.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(channels.next(), None);
        drop(channels);

        let mut channels = block.channels_mut();
        assert_eq!(channels.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(channels.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(channels.next(), None);
        drop(channels);

        // raw data
        assert_eq!(
            block.raw_data(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
        );

        // views
        let view = block.view();
        assert_eq!(view.num_channels(), block.num_channels());
        assert_eq!(view.num_frames(), block.num_frames());
        assert_eq!(
            view.num_channels_allocated(),
            block.num_channels_allocated()
        );
        assert_eq!(view.num_frames_allocated(), block.num_frames_allocated());
        assert_eq!(view.raw_data(), block.raw_data());

        let num_channels = block.num_channels();
        let num_frames = block.num_frames();
        let num_channels_allocated = block.num_channels_allocated();
        let num_frames_allocated = block.num_frames_allocated();
        let data = block.raw_data().to_vec();
        let view = block.view_mut();
        assert_eq!(view.num_channels(), num_channels);
        assert_eq!(view.num_frames(), num_frames);
        assert_eq!(view.num_channels_allocated(), num_channels_allocated);
        assert_eq!(view.num_frames_allocated(), num_frames_allocated);
        assert_eq!(view.raw_data(), &data);
    }

    #[test]
    fn test_samples() {
        let mut block = AudioBlockSequential::<f32>::new(2, 5);

        let num_frames = block.num_frames();
        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                *block.sample_mut(ch, f) = (ch as usize * num_frames + f) as f32;
            }
        }

        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                assert_eq!(block.sample(ch, f), (ch as usize * num_frames + f) as f32);
            }
        }

        assert_eq!(
            block.raw_data(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_channel_iter() {
        let mut block = AudioBlockSequential::<f32>::new(2, 5);

        let channel = block.channel_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = block.channel_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        block
            .channel_iter_mut(0)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        block
            .channel_iter_mut(1)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);

        let channel = block.channel_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = block.channel_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channel_iters() {
        let mut block = AudioBlockSequential::<f32>::new(2, 5);

        let mut channels_iter = block.channels_iter();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        let mut channels_iter = block.channels_iter_mut();
        channels_iter
            .next()
            .unwrap()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        channels_iter
            .next()
            .unwrap()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        let mut channels_iter = block.channels_iter();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);
    }

    #[test]
    fn test_frame_iter() {
        let mut block = AudioBlockSequential::<f32>::new(2, 5);

        for i in 0..block.num_frames() {
            let frame = block.frame_iter(i).copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }

        for i in 0..block.num_frames() {
            let add = i as f32 * 10.0;
            block
                .frame_iter_mut(i)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }

        let channel = block.frame_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = block.frame_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0]);
        let channel = block.frame_iter(2).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![20.0, 21.0]);
        let channel = block.frame_iter(3).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![30.0, 31.0]);
        let channel = block.frame_iter(4).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![40.0, 41.0]);
    }

    #[test]
    fn test_frame_iters() {
        let mut block = AudioBlockSequential::<f32>::new(3, 6);
        block.set_visible(2, 5);

        let num_frames = block.num_frames;
        let mut frames_iter = block.frames_iter();
        for _ in 0..num_frames {
            let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        let mut frames_iter = block.frames_iter_mut();
        for i in 0..num_frames {
            let add = i as f32 * 10.0;
            frames_iter
                .next()
                .unwrap()
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        let mut frames_iter = block.frames_iter();
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![0.0, 1.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![10.0, 11.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![20.0, 21.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![30.0, 31.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![40.0, 41.0]);
        assert!(frames_iter.next().is_none());
    }

    #[test]
    fn test_from_slice() {
        let block =
            AudioBlockSequential::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                2,
            ));
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(
            block.frame_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0]
        );
        assert_eq!(
            block.frame_iter(1).copied().collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
        assert_eq!(
            block.frame_iter(2).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0]
        );
        assert_eq!(
            block.frame_iter(3).copied().collect::<Vec<_>>(),
            vec![6.0, 7.0]
        );
        assert_eq!(
            block.frame_iter(4).copied().collect::<Vec<_>>(),
            vec![8.0, 9.0]
        );
    }

    #[test]
    fn test_view() {
        let block =
            AudioBlockSequential::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                2,
            ));
        assert!(block.as_interleaved_view().is_none());
        assert!(block.as_planar_view().is_none());
        assert!(block.as_sequential_view().is_some());
        let view = block.as_view();
        assert_eq!(
            view.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            view.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_view_mut() {
        let mut block = AudioBlockSequential::<f32>::new(2, 5);
        assert!(block.as_interleaved_view().is_none());
        assert!(block.as_planar_view().is_none());
        assert!(block.as_sequential_view().is_some());
        {
            let mut view = block.as_view_mut();
            view.channel_iter_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32);
            view.channel_iter_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + 10.0);
        }

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0, 13.0, 14.0]
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds() {
        let mut block = AudioBlockSequential::<f32>::new(3, 6);
        block.set_visible(2, 5);
        block.channel(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds_mut() {
        let mut block = AudioBlockSequential::<f32>::new(3, 6);
        block.set_visible(2, 5);
        block.channel_mut(2);
    }

    #[test]
    fn test_resize() {
        let mut block = AudioBlockSequential::<f32>::new(3, 10);
        assert_eq!(block.num_channels(), 3);
        assert_eq!(block.num_frames(), 10);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel_iter(i).count(), 10);
            assert_eq!(block.channel_iter_mut(i).count(), 10);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame_iter(i).count(), 3);
            assert_eq!(block.frame_iter_mut(i).count(), 3);
        }

        block.set_visible(3, 10);
        block.set_visible(2, 5);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel_iter(i).count(), 5);
            assert_eq!(block.channel_iter_mut(i).count(), 5);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame_iter(i).count(), 2);
            assert_eq!(block.frame_iter_mut(i).count(), 2);
        }
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_channels() {
        let mut block = AudioBlockSequential::<f32>::new(2, 10);
        block.set_visible(3, 10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut block = AudioBlockSequential::<f32>::new(2, 10);
        block.set_visible(2, 11);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut block = AudioBlockSequential::<f32>::new(2, 10);
        block.set_visible(1, 10);
        let _ = block.channel_iter(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame() {
        let mut block = AudioBlockSequential::<f32>::new(2, 10);
        block.set_visible(2, 5);
        let _ = block.frame_iter(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel_mut() {
        let mut block = AudioBlockSequential::<f32>::new(2, 10);
        block.set_visible(1, 10);
        let _ = block.channel_iter_mut(1);
    }
}
