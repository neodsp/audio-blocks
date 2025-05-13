use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
use core::{marker::PhantomData, ptr::NonNull};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

use super::{view::SequentialView, view_mut::SequentialViewMut};
use crate::{
    AudioBlock, AudioBlockMut, Sample,
    iter::{StridedSampleIter, StridedSampleIterMut},
};

/// A sequential / planar audio block that owns its data.
///
/// * **Layout:** `[ch0, ch0, ch0, ch1, ch1, ch1]`
/// * **Interpretation:** All samples from `ch0` are stored first, followed by all from `ch1`, etc.
/// * **Terminology:** Described as “planar” or “channels first” in the sense that all data for one channel appears before any data for the next.
/// * **Usage:** Used in DSP pipelines where per-channel processing is easier and more efficient.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let block = Sequential::new(2, 3);
/// let mut block = Sequential::from_block(&block);
///
/// block.channel_mut(0).for_each(|v| *v = 0.0);
/// block.channel_mut(1).for_each(|v| *v = 1.0);
///
/// assert_eq!(block.raw_data(None), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
/// ```
pub struct Sequential<S: Sample> {
    data: Box<[S]>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> Sequential<S> {
    /// Creates a new [`Sequential`] audio block with the specified dimensions.
    ///
    /// Allocates memory for a new sequential audio block with exactly the specified
    /// number of channels and frames. The block is initialized with the default value
    /// for the sample type.
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

    /// Creates a new [`Sequential`] audio block by copying data from another [`AudioBlock`].
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
        block.channels().for_each(|c| c.for_each(|&v| data.push(v)));
        Self {
            data: data.into_boxed_slice(),
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }
}

impl<S: Sample> AudioBlock<S> for Sequential<S> {
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
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_frames = self.num_frames; // Active frames per channel
        let num_frames_allocated = self.num_frames_allocated; // Allocated frames per channel (chunk size)

        self.data
            .chunks(num_frames_allocated)
            .take(self.num_channels as usize)
            .map(move |channel_chunk| channel_chunk.iter().take(num_frames))
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
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
    fn view(&self) -> impl AudioBlock<S> {
        SequentialView::from_slice_limited(
            &self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Sequential
    }

    #[nonblocking]
    fn channel_slice(&self, channel: u16) -> Option<&[S]> {
        assert!(channel < self.num_channels);
        let start = channel as usize * self.num_frames_allocated;
        let end = start + self.num_frames;
        Some(&self.data[start..end])
    }

    #[nonblocking]
    fn raw_data(&self, _: Option<u16>) -> &[S] {
        &self.data
    }
}

impl<S: Sample> AudioBlockMut<S> for Sequential<S> {
    #[nonblocking]
    fn resize(&mut self, num_channels: u16, num_frames: usize) {
        assert!(num_channels <= self.num_channels_allocated);
        assert!(num_frames <= self.num_frames_allocated);
        self.num_channels = num_channels;
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
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data
            .iter_mut()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_frames = self.num_frames;
        let num_frames_allocated = self.num_frames_allocated;
        self.data
            .chunks_mut(num_frames_allocated)
            .take(self.num_channels as usize)
            .map(move |channel_chunk| channel_chunk.iter_mut().take(num_frames))
    }

    #[nonblocking]
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
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
    fn view_mut(&mut self) -> impl AudioBlockMut<S> {
        SequentialViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn channel_slice_mut(&mut self, channel: u16) -> Option<&mut [S]> {
        assert!(channel < self.num_channels);
        let start = channel as usize * self.num_frames_allocated;
        let end = start + self.num_frames;
        Some(&mut self.data[start..end])
    }

    #[nonblocking]
    fn raw_data_mut(&mut self, _: Option<u16>) -> &mut [S] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use super::*;
    use crate::interleaved::InterleavedView;

    #[test]
    fn test_samples() {
        let mut block = Sequential::<f32>::new(2, 5);

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
            block.raw_data(None),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_channel() {
        let mut block = Sequential::<f32>::new(2, 5);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        block
            .channel_mut(0)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        block
            .channel_mut(1)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channels() {
        let mut block = Sequential::<f32>::new(2, 5);

        let mut channels_iter = block.channels();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        let mut channels_iter = block.channels_mut();
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

        let mut channels_iter = block.channels();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);
    }

    #[test]
    fn test_frame() {
        let mut block = Sequential::<f32>::new(2, 5);

        for i in 0..block.num_frames() {
            let frame = block.frame(i).copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }

        for i in 0..block.num_frames() {
            let add = i as f32 * 10.0;
            block
                .frame_mut(i)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }

        let channel = block.frame(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = block.frame(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0]);
        let channel = block.frame(2).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![20.0, 21.0]);
        let channel = block.frame(3).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![30.0, 31.0]);
        let channel = block.frame(4).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![40.0, 41.0]);
    }

    #[test]
    fn test_frames() {
        let mut block = Sequential::<f32>::new(3, 6);
        block.resize(2, 5);

        let num_frames = block.num_frames;
        let mut frames_iter = block.frames();
        for _ in 0..num_frames {
            let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        let mut frames_iter = block.frames_mut();
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

        let mut frames_iter = block.frames();
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
        let block = Sequential::<f32>::from_block(&InterleavedView::from_slice(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            2,
            5,
        ));
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(block.frame(0).copied().collect::<Vec<_>>(), vec![0.0, 1.0]);
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
        assert_eq!(block.frame(2).copied().collect::<Vec<_>>(), vec![4.0, 5.0]);
        assert_eq!(block.frame(3).copied().collect::<Vec<_>>(), vec![6.0, 7.0]);
        assert_eq!(block.frame(4).copied().collect::<Vec<_>>(), vec![8.0, 9.0]);
    }

    #[test]
    fn test_view() {
        let block = Sequential::<f32>::from_block(&InterleavedView::from_slice(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            2,
            5,
        ));
        let view = block.view();
        assert_eq!(
            view.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            view.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_view_mut() {
        let mut block = Sequential::<f32>::new(2, 5);
        {
            let mut view = block.view_mut();
            view.channel_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32);
            view.channel_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + 10.0);
        }

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0, 13.0, 14.0]
        );
    }

    #[test]
    fn test_slice() {
        let mut block = Sequential::<f32>::new(3, 6);
        block.resize(2, 5);
        assert!(block.frame_slice(0).is_none());

        block.channel_slice_mut(0).unwrap().fill(1.0);
        block.channel_slice_mut(1).unwrap().fill(2.0);
        assert_eq!(block.channel_slice(0).unwrap(), &[1.0; 5]);
        assert_eq!(block.channel_slice(1).unwrap(), &[2.0; 5]);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let mut block = Sequential::<f32>::new(3, 6);
        block.resize(2, 5);
        block.channel_slice(2);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds_mut() {
        let mut block = Sequential::<f32>::new(3, 6);
        block.resize(2, 5);
        block.channel_slice_mut(2);
    }

    #[test]
    fn test_resize() {
        let mut block = Sequential::<f32>::new(3, 10);
        assert_eq!(block.num_channels(), 3);
        assert_eq!(block.num_frames(), 10);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 10);
            assert_eq!(block.channel_mut(i).count(), 10);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 3);
            assert_eq!(block.frame_mut(i).count(), 3);
        }

        block.resize(3, 10);
        block.resize(2, 5);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 5);
            assert_eq!(block.channel_mut(i).count(), 5);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
            assert_eq!(block.frame_mut(i).count(), 2);
        }
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_channels() {
        let mut block = Sequential::<f32>::new(2, 10);
        block.resize(3, 10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut block = Sequential::<f32>::new(2, 10);
        block.resize(2, 11);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut block = Sequential::<f32>::new(2, 10);
        block.resize(1, 10);
        let _ = block.channel(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame() {
        let mut block = Sequential::<f32>::new(2, 10);
        block.resize(2, 5);
        let _ = block.frame(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel_mut() {
        let mut block = Sequential::<f32>::new(2, 10);
        block.resize(1, 10);
        let _ = block.channel_mut(1);
    }
}
