use rtsan_standalone::nonblocking;

use core::{marker::PhantomData, ptr::NonNull};

use crate::{AudioBlock, Sample, iter::StridedSampleIter};

/// A read-only view of interleaved audio data.
///
/// * **Layout:** `[ch0, ch1, ch0, ch1, ch0, ch1]`
/// * **Interpretation:** Each group of channel samples represents a frame. So, this layout stores frames one after another.
/// * **Terminology:** Described as “packed” or “frames first” because each time step is grouped and processed as a unit (a frame).
/// * **Usage:** Often used in APIs or hardware-level interfaces, where synchronized playback across channels is crucial.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
///
/// let block = InterleavedView::from_slice(&data, 2, 3);
///
/// block.channel(0).for_each(|&v| assert_eq!(v, 0.0));
/// block.channel(1).for_each(|&v| assert_eq!(v, 1.0));
/// ```
pub struct AudioBlockInterleavedView<'a, S: Sample> {
    data: &'a [S],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<'a, S: Sample> AudioBlockInterleavedView<'a, S> {
    /// Creates a new interleaved audio block from a slice of interleaved audio data.
    ///
    /// # Parameters
    /// * `data` - The slice containing interleaved audio samples
    /// * `num_channels` - Number of audio channels in the data
    /// * `num_frames` - Number of audio frames in the data
    ///
    /// # Panics
    /// Panics if the length of `data` doesn't equal `num_channels * num_frames`.
    #[nonblocking]
    pub fn from_slice(data: &'a [S], num_channels: u16, num_frames: usize) -> Self {
        assert_eq!(data.len(), num_channels as usize * num_frames);
        Self {
            data,
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new interleaved audio block from a slice with limited visibility.
    ///
    /// This function allows creating a view that exposes only a subset of the allocated channels
    /// and frames, which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The slice containing interleaved audio samples
    /// * `num_channels_visible` - Number of audio channels to expose in the view
    /// * `num_frames_visible` - Number of audio frames to expose in the view
    /// * `num_channels_allocated` - Total number of channels allocated in the data buffer
    /// * `num_frames_allocated` - Total number of frames allocated in the data buffer
    ///
    /// # Panics
    /// * Panics if the length of `data` doesn't equal `num_channels_allocated * num_frames_allocated`
    /// * Panics if `num_channels_visible` exceeds `num_channels_allocated`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a [S],
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
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }

    /// Creates a new interleaved audio block from a pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_ptr(ptr: *const S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: unsafe { std::slice::from_raw_parts(ptr, num_channels as usize * num_frames) },
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new audio block from a pointer with a limited amount of channels and/or frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_ptr_limited(
        ptr: *const S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: unsafe {
                std::slice::from_raw_parts(
                    ptr,
                    num_channels_allocated as usize * num_frames_allocated,
                )
            },
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockInterleavedView<'_, S> {
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
                .get_unchecked(frame * self.num_channels_allocated as usize + channel as usize)
        }
    }

    #[nonblocking]
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize)
            .step_by(self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_channels_allocated as usize;
        let data_ptr = self.data.as_ptr();

        (0..num_channels).map(move |channel_idx| {
            // Safety check: Ensure data isn't empty if we calculate a start_ptr.
            // If num_frames or num_channels is 0, remaining will be 0, iterator is safe.
            // If data is empty, ptr is dangling, but add(0) is okay. add(>0) is UB.
            // But if data is empty, num_channels or num_frames must be 0.
            let start_ptr = if self.data.is_empty() {
                NonNull::dangling().as_ptr() // Use dangling pointer if slice is empty
            } else {
                // Safety: channel_idx is < num_channels <= num_channels_allocated.
                // Adding it to a valid data_ptr is safe within slice bounds.
                unsafe { data_ptr.add(channel_idx) }
            };

            StridedSampleIter::<'_, S> {
                // Note: '_ lifetime from &self borrow
                // Safety: Pointer is either dangling (if empty) or valid start pointer.
                // NonNull::new is safe if start_ptr is non-null (i.e., data not empty).
                ptr: NonNull::new(start_ptr as *mut S).unwrap_or(NonNull::dangling()), // Use dangling on null/empty
                stride,
                remaining: num_frames, // If 0, iterator yields None immediately
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame * self.num_channels_allocated as usize)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_channels_allocated = self.num_channels_allocated as usize;
        self.data
            .chunks(num_channels_allocated)
            .take(self.num_frames)
            .map(move |channel_chunk| channel_chunk.iter().take(num_channels))
    }

    #[nonblocking]
    fn frame_slice(&self, frame: usize) -> Option<&[S]> {
        assert!(frame < self.num_frames);
        let start = frame * self.num_channels_allocated as usize;
        let end = start + self.num_channels as usize;
        Some(&self.data[start..end])
    }

    #[nonblocking]
    fn view(&self) -> impl AudioBlock<S> {
        AudioBlockInterleavedView::from_slice_limited(
            self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Interleaved
    }

    #[nonblocking]
    fn raw_data(&self, _: Option<u16>) -> &[S] {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use super::*;

    #[test]
    fn test_samples() {
        let data = vec![0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                assert_eq!(
                    block.sample(ch, f),
                    (ch as usize * block.num_frames() + f) as f32
                );
            }
        }
    }

    #[test]
    fn test_channel() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_channels() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        let mut channels_iter = block.channels();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        assert!(channels_iter.next().is_none());
    }

    #[test]
    fn test_frame() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        let channel = block.frame(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = block.frame(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![2.0, 3.0]);
        let channel = block.frame(2).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![4.0, 5.0]);
        let channel = block.frame(3).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![6.0, 7.0]);
        let channel = block.frame(4).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![8.0, 9.0]);
    }

    #[test]
    fn test_frames() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        let mut frames_iter = block.frames();
        let channel = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![2.0, 3.0]);
        let channel = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![4.0, 5.0]);
        let channel = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![6.0, 7.0]);
        let channel = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![8.0, 9.0]);
        assert!(frames_iter.next().is_none());
    }

    #[test]
    fn test_from_slice() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
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
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);
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
    fn test_limited() {
        let data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let block = AudioBlockInterleavedView::from_slice_limited(&data, 2, 3, 3, 4);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
        }
    }

    #[test]
    fn test_from_raw() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = unsafe { AudioBlockInterleavedView::<f32>::from_ptr(data.as_mut_ptr(), 2, 5) };
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
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
    fn test_from_raw_limited() {
        let data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let block =
            unsafe { AudioBlockInterleavedView::from_ptr_limited(data.as_ptr(), 2, 3, 3, 4) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
        }
    }

    #[test]
    fn test_slice() {
        let data = [1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice_limited(&data, 2, 3, 3, 4);
        assert!(block.channel_slice(0).is_none());

        assert_eq!(block.frame_slice(0).unwrap(), &[1.0; 2]);
        assert_eq!(block.frame_slice(1).unwrap(), &[2.0; 2]);
        assert_eq!(block.frame_slice(2).unwrap(), &[3.0; 2]);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds() {
        let data = [1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice_limited(&data, 2, 3, 3, 4);
        block.frame_slice(5);
    }

    #[test]
    fn test_raw_data() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = AudioBlockInterleavedView::<f32>::from_slice(&data, 2, 5);

        assert_eq!(block.layout(), crate::BlockLayout::Interleaved);

        assert_eq!(
            block.raw_data(None),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }
}
