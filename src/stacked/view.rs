use core::mem::MaybeUninit;
use rtsan_standalone::nonblocking;
use std::marker::PhantomData;

use crate::{AudioBlock, Sample};

/// A read-only view of stacked / separate-channel audio data.
///
/// * **Layout:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
/// * **Interpretation:** Each channel has its own separate buffer or array.
/// * **Terminology:** Also described as “planar” or “channels first” though more specifically it’s channel-isolated buffers.
/// * **Usage:** Very common in real-time DSP, as it simplifies memory access and can improve SIMD/vectorization efficiency.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let data = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
///
/// let block = StackedView::from_slice(&data);
///
/// block.channel(0).for_each(|&v| assert_eq!(v, 0.0));
/// block.channel(1).for_each(|&v| assert_eq!(v, 1.0));
/// ```
pub struct StackedView<'a, S: Sample, V: AsRef<[S]>> {
    data: &'a [V],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
    _phantom: PhantomData<S>,
}

impl<'a, S: Sample, V: AsRef<[S]>> StackedView<'a, S, V> {
    /// Creates a new [`StackedView`] from a slice of stacked audio data.
    ///
    /// # Parameters
    /// * `data` - The slice containing stacked audio samples (one slice per channel)
    ///
    /// # Panics
    /// Panics if the channel slices have different lengths.
    #[nonblocking]
    pub fn from_slice(data: &'a [V]) -> Self {
        let num_frames_available = if data.is_empty() {
            0
        } else {
            data[0].as_ref().len()
        };
        Self::from_slice_limited(data, data.len() as u16, num_frames_available)
    }

    /// Creates a new [`StackedView`] from a slice with limited visibility.
    ///
    /// This function allows creating a view that exposes only a subset of the allocated channels
    /// and frames, which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The slice containing stacked audio samples (one slice per channel)
    /// * `num_channels_visible` - Number of audio channels to expose in the view
    /// * `num_frames_visible` - Number of audio frames to expose in the view
    ///
    /// # Panics
    /// * Panics if `num_channels_visible` exceeds the number of channels in `data`
    /// * Panics if `num_frames_visible` exceeds the length of any channel buffer
    /// * Panics if channel slices have different lengths
    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a [V],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_allocated = data.len() as u16;
        let num_frames_allocated = if num_channels_allocated == 0 {
            0
        } else {
            data[0].as_ref().len()
        };
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        data.iter()
            .for_each(|v| assert_eq!(v.as_ref().len(), num_frames_allocated));

        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
            _phantom: PhantomData,
        }
    }
}

impl<S: Sample, V: AsRef<[S]>> AudioBlock<S> for StackedView<'_, S, V> {
    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.num_channels
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
                .get_unchecked(channel as usize)
                .as_ref()
                .get_unchecked(frame)
        }
    }

    #[nonblocking]
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked(channel as usize)
                .as_ref()
                .iter()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_frames = self.num_frames; // Capture num_frames for the closure
        self.data
            .iter()
            // Limit to the active number of channels
            .take(self.num_channels as usize)
            // For each channel slice, create an iterator over its samples
            .map(move |channel_data| channel_data.as_ref().iter().take(num_frames))
    }

    #[nonblocking]
    fn channel_slice(&self, channel: u16) -> Option<&[S]> {
        assert!(channel < self.num_channels);
        Some(&self.data[channel as usize].as_ref()[..self.num_frames])
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.as_ref().get_unchecked(frame) })
    }

    #[nonblocking]
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &'_ S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let data_slice: &[V] = self.data;

        (0..num_frames).map(move |frame_idx| {
            // For each frame index, create an iterator over the relevant channel views.
            data_slice[..num_channels]
                .iter() // Yields `&'a V`
                .map(move |channel_view: &V| {
                    // Get the immutable slice `&[S]` from the view using AsRef.
                    let channel_slice: &[S] = channel_view.as_ref();
                    // Access the sample immutably using safe indexing.
                    // Assumes frame_idx is valid based on outer loop and struct invariants.
                    &channel_slice[frame_idx]
                    // For max performance (if bounds are absolutely guaranteed):
                    // unsafe { channel_slice.get_unchecked(frame_idx) }
                })
        })
    }

    #[nonblocking]
    fn view(&self) -> impl AudioBlock<S> {
        StackedView::<S, V>::from_slice_limited(self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Stacked
    }

    #[nonblocking]
    fn raw_data(&self, stacked_ch: Option<u16>) -> &[S] {
        let ch = stacked_ch.expect("For stacked layout channel needs to be provided!");
        assert!(ch < self.num_channels_allocated);
        unsafe { self.data.get_unchecked(ch as usize).as_ref() }
    }
}

/// Adapter for creating stacked audio block views from raw pointers.
///
/// This adapter provides a safe interface to work with audio data stored in external buffers,
/// which is common when interfacing with audio APIs or hardware.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// // Create sample data for two channels with five frames each
/// let ch1 = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
/// let ch2 = vec![5.0f32, 6.0, 7.0, 8.0, 9.0];
///
/// // Create pointers to the channel data
/// let ptrs = [ch1.as_ptr(), ch2.as_ptr()];
/// let data = ptrs.as_ptr();
/// let num_channels = 2u16;
/// let num_frames = 5;
///
/// // Create an adapter from raw pointers to audio channel data
/// let adapter = unsafe { StackedPtrAdapter::<f32, 16>::from_ptr(data, num_channels, num_frames) };
///
/// // Get a safe view of the audio data
/// let block = adapter.stacked_view();
///
/// // Verify the data access works
/// assert_eq!(block.sample(0, 2), 2.0);
/// assert_eq!(block.sample(1, 3), 8.0);
/// ```
///
/// # Safety
///
/// When creating an adapter from raw pointers, you must ensure that:
/// - The pointers are valid and properly aligned
/// - The memory they point to remains valid for the lifetime of the adapter
/// - The channel count doesn't exceed the adapter's `MAX_CHANNELS` capacity
pub struct StackedPtrAdapter<'a, S: Sample, const MAX_CHANNELS: usize> {
    data: [MaybeUninit<&'a [S]>; MAX_CHANNELS],
    num_channels: u16,
}

impl<'a, S: Sample, const MAX_CHANNELS: usize> StackedPtrAdapter<'a, S, MAX_CHANNELS> {
    /// Creates new StackedPtrAdapter from raw pointers.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to an array of pointers
    /// - The array must contain at least `num_channels` valid pointers
    /// - Each pointer in the array must point to a valid array of samples with `num_frames` length
    /// - The pointed memory must remain valid for the lifetime of the returned adapter
    /// - The data must not be modified through other pointers for the lifetime of the returned adapter
    #[nonblocking]
    #[nonblocking]
    pub unsafe fn from_ptr(ptr: *const *const S, num_channels: u16, num_frames: usize) -> Self {
        assert!(
            num_channels as usize <= MAX_CHANNELS,
            "num_channels exceeds MAX_CHANNELS"
        );

        let mut data: [core::mem::MaybeUninit<&'a [S]>; MAX_CHANNELS] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() }; // Or other safe initialization

        // SAFETY: Caller guarantees `ptr` is valid for `num_channels` elements.
        let ptr_slice: &[*const S] =
            unsafe { core::slice::from_raw_parts(ptr, num_channels as usize) };

        for ch in 0..num_channels as usize {
            // SAFETY: See previous explanation
            data[ch].write(unsafe { core::slice::from_raw_parts(ptr_slice[ch], num_frames) });
        }

        Self { data, num_channels }
    }

    /// Returns a slice of references to the initialized channel data buffers.
    ///
    /// This method provides access to the underlying audio data as a slice of slices,
    /// with each inner slice representing one audio channel.
    #[inline]
    pub fn data_slice(&self) -> &[&'a [S]] {
        let initialized_part: &[MaybeUninit<&'a [S]>] = &self.data[..self.num_channels as usize];
        unsafe {
            core::slice::from_raw_parts(
                initialized_part.as_ptr() as *const &'a [S],
                self.num_channels as usize,
            )
        }
    }

    /// Creates a safe [`StackedView`] for accessing the audio data.
    ///
    /// This provides a convenient way to interact with the audio data through
    /// the full [`AudioBlock`] interface, enabling operations like iterating
    /// through channels or frames.
    ///
    /// # Returns
    ///
    /// A [`StackedView`] that provides safe, immutable access to the audio data.
    #[nonblocking]
    pub fn stacked_view(&self) -> StackedView<'a, S, &[S]> {
        StackedView::from_slice(self.data_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples() {
        let ch1 = &[0.0, 1.0, 2.0, 3.0, 4.0];
        let ch2 = &[5.0, 6.0, 7.0, 8.0, 9.0];
        let data = [ch1, ch2];
        let block = StackedView::from_slice(&data);

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
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1, ch2];
        let block = StackedView::from_slice(&data);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_channels() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1, ch2];
        let block = StackedView::from_slice(&data);

        let mut channels_iter = block.channels();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        assert!(channels_iter.next().is_none());
    }

    #[test]
    fn test_frame() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slice(&data);

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
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slice(&data);

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
    fn test_from_vec() {
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slice(&vec);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
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
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slice(&vec);
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
        let data = vec![vec![0.0; 4]; 3];

        let block = StackedView::from_slice_limited(&data, 2, 3);

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
    fn test_pointer() {
        unsafe {
            let num_channels = 2;
            let num_frames = 5;
            let mut vec = [vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];

            let ptr_vec: Vec<*const f32> =
                vec.iter_mut().map(|inner_vec| inner_vec.as_ptr()).collect();
            let ptr = ptr_vec.as_ptr();

            let adapter = StackedPtrAdapter::<_, 16>::from_ptr(ptr, num_channels, num_frames);
            let stacked = adapter.stacked_view();

            assert_eq!(
                stacked.channel(0).copied().collect::<Vec<_>>(),
                vec![0.0, 2.0, 4.0, 6.0, 8.0]
            );

            assert_eq!(
                stacked.channel(1).copied().collect::<Vec<_>>(),
                vec![1.0, 3.0, 5.0, 7.0, 9.0]
            );
        }
    }

    #[test]
    fn test_slice() {
        let data = [[1.0; 4], [2.0; 4], [0.0; 4]];
        let block = StackedView::from_slice_limited(&data, 2, 3);

        assert!(block.frame_slice(0).is_none());

        assert_eq!(block.channel_slice(0).unwrap(), &[1.0; 3]);
        assert_eq!(block.channel_slice(1).unwrap(), &[2.0; 3]);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds() {
        let data = [[1.0; 4], [2.0; 4], [0.0; 4]];
        let block = StackedView::from_slice_limited(&data, 2, 3);

        block.channel_slice(2);
    }

    #[test]
    fn test_raw_data() {
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slice(&vec);

        assert_eq!(block.layout(), crate::BlockLayout::Stacked);

        assert_eq!(block.raw_data(Some(0)), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data(Some(1)), &[1.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
