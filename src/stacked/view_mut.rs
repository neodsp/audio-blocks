use core::mem::MaybeUninit;
use rtsan_standalone::nonblocking;
use std::marker::PhantomData;

use crate::{AudioBlock, AudioBlockMut, Sample};

use super::StackedView;

pub struct StackedViewMut<'a, S: Sample, V: AsMut<[S]> + AsRef<[S]>> {
    data: &'a mut [V],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
    _phantom: PhantomData<S>,
}

impl<'a, S: Sample, V: AsMut<[S]> + AsRef<[S]>> StackedViewMut<'a, S, V> {
    #[nonblocking]
    pub fn from_slice(data: &'a mut [V]) -> Self {
        let num_frames_available = if data.is_empty() {
            0
        } else {
            data[0].as_ref().len()
        };
        Self::from_slice_limited(data, data.len() as u16, num_frames_available)
    }

    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a mut [V],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_available = data.len();
        let num_frames_available = if num_channels_available == 0 {
            0
        } else {
            data[0].as_ref().len()
        };
        assert!(num_channels_visible <= num_channels_available as u16);
        assert!(num_frames_visible <= num_frames_available);
        data.iter()
            .for_each(|v| assert_eq!(v.as_ref().len(), num_frames_available));

        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available as u16,
            num_frames_allocated: num_frames_available,
            _phantom: PhantomData,
        }
    }
}

impl<S: Sample, V: AsMut<[S]> + AsRef<[S]>> AudioBlock<S> for StackedViewMut<'_, S, V> {
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
        // `self.data` is the field `&'data [V]`. We get `&'a [V]` from `&'a self`.
        let data_slice: &[V] = self.data;

        // Assumes the struct/caller guarantees that for all `chan` in `0..num_channels`,
        // `self.data[chan].as_ref().len() >= num_frames`.

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
        StackedView::from_slice_limited(self.data, self.num_channels, self.num_frames)
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

impl<S: Sample, V: AsMut<[S]> + AsRef<[S]>> AudioBlockMut<S> for StackedViewMut<'_, S, V> {
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
                .get_unchecked_mut(channel as usize)
                .as_mut()
                .get_unchecked_mut(frame)
        }
    }

    #[nonblocking]
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked_mut(channel as usize)
                .as_mut()
                .iter_mut()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_frames = self.num_frames;
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| channel_data.as_mut().iter_mut().take(num_frames))
    }

    #[nonblocking]
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.as_mut().get_unchecked_mut(frame) })
    }

    #[nonblocking]
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let data_slice: &mut [V] = self.data;
        let data_ptr: *mut [V] = data_slice;

        (0..num_frames).map(move |frame_idx| {
            // Re-borrow mutably inside the closure via the raw pointer.
            // Safety: Safe because the outer iterator executes this sequentially per frame.
            let current_channel_views: &mut [V] = unsafe { &mut *data_ptr };

            // Iterate over the relevant channel views up to num_channels.
            current_channel_views[..num_channels]
                .iter_mut() // Yields `&mut V`
                .map(move |channel_view: &mut V| {
                    // Get the mutable slice `&mut [S]` from the view using AsMut.
                    let channel_slice: &mut [S] = channel_view.as_mut();
                    // Access the sample for the current channel view at the current frame index.
                    // Safety: Relies on `frame_idx < channel_slice.len()`.
                    unsafe { channel_slice.get_unchecked_mut(frame_idx) }
                })
        })
    }

    #[nonblocking]
    fn view_mut(&mut self) -> impl AudioBlockMut<S> {
        StackedViewMut::from_slice_limited(self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [S] {
        let ch = stacked_ch.expect("For stacked layout channel needs to be provided!");
        assert!(ch < self.num_channels_allocated);
        unsafe { self.data.get_unchecked_mut(ch as usize).as_mut() }
    }
}

pub struct StackedPtrAdapterMut<'a, S: Sample, const MAX_CHANNELS: usize> {
    data: [MaybeUninit<&'a mut [S]>; MAX_CHANNELS],
    num_channels: u16,
}

impl<'a, S: Sample, const MAX_CHANNELS: usize> StackedPtrAdapterMut<'a, S, MAX_CHANNELS> {
    /// Creates new StackedPtrAdapterNew from raw pointers.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to an array of pointers
    /// - The array must contain at least `num_channels` valid pointers
    /// - Each pointer in the array must point to a valid array of samples with `num_frames` length
    /// - The pointed memory must remain valid for the lifetime of the returned adapter
    /// - The data must not be modified through other pointers for the lifetime of the returned adapter
    #[nonblocking]
    pub unsafe fn from_ptr(ptrs: *mut *mut S, num_channels: u16, num_frames: usize) -> Self {
        assert!(
            num_channels as usize <= MAX_CHANNELS,
            "num_channels exceeds MAX_CHANNELS"
        );

        let mut data: [MaybeUninit<&'a mut [S]>; MAX_CHANNELS] =
            unsafe { MaybeUninit::uninit().assume_init() }; // Or other safe initialization

        // SAFETY: Caller guarantees `ptr` is valid for `num_channels` elements.
        let ptr_slice: &mut [*mut S] =
            unsafe { core::slice::from_raw_parts_mut(ptrs, num_channels as usize) };

        for ch in 0..num_channels as usize {
            // SAFETY: See previous explanation
            data[ch].write(unsafe { core::slice::from_raw_parts_mut(ptr_slice[ch], num_frames) });
        }

        Self { data, num_channels }
    }

    #[inline]
    pub fn data_slice_mut(&mut self) -> &mut [&'a mut [S]] {
        let initialized_part: &mut [MaybeUninit<&'a mut [S]>] =
            &mut self.data[..self.num_channels as usize];
        unsafe {
            core::slice::from_raw_parts_mut(
                initialized_part.as_mut_ptr() as *mut &'a mut [S],
                self.num_channels as usize,
            )
        }
    }

    #[nonblocking]
    pub fn stacked_view_mut(&mut self) -> StackedViewMut<'a, S, &mut [S]> {
        StackedViewMut::from_slice(self.data_slice_mut())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_samples() {
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::from_slice(&mut data);

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

        assert_eq!(block.raw_data(Some(0)), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(block.raw_data(Some(1)), &[5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_channel() {
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 10];
        let mut ch2 = vec![0.0; 10];
        let mut ch3 = vec![0.0; 10];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice(), ch3.as_mut_slice()];
        let mut block = StackedViewMut::from_slice(&mut data);
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
    fn test_from_vec() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedViewMut::from_slice(&mut vec);
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
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedViewMut::from_slice(&mut vec);
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
        let mut data = vec![vec![0.0; 5]; 2];
        let mut block = StackedViewMut::from_slice(&mut data);

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
    fn test_limited() {
        let mut data = vec![vec![0.0; 4]; 3];

        let mut block = StackedViewMut::from_slice_limited(&mut data, 2, 3);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
            assert_eq!(block.channel_mut(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
            assert_eq!(block.frame_mut(i).count(), 2);
        }
    }

    #[test]
    fn test_pointer() {
        unsafe {
            let num_channels = 2;
            let num_frames = 5;
            let mut data = [vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];

            let mut ptr_vec: Vec<*mut f32> = data
                .iter_mut()
                .map(|inner_vec| inner_vec.as_mut_ptr())
                .collect();
            let ptr = ptr_vec.as_mut_ptr();

            let mut adaptor =
                StackedPtrAdapterMut::<_, 16>::from_ptr(ptr, num_channels, num_frames);

            let stacked = adaptor.stacked_view_mut();

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
    fn test_raw_data() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let mut block = StackedViewMut::from_slice(&mut vec);

        assert_eq!(block.layout(), crate::BlockLayout::Stacked);

        assert_eq!(block.raw_data(Some(0)), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data(Some(1)), &[1.0, 3.0, 5.0, 7.0, 9.0]);

        assert_eq!(block.raw_data_mut(Some(0)), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data_mut(Some(1)), &[1.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
