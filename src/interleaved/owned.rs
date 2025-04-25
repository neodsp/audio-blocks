use super::{view::InterleavedView, view_mut::InterleavedViewMut};
use crate::{
    AudioBlock, AudioBlockMut, Sample,
    iter::{InterleavedDataIter, InterleavedDataIterMut},
};
use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
use core::{marker::PhantomData, ptr::NonNull};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

#[derive(Clone)]
pub struct Interleaved<S: Sample> {
    data: Box<[S]>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> Interleaved<S> {
    #[blocking]
    pub fn new(num_channels: u16, num_frames: usize) -> Self {
        let total_samples = (num_channels as usize)
            .checked_mul(num_frames)
            .expect("Multiplication overflow: num_channels * num_frames is too large");

        Self {
            data: vec![S::default(); total_samples].into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels, // Assuming allocation matches exactly
            num_frames_allocated: num_frames,     // Assuming allocation matches exactly
        }
    }

    #[blocking]
    pub fn from_block(block: &impl AudioBlock<S>) -> Self {
        let mut data = Vec::with_capacity(block.num_channels() as usize * block.num_frames());
        for i in 0..block.num_frames() {
            block.frame(i).for_each(|&v| data.push(v));
        }
        Self {
            data: data.into_boxed_slice(),
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }
}

impl<S: Sample> AudioBlock<S> for Interleaved<S> {
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

        // If the data slice is empty (num_channels or num_frames was 0),
        // the effective number of frames any iterator should yield is 0.
        let effective_num_frames = if self.data.is_empty() { 0 } else { num_frames };

        // Get base pointer. If data is empty, it's dangling, but this is fine
        // because effective_num_frames will be 0, preventing its use in next().
        let data_ptr = self.data.as_ptr();

        (0..num_channels).map(move |channel_idx| {
            // Calculate start pointer for the channel.
            // Safety: If data is not empty, data_ptr is valid and channel_idx is
            // within bounds [0, num_channels), which is <= num_channels_allocated.
            // Pointer arithmetic is contained within the allocation.
            // If data is empty, data_ptr is dangling, but add(0) is okay.
            // If channel_idx > 0 and data is empty, this relies on num_channels being 0
            // (so this closure doesn't run) or effective_num_frames being 0
            // (so the resulting iterator is a no-op).
            // We rely on effective_num_frames == 0 when data is empty.
            let start_ptr = unsafe { data_ptr.add(channel_idx) };

            InterleavedDataIter::<'_, S> {
                // Safety: Cast to *mut S for NonNull::new.
                // If effective_num_frames is 0, ptr can be dangling (NonNull::dangling()).
                // If effective_num_frames > 0, data is not empty, start_ptr is valid and non-null.
                ptr: NonNull::new(start_ptr as *mut S).unwrap_or(NonNull::dangling()),
                stride,
                remaining: effective_num_frames, // Use the safe frame count
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
    fn view(&self) -> impl AudioBlock<S> {
        InterleavedView::from_slice_limited(
            &self.data,
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
    fn raw_data(&self, ch: Option<u16>) -> &[S] {
        assert!(ch.is_none());
        &self.data
    }
}

impl<S: Sample> AudioBlockMut<S> for Interleaved<S> {
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
                .get_unchecked_mut(frame * self.num_channels_allocated as usize + channel as usize)
        }
    }

    #[nonblocking]
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data
            .iter_mut()
            .skip(channel as usize)
            .step_by(self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_channels_allocated as usize;

        // Ensure iterator is empty if underlying data is empty.
        let effective_num_frames = if self.data.is_empty() { 0 } else { num_frames };

        // Get base mutable pointer.
        let data_ptr = self.data.as_mut_ptr();

        (0..num_channels).map(move |channel_idx| {
            // Calculate start pointer.
            // Safety: Same reasoning as the immutable version applies.
            let start_ptr = unsafe { data_ptr.add(channel_idx) };

            InterleavedDataIterMut::<'_, S> {
                // Safety: Same reasoning as the immutable version applies.
                ptr: NonNull::new(start_ptr).unwrap_or(NonNull::dangling()),
                stride,
                remaining: effective_num_frames, // Use the safe frame count
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .skip(frame * self.num_channels_allocated as usize)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_channels_allocated = self.num_channels_allocated as usize;
        self.data
            .chunks_mut(num_channels_allocated)
            .take(self.num_frames)
            .map(move |channel_chunk| channel_chunk.iter_mut().take(num_channels))
    }

    #[nonblocking]
    fn view_mut(&mut self) -> impl AudioBlockMut<S> {
        InterleavedViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn raw_data_mut(&mut self, ch: Option<u16>) -> &mut [S] {
        assert!(ch.is_none());
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use super::*;
    use crate::sequential::SequentialView;

    #[test]
    fn test_samples() {
        let mut block = Interleaved::<f32>::new(2, 5);

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
            &[0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0]
        );
    }

    #[test]
    fn test_channel() {
        let mut block = Interleaved::<f32>::new(2, 5);

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
        let mut block = Interleaved::<f32>::new(2, 5);

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
        let mut block = Interleaved::<f32>::new(2, 5);

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

        let frame = block.frame(0).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![0.0, 1.0]);
        let frame = block.frame(1).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![10.0, 11.0]);
        let frame = block.frame(2).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![20.0, 21.0]);
        let frame = block.frame(3).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![30.0, 31.0]);
        let frame = block.frame(4).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![40.0, 41.0]);
    }

    #[test]
    fn test_frames() {
        let mut block = Interleaved::<f32>::new(2, 5);
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
        let block = Interleaved::<f32>::from_block(&InterleavedView::from_slice(
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
        let block = Interleaved::<f32>::from_block(&InterleavedView::from_slice(
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
        let mut block = Interleaved::<f32>::new(2, 5);
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
    fn test_from_block() {
        let block = SequentialView::<f32>::from_slice(
            &[0.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 9.0],
            2,
            5,
        );

        let block = Interleaved::<f32>::from_block(&block);

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_resize() {
        let mut block = Interleaved::<f32>::new(3, 10);
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
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(3, 10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(2, 11);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(1, 10);
        let _ = block.channel(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame() {
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(2, 5);
        let _ = block.frame(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel_mut() {
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(1, 10);
        let _ = block.channel_mut(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame_mut() {
        let mut block = Interleaved::<f32>::new(2, 10);
        block.resize(2, 5);
        let _ = block.frame_mut(5);
    }

    #[test]
    fn test_raw_data() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut block =
            Interleaved::<f32>::from_block(&InterleavedView::<f32>::from_slice(&data, 2, 5));

        assert_eq!(block.layout(), crate::BlockLayout::Interleaved);

        assert_eq!(
            block.raw_data(None),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );

        assert_eq!(
            block.raw_data_mut(None),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }
}
