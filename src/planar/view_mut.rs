use rtsan_standalone::nonblocking;

use crate::{AudioBlock, AudioBlockMut, Sample};

use super::view::PlanarView;

pub struct PlanarViewMut<'a, S: Sample> {
    data: &'a mut [S],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

#[test]
fn hell() {
    let mut data = [0.0, 1.0, 2.0, 3.0];
    let mut block = PlanarViewMut::from_slice(&mut data, 2, 2);
    for channel in block.channels_mut() {
        for sample in channel {
            dbg!(sample);
        }
    }
}

impl<'a, S: Sample> PlanarViewMut<'a, S> {
    #[nonblocking]
    pub fn from_slice(data: &'a mut [S], num_channels: u16, num_frames: usize) -> Self {
        assert_eq!(data.len(), num_channels as usize * num_frames);
        Self {
            data,
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a mut [S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_available: u16,
        num_frames_available: usize,
    ) -> Self {
        assert!(num_channels_visible <= num_channels_available);
        assert!(num_frames_visible <= num_frames_available);
        assert_eq!(
            data.len(),
            num_channels_available as usize * num_frames_available
        );
        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available,
            num_frames_allocated: num_frames_available,
        }
    }

    /// Creates a new `SequentialViewMut` from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_raw(ptr: *mut S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: unsafe {
                std::slice::from_raw_parts_mut(ptr, num_channels as usize * num_frames)
            },
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new `SequentialViewMut` from raw parts with a limited amount of channels and/or frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_raw_limited(
        ptr: *mut S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: unsafe {
                std::slice::from_raw_parts_mut(
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

impl<S: Sample> AudioBlock<S> for PlanarViewMut<'_, S> {
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
    fn view(&self) -> impl AudioBlock<S> {
        PlanarView::from_slice_limited(
            self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Planar
    }

    #[nonblocking]
    fn raw_data(&self, ch: Option<u16>) -> &[S] {
        assert!(ch.is_none());
        self.data
    }
}

impl<S: Sample> AudioBlockMut<S> for PlanarViewMut<'_, S> {
    #[nonblocking]
    fn set_num_channels(&mut self, num_channels: u16) {
        assert!(num_channels <= self.num_channels_allocated);
        self.num_channels = num_channels;
    }

    #[nonblocking]
    fn set_num_frames(&mut self, num_frames: usize) {
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
    fn view_mut(&mut self) -> impl AudioBlockMut<S> {
        PlanarViewMut::from_slice_limited(
            self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn raw_data_mut(&mut self, ch: Option<u16>) -> &mut [S] {
        assert!(ch.is_none());
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples() {
        let mut data = vec![0.0; 10];
        let mut block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);

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
    fn test_channels() {
        let mut data = vec![0.0; 10];
        let mut block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);

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
    fn test_frames() {
        let mut data = vec![0.0; 10];
        let mut block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);

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
    fn test_from_slice() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(block.frame(0).copied().collect::<Vec<_>>(), vec![0.0, 5.0]);
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![1.0, 6.0]);
        assert_eq!(block.frame(2).copied().collect::<Vec<_>>(), vec![2.0, 7.0]);
        assert_eq!(block.frame(3).copied().collect::<Vec<_>>(), vec![3.0, 8.0]);
        assert_eq!(block.frame(4).copied().collect::<Vec<_>>(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_view() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);
        let view = block.view();
        assert_eq!(
            view.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            view.channel(1).copied().collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_view_mut() {
        let mut data = vec![0.0; 10];
        let mut block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);

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
        let mut data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let mut block = PlanarViewMut::from_slice_limited(&mut data, 2, 3, 3, 4);

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
    fn test_from_raw() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = unsafe { PlanarViewMut::<f32>::from_raw(data.as_mut_ptr(), 2, 5) };
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(block.frame(0).copied().collect::<Vec<_>>(), vec![0.0, 5.0]);
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![1.0, 6.0]);
        assert_eq!(block.frame(2).copied().collect::<Vec<_>>(), vec![2.0, 7.0]);
        assert_eq!(block.frame(3).copied().collect::<Vec<_>>(), vec![3.0, 8.0]);
        assert_eq!(block.frame(4).copied().collect::<Vec<_>>(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_from_raw_limited() {
        let mut data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let mut block = unsafe { PlanarViewMut::from_raw_limited(data.as_mut_ptr(), 2, 3, 3, 4) };

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
    fn test_raw_data() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut block = PlanarViewMut::<f32>::from_slice(&mut data, 2, 5);

        assert_eq!(block.layout(), crate::BlockLayout::Planar);

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
