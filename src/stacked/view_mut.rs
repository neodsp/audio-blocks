use std::mem::MaybeUninit;

use crate::{BlockRead, BlockWrite, Sample};

use super::StackedView;

pub struct StackedViewMut<'a, S: Sample, const C: usize> {
    pub(super) data: [MaybeUninit<&'a mut [S]>; C],
    pub(super) num_channels: u16,
    pub(super) num_frames: usize,
    pub(super) num_channels_available: u16,
    pub(super) num_frames_available: usize,
}

impl<'a, S: Sample, const C: usize> StackedViewMut<'a, S, C> {
    pub fn from_slices(data: &'a mut [&'a mut [S]]) -> Self {
        let num_frames_available = if data.is_empty() { 0 } else { data[0].len() };
        Self::from_slices_limited(data, data.len() as u16, num_frames_available)
    }

    pub fn from_slices_limited(
        data: &'a mut [&'a mut [S]],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_available = data.len();
        assert!(num_channels_available <= C);
        let num_frames_available = if num_channels_available == 0 {
            0
        } else {
            data[0].len()
        };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter_mut().enumerate() {
            slice[i] = MaybeUninit::new(*item);
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_available: num_channels_available as u16,
            num_frames_available,
        }
    }

    pub fn from_vec(data: &'a mut Vec<Vec<S>>) -> Self {
        let num_frames_available = if data.is_empty() { 0 } else { data[0].len() };
        Self::from_vec_limited(data, data.len() as u16, num_frames_available)
    }

    pub fn from_vec_limited(
        data: &'a mut Vec<Vec<S>>,
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_available = data.len();
        assert!(num_channels_available <= C);
        let num_frames_available = if num_channels_available == 0 {
            0
        } else {
            data[0].len()
        };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter_mut().enumerate() {
            slice[i] = MaybeUninit::new(item.as_mut_slice());
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_available: num_channels_available as u16,
            num_frames_available,
        }
    }

    pub unsafe fn from_raw(data: *const *mut S, num_channels: u16, num_frames: usize) -> Self {
        Self::from_raw_limited(data, num_channels, num_frames, num_channels, num_frames)
    }

    pub unsafe fn from_raw_limited(
        data: *const *mut S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_available: u16,
        num_frames_available: usize,
    ) -> Self {
        assert!(num_channels_available as usize <= C);
        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        let data = unsafe { std::slice::from_raw_parts(data, num_channels_available as usize) };

        // Fill the slice with the provided data
        for i in 0..num_channels_available as usize {
            slice[i] = MaybeUninit::new(std::slice::from_raw_parts_mut(
                data[i],
                num_frames_available,
            ))
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available as usize..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_available,
            num_frames_available,
        }
    }
}

impl<'a, S: Sample, const C: usize> BlockRead<S> for StackedViewMut<'a, S, C> {
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data[channel as usize]
                .assume_init_ref()
                .iter()
                .take(self.num_frames)
        }
    }

    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { &channel_data.assume_init_read()[frame] })
    }

    fn view(&self) -> impl BlockRead<S> {
        let mut out = [MaybeUninit::<&'a [S]>::uninit(); C];
        for (i, ch) in self.data.iter().enumerate() {
            out[i].write(unsafe { &*ch.assume_init_ref() });
        }
        StackedView {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_channels_available: self.num_channels_available,
            num_frames_available: self.num_frames_available,
        }
    }
}

impl<'a, S: Sample, const C: usize> BlockWrite<S> for StackedViewMut<'a, S, C> {
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data[channel as usize]
                .assume_init_mut()
                .iter_mut()
                .take(self.num_frames)
        }
    }

    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { &mut channel_data.assume_init_mut()[frame] })
    }

    fn view_mut(&mut self) -> impl BlockWrite<S> {
        let mut out: [MaybeUninit<&'_ mut [S]>; C] = std::array::from_fn(|_| MaybeUninit::uninit());
        for (i, ch) in self.data.iter_mut().enumerate() {
            out[i].write(unsafe { &mut *ch.assume_init_mut() });
        }
        StackedViewMut {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_channels_available: self.num_channels_available,
            num_frames_available: self.num_frames_available,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channels() {
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::<f32, 10>::from_slices(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::<f32, 10>::from_slices(&mut data);

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
    fn test_from_vec() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedViewMut::<f32, 16>::from_vec(&mut vec);
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
        let block = StackedViewMut::<f32, 16>::from_vec(&mut vec);
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
        let mut block = StackedViewMut::<f32, 16>::from_vec(&mut data);

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

        let mut block = StackedViewMut::<f32, 16>::from_vec_limited(&mut data, 2, 3);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_available, 3);
        assert_eq!(block.num_frames_available, 4);

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
        let mut ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let mut ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let mut data = [ch1.as_mut_ptr(), ch2.as_mut_ptr()];
        let block = unsafe { StackedViewMut::<f32, 16>::from_raw(data.as_mut_ptr(), 2, 5) };
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
    fn test_from_raw_limited() {
        let mut ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let mut ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let mut ch3 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let mut data = [ch1.as_mut_ptr(), ch2.as_mut_ptr(), ch3.as_mut_ptr()];

        let mut block =
            unsafe { StackedViewMut::<_, 16>::from_raw_limited(data.as_mut_ptr(), 2, 3, 3, 5) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_available, 3);
        assert_eq!(block.num_frames_available, 5);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
            assert_eq!(block.channel_mut(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
            assert_eq!(block.frame_mut(i).count(), 2);
        }
    }
}
