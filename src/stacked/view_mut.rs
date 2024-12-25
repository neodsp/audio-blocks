use std::{marker::PhantomData, mem::MaybeUninit};

use rtsan::nonblocking;

use crate::{BlockRead, BlockWrite, Sample};

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
    pub fn from_slices(data: &'a mut [V]) -> Self {
        let num_frames_available = if data.is_empty() {
            0
        } else {
            data[0].as_ref().len()
        };
        Self::from_slices_limited(data, data.len() as u16, num_frames_available)
    }

    #[nonblocking]
    pub fn from_slices_limited(
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

impl<'a, S: Sample, C: AsMut<[S]> + AsRef<[S]>> BlockRead<S> for StackedViewMut<'a, S, C> {
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
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data[channel as usize]
            .as_ref()
            .iter()
            .take(self.num_frames)
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| &channel_data.as_ref()[frame])
    }

    #[nonblocking]
    fn view(&self) -> impl BlockRead<S> {
        StackedView::from_slices_limited(&self.data, self.num_channels, self.num_frames)
    }
}

impl<'a, S: Sample, V: AsMut<[S]> + AsRef<[S]>> BlockWrite<S> for StackedViewMut<'a, S, V> {
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
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data[channel as usize]
            .as_mut()
            .iter_mut()
            .take(self.num_frames)
    }

    #[nonblocking]
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| &mut channel_data.as_mut()[frame])
    }

    #[nonblocking]
    fn view_mut(&mut self) -> impl BlockWrite<S> {
        StackedViewMut::from_slices_limited(&mut self.data, self.num_channels, self.num_frames)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;

    #[test]
    fn test_channels() {
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = StackedViewMut::from_slices(&mut data);

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
        let mut block = StackedViewMut::from_slices(&mut data);

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
        let block = StackedViewMut::from_slices(&mut vec);
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
        let block = StackedViewMut::from_slices(&mut vec);
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
        let mut block = StackedViewMut::from_slices(&mut data);

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

        let mut block = StackedViewMut::from_slices_limited(&mut data, 2, 3);

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

    unsafe fn adapt_stacked_ptr<'a, const MAX_CHANNELS: usize>(
        ptr: *const *mut f32,
        num_channels: usize,
        num_frames: usize,
    ) -> [&'a mut [f32]; MAX_CHANNELS] {
        assert!(num_channels <= MAX_CHANNELS);

        let mut data: [MaybeUninit<&mut [f32]>; MAX_CHANNELS] = MaybeUninit::uninit().assume_init();

        let ptr_slice: &mut [*mut f32] =
            std::slice::from_raw_parts_mut(ptr as *mut *mut f32, num_channels);

        for ch in 0..num_channels {
            data[ch] = MaybeUninit::new(std::slice::from_raw_parts_mut(ptr_slice[ch], num_frames));
        }

        // Fill remaining slots with dummy data to satisfy type requirements
        for ch in num_channels..MAX_CHANNELS {
            data[ch] = MaybeUninit::new(&mut []);
        }

        std::mem::transmute_copy(&data)
    }

    #[test]
    fn test_pointer() {
        unsafe {
            let num_channels = 2;
            let num_frames = 5;
            let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];

            let mut ptr_vec: Vec<*mut f32> = vec
                .iter_mut()
                .map(|inner_vec| inner_vec.as_mut_ptr())
                .collect();
            let ptr = ptr_vec.as_mut_ptr();

            let mut array = adapt_stacked_ptr::<16>(ptr, num_channels, num_frames);

            let stacked = StackedViewMut::from_slices(&mut array[..num_channels]);

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
}
