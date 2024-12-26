use rtsan::nonblocking;

use crate::{BlockRead, BlockWrite, Sample};

use super::{view::StackedView, view_mut::StackedViewMut};

#[derive(Clone)]
pub struct Stacked<S: Sample> {
    data: Vec<Vec<S>>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> Stacked<S> {
    pub fn empty(num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: vec![vec![S::default(); num_frames]; num_channels as usize],
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    pub fn from_block(block: &impl BlockRead<S>) -> Self {
        let mut data = Vec::new();
        for i in 0..block.num_channels() {
            data.push(block.channel(i).copied().collect());
        }
        Self {
            data,
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }
}

impl<S: Sample> BlockRead<S> for Stacked<S> {
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
        self.data[channel as usize].iter().take(self.num_frames)
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| &channel_data[frame])
    }

    #[nonblocking]
    fn view(&self) -> impl BlockRead<S> {
        StackedView::from_slices_limited(&self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Stacked
    }

    #[nonblocking]
    fn raw_data(&self, stacked_ch: u16) -> &[S] {
        self.data[stacked_ch as usize].as_slice()
    }
}

impl<S: Sample> BlockWrite<S> for Stacked<S> {
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
        self.data[channel as usize].iter_mut().take(self.num_frames)
    }

    #[nonblocking]
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| &mut channel_data[frame])
    }

    #[nonblocking]
    fn view_mut(&mut self) -> impl BlockWrite<S> {
        StackedViewMut::from_slices_limited(&mut self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    fn raw_data_mut(&mut self, stacked_ch: u16) -> &mut [S] {
        self.data[stacked_ch as usize].as_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::interleaved::InterleavedView;

    use super::*;

    #[test]
    fn test_channels() {
        let mut block = Stacked::<f32>::empty(2, 5);

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
        let mut block = Stacked::<f32>::empty(2, 5);

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
    fn test_from_block() {
        let block = Stacked::<f32>::from_block(&InterleavedView::from_slice(
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
        let block = Stacked::<f32>::from_block(&InterleavedView::from_slice(
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
        let mut block = Stacked::<f32>::empty(2, 5);
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
    fn test_resize() {
        let mut block = Stacked::<f32>::empty(3, 10);
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

        block.set_num_channels(3);
        block.set_num_channels(2);
        block.set_num_frames(10);
        block.set_num_frames(5);

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
    #[rtsan::no_sanitize]
    fn test_wrong_resize_channels() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_channels(3);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_wrong_resize_frames() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_frames(11);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_wrong_channel() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_channels(1);
        let _ = block.channel(1);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_wrong_frame() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_frames(5);
        let _ = block.frame(5);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_wrong_channel_mut() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_channels(1);
        let _ = block.channel_mut(1);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_wrong_frame_mut() {
        let mut block = Stacked::<f32>::empty(2, 10);
        block.set_num_frames(5);
        let _ = block.frame_mut(5);
    }

    #[test]
    fn test_raw_data() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let mut block = Stacked::from_block(&mut StackedViewMut::from_slices(&mut vec));

        assert_eq!(block.layout(), crate::BlockLayout::Stacked);

        assert_eq!(block.raw_data(0), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data(1), &[1.0, 3.0, 5.0, 7.0, 9.0]);

        assert_eq!(block.raw_data_mut(0), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data_mut(1), &[1.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
