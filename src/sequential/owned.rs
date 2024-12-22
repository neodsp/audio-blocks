use crate::{BlockOwned, BlockRead, BlockWrite, Sample};

use super::{view::SequentialView, view_mut::SequentialViewMut};

pub struct Sequential<S: Sample> {
    data: Vec<S>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> Sequential<S> {
    pub fn empty(num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: vec![S::default(); num_channels as usize * num_frames],
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }
}

impl<S: Sample> BlockRead<S> for Sequential<S> {
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_frames)
    }

    fn view(&self) -> impl BlockRead<S> {
        SequentialView::from_slice_limited(
            &self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> BlockWrite<S> for Sequential<S> {
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data
            .iter_mut()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
    }

    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_frames)
    }

    fn view_mut(&mut self) -> impl BlockWrite<S> {
        SequentialViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> BlockOwned<S> for Sequential<S> {
    fn from_block(block: &impl BlockRead<S>) -> Self {
        let mut data = Vec::new();
        for i in 0..block.num_channels() {
            block.channel(i).copied().for_each(|v| data.push(v));
        }
        Self {
            data,
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }

    fn num_channels_allocated(&self) -> u16 {
        self.num_channels_allocated
    }

    fn num_frames_allocated(&self) -> usize {
        self.num_frames_allocated
    }

    fn set_num_channels(&mut self, num_channels: u16) {
        assert!(num_channels <= self.num_channels_allocated);
        self.num_channels = num_channels;
    }

    fn set_num_frames(&mut self, num_frames: usize) {
        assert!(num_frames <= self.num_frames_allocated);
        self.num_frames = num_frames;
    }
}
