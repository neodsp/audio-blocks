use crate::{BlockOwned, BlockRead, BlockWrite, Sample};

use super::{view::StackedView, view_mut::StackedViewMut};

#[derive(Clone)]
pub struct Stacked<S: Sample> {
    data: Vec<Vec<S>>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> BlockRead<S> for Stacked<S> {
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data[channel as usize].iter().take(self.num_frames)
    }

    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| &channel_data[frame])
    }

    fn view(&self) -> impl BlockRead<S> {
        StackedView::<S, 256>::from_vec(&self.data)
    }
}

impl<S: Sample> BlockWrite<S> for Stacked<S> {
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data[channel as usize].iter_mut().take(self.num_frames)
    }

    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| &mut channel_data[frame])
    }

    fn view_mut(&mut self) -> impl BlockWrite<S> {
        StackedViewMut::<S, 256>::from_vec(&mut self.data)
    }
}

impl<S: Sample> BlockOwned<S> for Stacked<S> {
    fn from_block(block: &impl BlockRead<S>) -> Self {
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
