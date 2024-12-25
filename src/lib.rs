use num::Float;

pub mod interleaved;
pub mod ops;
pub mod sequential;
pub mod stacked;

pub trait Sample: Float + Default + 'static {}

impl Sample for f32 {}

pub trait BlockRead<S: Sample> {
    fn num_frames(&self) -> usize;
    fn num_channels(&self) -> u16;
    fn num_channels_allocated(&self) -> u16;
    fn num_frames_allocated(&self) -> usize;
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
    fn view(&self) -> impl BlockRead<S>;
}

pub trait BlockWrite<S: Sample>: BlockRead<S> {
    fn set_num_channels(&mut self, num_channels: u16);
    fn set_num_frames(&mut self, num_frames: usize);
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
    fn view_mut(&mut self) -> impl BlockWrite<S>;
}
