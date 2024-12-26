use num::Float;

pub mod interleaved;
pub mod ops;
pub mod sequential;
pub mod stacked;

pub trait Sample: Float + Default + 'static {}

impl Sample for f32 {}

#[derive(PartialEq, Debug)]
pub enum BlockLayout {
    Sequential,
    Interleaved,
    Stacked,
}

pub trait BlockRead<S: Sample> {
    fn num_frames(&self) -> usize;
    fn num_channels(&self) -> u16;
    fn num_channels_allocated(&self) -> u16;
    fn num_frames_allocated(&self) -> usize;
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
    fn view(&self) -> impl BlockRead<S>;
    fn layout(&self) -> BlockLayout;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or sequential layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data(&self, stacked_ch: u16) -> &[S];
}

pub trait BlockWrite<S: Sample>: BlockRead<S> {
    fn set_num_channels(&mut self, num_channels: u16);
    fn set_num_frames(&mut self, num_frames: usize);
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
    fn view_mut(&mut self) -> impl BlockWrite<S>;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or sequential layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data_mut(&mut self, stacked_ch: u16) -> &mut [S];
}
