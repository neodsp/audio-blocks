// #![cfg_attr(not(feature = "std"), no_std)] // enable std library when feature std is provided
#![cfg_attr(not(test), no_std)] // activate std library only for tests

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate std;

pub use ops::Ops;

pub mod interleaved;
mod iter;
pub mod ops;
pub mod sequential;
pub mod stacked;

#[derive(PartialEq, Debug)]
pub enum BlockLayout {
    Planar,
    Interleaved,
    Stacked,
}

pub trait Sample: Copy + Default + 'static {}
impl<T> Sample for T where T: Copy + Default + 'static {}

pub trait AudioBlock<T: Sample> {
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> usize;
    fn num_channels_allocated(&self) -> u16;
    fn num_frames_allocated(&self) -> usize;
    fn sample(&self, channel: u16, frame: usize) -> T;
    fn channel(&self, channel: u16) -> impl Iterator<Item = &T>;
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &T> + '_> + '_;
    fn frame(&self, frame: usize) -> impl Iterator<Item = &T>;
    fn frames(&self) -> impl Iterator<Item = impl Iterator<Item = &T> + '_> + '_;
    fn view(&self) -> impl AudioBlock<T>;
    fn layout(&self) -> BlockLayout;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or planar layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data(&self, stacked_ch: Option<u16>) -> &[T];
}

pub trait AudioBlockMut<T: Sample>: AudioBlock<T> {
    fn resize(&mut self, num_channels: u16, num_frames: usize);
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut T;
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut T>;
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut T> + '_> + '_;
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut T>;
    fn frames_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut T> + '_> + '_;
    fn view_mut(&mut self) -> impl AudioBlockMut<T>;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or planar layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [T];
}
