// #![cfg_attr(not(feature = "std"), no_std)] // enable std library when feature std is provided
#![cfg_attr(not(test), no_std)] // activate std library only for tests

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate std;

use num_traits::Float;
pub use ops::Ops;

pub mod interleaved;
pub mod ops;
pub mod planar;
pub mod stacked;

pub trait Sample: Float + 'static {}
impl Sample for f32 {}
impl Sample for f64 {}

#[derive(PartialEq, Debug)]
pub enum BlockLayout {
    Planar,
    Interleaved,
    Stacked,
}

pub trait AudioBlock<S: Sample> {
    fn num_channels(&self) -> u16;
    fn num_frames(&self) -> usize;
    fn num_channels_allocated(&self) -> u16;
    fn num_frames_allocated(&self) -> usize;
    fn sample(&self, channel: u16, frame: usize) -> S;
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_;
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
    fn view(&self) -> impl AudioBlock<S>;
    fn layout(&self) -> BlockLayout;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or planar layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data(&self, stacked_ch: Option<u16>) -> &[S];
}

pub trait AudioBlockMut<S: Sample>: AudioBlock<S> {
    fn set_num_channels(&mut self, num_channels: u16);
    fn set_num_frames(&mut self, num_frames: usize);
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S;
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
    fn channels_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_;
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
    fn view_mut(&mut self) -> impl AudioBlockMut<S>;
    /// In case of Layout::Stacked, this will return just one channel.
    /// Otherwise you will get all data in interleaved or planar layout.
    /// The returned slice includes all allocated data and not only the one
    /// that should be visible.
    fn raw_data_mut(&mut self, stacked_ch: Option<u16>) -> &mut [S];
}

// #[macro_export]
// macro_rules! channels {
//     ($block:expr, $channel:ident, $body:expr) => {
//         for c in 0..$block.num_channels() {
//             let $channel = $block.channel(c);
//             $body
//         }
//     };
// }

// #[macro_export]
// macro_rules! channels_mut {
//     ($block:expr, $channel:ident, $body:expr) => {
//         for c in 0..$block.num_channels() {
//             let $channel = $block.channel_mut(c);
//             $body
//         }
//     };
// }

// #[macro_export]
// macro_rules! frames {
//     ($block:expr, $channel:ident, $body:expr) => {
//         for c in 0..$block.num_frames() {
//             let $channel = $block.frame(c);
//             $body
//         }
//     };
// }

// #[macro_export]
// macro_rules! frames_mut {
//     ($block:expr, $channel:ident, $body:expr) => {
//         for c in 0..$block.num_frames() {
//             let $channel = $block.frame_mut(c);
//             $body
//         }
//     };
// }

// #[cfg(test)]
// mod tests {
//     use crate::{interleaved::InterleavedViewMut, planar::PlanarViewMut};

//     use super::*;

//     #[test]
//     fn test_macros() {
//         let mut data = [0.0, 1.0, 2.0, 3.0];
//         let mut block = PlanarViewMut::from_slice(&mut data, 2, 2);

//         let mut i = 0;
//         channels!(block, channel, {
//             for sample in channel {
//                 assert_eq!(*sample, i as f32);
//                 i += 1;
//             }
//         });

//         let mut i = 0;
//         channels_mut!(block, channel, {
//             for sample in channel {
//                 assert_eq!(*sample, i as f32);
//                 i += 1;
//             }
//         });

//         let mut data = [0.0, 1.0, 2.0, 3.0];
//         let mut block = InterleavedViewMut::from_slice(&mut data, 2, 2);

//         let mut i = 0;
//         frames!(block, frame, {
//             for sample in frame {
//                 assert_eq!(*sample, i as f32);
//                 i += 1;
//             }
//         });

//         let mut i = 0;
//         frames_mut!(block, frame, {
//             for sample in frame {
//                 assert_eq!(*sample, i as f32);
//                 i += 1;
//             }
//         });
//     }
// }
