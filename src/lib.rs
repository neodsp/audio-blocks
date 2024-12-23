use num::Float;

pub mod interleaved;
pub mod sequential;
pub mod stacked;

pub trait Sample: Float + Default + 'static {}

impl Sample for f32 {}

pub trait BlockRead<S: Sample> {
    fn num_frames(&self) -> usize;
    fn num_channels(&self) -> u16;
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S>;
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S>;
    fn view(&self) -> impl BlockRead<S>;
}

pub trait BlockWrite<S: Sample>: BlockRead<S> {
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S>;
    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S>;
    fn view_mut(&mut self) -> impl BlockWrite<S>;

    // ops
    // fn copy_from_block(&mut self, block: &impl BlockRead<S>) {}
    // fn for_each(&mut self, f: impl FnMut(&mut S)) {}
    // fn for_channel(&mut self, f: impl FnMut(u16, &mut S)) {}
    // fn for_frame(&mut self, f: impl FnMut(u16, &mut S)) {}
}

pub trait BlockOwned<S: Sample>: BlockRead<S> + BlockWrite<S> {
    fn from_block(block: &impl BlockRead<S>) -> Self;
    fn num_channels_allocated(&self) -> u16;
    fn num_frames_allocated(&self) -> usize;
    fn set_num_channels(&mut self, num_channels: u16);
    fn set_num_frames(&mut self, num_frames: usize);
}

// pub trait Ops<S: Sample> {
//     fn for_channel<'a>(&mut self, f: impl FnMut(u16, I));
//     fn gain(&mut self, gain: S);
//     fn clear(&mut self);
// }

// impl<S: Sample, B: BlockWrite<S>> Ops<S> for B {
//     fn for_channel<'a, I: Iterator<Item = &'a mut S>>(&mut self, f: impl FnMut(u16, I)) {
//         for i in 0..self.num_channels() {
//             f(i, self.channel_mut(i));
//         }
//     }

//     fn gain(&mut self, gain: S) {
//         self.for_each(|s| *s = *s * gain);
//     }

//     fn clear(&mut self) {
//         self.for_each(|s| *s = S::zero());
//     }
// }
