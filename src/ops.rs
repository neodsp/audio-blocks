use rtsan_standalone::nonblocking;

use crate::{AudioBlock, AudioBlockMut, BlockLayout, Sample};

pub trait Ops<S: Sample> {
    /// Copy will panic if blocks don't have the exact same size
    fn copy_from_block(&mut self, block: &impl AudioBlock<S>);
    /// Destination block will take over the size from source block.
    /// Panics if destination block has not enough memory allocated to grow.
    fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>);
    /// Gives access to all samples in the block.
    fn for_each(&mut self, f: impl FnMut(&mut S));
    /// Gives access to all samples in the block.
    /// This is faster than `for_each` by not checking bounds of the block.
    /// It can be used if your algorithm does not change if wrong samples are accessed.
    /// For example this is the case for gain, clear, etc.
    fn for_each_including_non_visible(&mut self, f: impl FnMut(&mut S));
    /// Gives access to all samples in the block while supplying the information
    /// about which channel and frame number the sample is stored in.
    fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
    /// Gives access to all samples in the block while supplying the information
    /// about which channel and frame number the sample is stored in.
    ///
    /// This is faster than `enumerate` by not checking bounds of the block.
    /// It can be used if your algorithm does not change if wrong samples are accessed.
    /// For example this is the case for gain, clear, etc.
    fn enumerate_including_non_visible(&mut self, f: impl FnMut(u16, usize, &mut S));
    /// Applies a linear gain to all samples in the buffer
    fn apply_gain(&mut self, gain: S);
    /// Sets all samples in the block to zero
    fn clear(&mut self);
}

impl<S: Sample, B: AudioBlockMut<S>> Ops<S> for B {
    #[nonblocking]
    fn copy_from_block(&mut self, block: &impl AudioBlock<S>) {
        assert_eq!(block.num_channels(), self.num_channels());
        assert_eq!(block.num_frames(), self.num_frames());
        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_mut(ch).zip(block.channel(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>) {
        assert!(block.num_channels() <= self.num_channels_allocated());
        assert!(block.num_frames() <= self.num_frames_allocated());
        self.resize(block.num_channels(), block.num_frames());

        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_mut(ch).zip(block.channel(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn for_each(&mut self, mut f: impl FnMut(&mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for channel in self.channels_mut() {
                channel.for_each(&mut f);
            }
        } else {
            match self.layout() {
                BlockLayout::Planar | BlockLayout::Stacked => {
                    for channel in self.channels_mut() {
                        channel.for_each(&mut f);
                    }
                }
                BlockLayout::Interleaved => {
                    for frame in 0..self.num_frames() {
                        self.frame_mut(frame).for_each(&mut f);
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn for_each_including_non_visible(&mut self, mut f: impl FnMut(&mut S)) {
        match self.layout() {
            BlockLayout::Planar => {
                self.raw_data_mut(None).iter_mut().for_each(&mut f);
            }
            BlockLayout::Interleaved => {
                self.raw_data_mut(None).iter_mut().for_each(&mut f);
            }
            BlockLayout::Stacked => {
                for ch in 0..self.num_channels() {
                    self.raw_data_mut(Some(ch)).iter_mut().for_each(&mut f);
                }
            }
        }
    }

    #[nonblocking]
    fn enumerate(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for ch in 0..self.num_channels() {
                self.channel_mut(ch)
                    .enumerate()
                    .for_each(|(frame, sample)| f(ch, frame, sample));
            }
        } else {
            match self.layout() {
                BlockLayout::Planar | BlockLayout::Stacked => {
                    for ch in 0..self.num_channels() {
                        self.channel_mut(ch)
                            .enumerate()
                            .for_each(|(frame, sample)| f(ch, frame, sample));
                    }
                }
                BlockLayout::Interleaved => {
                    for frame in 0..self.num_frames() {
                        self.frame_mut(frame)
                            .enumerate()
                            .for_each(|(ch, sample)| f(ch as u16, frame, sample));
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn enumerate_including_non_visible(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        match self.layout() {
            BlockLayout::Planar => {
                let num_frames = self.num_frames_allocated();
                self.raw_data_mut(None)
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i / num_frames;
                        let frame = i % num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
            BlockLayout::Interleaved => {
                let num_frames = self.num_frames_allocated();
                self.raw_data_mut(None)
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i % num_frames;
                        let frame = i / num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
            BlockLayout::Stacked => {
                for ch in 0..self.num_channels() {
                    self.raw_data_mut(Some(ch))
                        .iter_mut()
                        .enumerate()
                        .for_each(|(frame, sample)| f(ch, frame, sample));
                }
            }
        }
    }

    #[nonblocking]
    fn apply_gain(&mut self, gain: S) {
        self.for_each_including_non_visible(|v| *v = *v * gain);
    }

    #[nonblocking]
    fn clear(&mut self) {
        self.for_each_including_non_visible(|v| *v = S::zero());
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use crate::{
        interleaved::InterleavedViewMut,
        sequential::{SequentialView, SequentialViewMut},
        stacked::StackedViewMut,
    };

    use super::*;

    #[test]
    fn test_copy_from() {
        let mut data = [0.0; 15];
        let mut block = InterleavedViewMut::from_slice(&mut data, 3, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_resize(&view);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 4);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 5);

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_copy_from_exact() {
        let mut data = [0.0; 8];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2, 4);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 4);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames_allocated(), 4);

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_wrong_channels() {
        let mut data = [0.0; 5];
        let mut block = InterleavedViewMut::from_slice(&mut data, 1, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_resize(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_wrong_frames() {
        let mut data = [0.0; 9];
        let mut block = InterleavedViewMut::from_slice(&mut data, 3, 3);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_exact_wrong_channels() {
        let mut data = [0.0; 12];
        let mut block = InterleavedViewMut::from_slice(&mut data, 3, 4);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_exact_wrong_frames() {
        let mut data = [0.0; 10];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    fn test_for_each() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2, 4);

        let mut i = 0;
        let mut c_exp = 0;
        let mut f_exp = 0;
        block.enumerate_including_non_visible(|c, f, v| {
            assert_eq!(c, c_exp);
            assert_eq!(f, f_exp);
            assert_eq!(*v, i as f32);
            if f_exp == 3 {
                c_exp = (c_exp + 1) % 4;
            }
            f_exp = (f_exp + 1) % 4;
            i += 1;
        });

        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2, 4);

        let mut i = 0;
        let mut f_exp = 0;
        let mut c_exp = 0;
        block.enumerate_including_non_visible(|c, f, v| {
            assert_eq!(c, c_exp);
            assert_eq!(f, f_exp);
            assert_eq!(*v, i as f32);
            if c_exp == 3 {
                f_exp = (f_exp + 1) % 4;
            }
            c_exp = (c_exp + 1) % 4;
            i += 1;
        });

        let mut data = [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]];
        let mut block = StackedViewMut::from_slices(&mut data);

        let mut i = 0;
        let mut c_exp = 0;
        let mut f_exp = 0;
        block.enumerate_including_non_visible(|c, f, v| {
            assert_eq!(c, c_exp);
            assert_eq!(f, f_exp);
            assert_eq!(*v, i as f32);
            if f_exp == 3 {
                c_exp = (c_exp + 1) % 4;
            }
            f_exp = (f_exp + 1) % 4;
            i += 1;
        });
    }

    #[test]
    fn test_gain() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2, 4);

        block.apply_gain(2.0);

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![8.0, 10.0, 12.0, 14.0]
        );
    }

    #[test]
    fn test_clear() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2, 4);

        block.clear();

        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }
}
