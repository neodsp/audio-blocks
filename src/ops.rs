use rtsan_standalone::nonblocking;

use crate::{AudioBlock, AudioBlockMut, BlockLayout, Sample};

pub trait AudioBlockOps<S: Sample> {
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
    /// For example this is the case for applying a gain, clear, etc.
    fn enumerate_including_non_visible(&mut self, f: impl FnMut(u16, usize, &mut S));
    /// Sets all samples in the block to the specified value
    fn fill_with(&mut self, sample: S);
    /// Sets all samples in the block to zero
    fn clear(&mut self);
}

impl<S: Sample, B: AudioBlockMut<S>> AudioBlockOps<S> for B {
    #[nonblocking]
    fn copy_from_block(&mut self, block: &impl AudioBlock<S>) {
        assert_eq!(block.num_channels(), self.num_channels());
        assert_eq!(block.num_frames(), self.num_frames());
        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_iter_mut(ch).zip(block.channel_iter(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn copy_from_block_resize(&mut self, block: &impl AudioBlock<S>) {
        assert!(block.num_channels() <= self.num_channels_allocated());
        assert!(block.num_frames() <= self.num_frames_allocated());
        self.set_active_size(block.num_channels(), block.num_frames());

        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_iter_mut(ch).zip(block.channel_iter(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn for_each(&mut self, mut f: impl FnMut(&mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for channel in self.channel_iters_mut() {
                channel.for_each(&mut f);
            }
        } else {
            match self.layout() {
                BlockLayout::Sequential | BlockLayout::Planar => {
                    for channel in self.channel_iters_mut() {
                        channel.for_each(&mut f);
                    }
                }
                BlockLayout::Interleaved => {
                    for frame in 0..self.num_frames() {
                        self.frame_iter_mut(frame).for_each(&mut f);
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn for_each_including_non_visible(&mut self, mut f: impl FnMut(&mut S)) {
        match self.layout() {
            BlockLayout::Interleaved => self
                .as_interleaved_view_mut()
                .expect("Layout is interleaved")
                .raw_data_mut()
                .iter_mut()
                .for_each(&mut f),
            BlockLayout::Planar => self
                .as_planar_view_mut()
                .expect("Layout is planar")
                .raw_data_mut()
                .iter_mut()
                .for_each(|c| c.as_mut().iter_mut().for_each(&mut f)),
            BlockLayout::Sequential => self
                .as_sequential_view_mut()
                .expect("Layout is sequential")
                .raw_data_mut()
                .iter_mut()
                .for_each(&mut f),
        }
    }

    #[nonblocking]
    fn enumerate(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for (ch, channel) in self.channel_iters_mut().enumerate() {
                for (fr, sample) in channel.enumerate() {
                    f(ch as u16, fr, sample)
                }
            }
        } else {
            match self.layout() {
                BlockLayout::Interleaved => {
                    for (fr, frame) in self.frame_iters_mut().enumerate() {
                        for (ch, sample) in frame.enumerate() {
                            f(ch as u16, fr, sample)
                        }
                    }
                }
                BlockLayout::Planar | BlockLayout::Sequential => {
                    for (ch, channel) in self.channel_iters_mut().enumerate() {
                        for (fr, sample) in channel.enumerate() {
                            f(ch as u16, fr, sample)
                        }
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn enumerate_including_non_visible(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        match self.layout() {
            BlockLayout::Interleaved => {
                let num_frames = self.num_frames_allocated();
                self.as_interleaved_view_mut()
                    .expect("Layout is interleaved")
                    .raw_data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i % num_frames;
                        let frame = i / num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
            BlockLayout::Planar => self
                .as_planar_view_mut()
                .expect("Layout is planar")
                .raw_data_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(ch, v)| {
                    v.as_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(frame, sample)| f(ch as u16, frame, sample))
                }),
            BlockLayout::Sequential => {
                let num_frames = self.num_frames_allocated();
                self.as_sequential_view_mut()
                    .expect("Layout is sequential")
                    .raw_data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i / num_frames;
                        let frame = i % num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
        }
    }

    #[nonblocking]
    fn fill_with(&mut self, sample: S) {
        self.for_each_including_non_visible(|v| *v = sample);
    }

    #[nonblocking]
    fn clear(&mut self) {
        self.fill_with(S::zero());
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use crate::{
        interleaved::AudioBlockInterleavedViewMut,
        planar::AudioBlockPlanarViewMut,
        sequential::{AudioBlockSequentialView, AudioBlockSequentialViewMut},
    };

    use super::*;

    #[test]
    fn test_copy_from() {
        let mut data = [0.0; 15];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 3, 5);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_resize(&view);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 4);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 5);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_copy_from_exact() {
        let mut data = [0.0; 8];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 2, 4);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 4);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames_allocated(), 4);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_wrong_channels() {
        let mut data = [0.0; 5];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 1, 5);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_resize(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_wrong_frames() {
        let mut data = [0.0; 9];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 3, 3);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_exact_wrong_channels() {
        let mut data = [0.0; 12];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 3, 4);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_data_exact_wrong_frames() {
        let mut data = [0.0; 10];
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 2, 5);
        let view =
            AudioBlockSequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    fn test_for_each() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = AudioBlockSequentialViewMut::from_slice(&mut data, 2, 4);

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
        let mut block = AudioBlockInterleavedViewMut::from_slice(&mut data, 2, 4);

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
        let mut block = AudioBlockPlanarViewMut::from_slice(&mut data);

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
    fn test_clear() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = AudioBlockSequentialViewMut::from_slice(&mut data, 2, 4);

        block.fill_with(1.0);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0]
        );

        block.clear();

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }
}
