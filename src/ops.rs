use rtsan::nonblocking;

use crate::{BlockRead, BlockWrite, Sample};

pub trait Ops<S: Sample> {
    /// Destination block will take over the size from source block.
    /// Panics if destination block has not enough memory allocated to grow.
    fn copy_from_block(&mut self, block: &impl BlockRead<S>);
    /// Copy will panic if blocks don't have the exact same size
    fn copy_from_block_exact(&mut self, block: &impl BlockRead<S>);
    fn for_each(&mut self, f: impl FnMut(&mut S));
    fn gain(&mut self, gain: S);
    fn clear(&mut self);
}

impl<S: Sample, B: BlockWrite<S>> Ops<S> for B {
    #[nonblocking]
    fn copy_from_block(&mut self, block: &impl BlockRead<S>) {
        assert!(block.num_channels() <= self.num_channels_allocated());
        assert!(block.num_frames() <= self.num_frames_allocated());
        self.set_num_channels(block.num_channels());
        self.set_num_frames(block.num_frames());
        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_mut(ch).zip(block.channel(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn copy_from_block_exact(&mut self, block: &impl BlockRead<S>) {
        assert_eq!(block.num_channels(), self.num_channels());
        assert_eq!(block.num_frames(), self.num_frames());
        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_mut(ch).zip(block.channel(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn for_each(&mut self, mut f: impl FnMut(&mut S)) {
        for ch in 0..self.num_channels() {
            self.channel_mut(ch).for_each(|v| f(v));
        }
    }

    #[nonblocking]
    fn gain(&mut self, gain: S) {
        self.for_each(|s| *s = *s * gain);
    }

    #[nonblocking]
    fn clear(&mut self) {
        self.for_each(|s| *s = S::zero());
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        interleaved::Interleaved,
        sequential::{SequentialView, SequentialViewMut},
    };

    use super::*;

    #[test]
    fn test_copy_from() {
        let mut block = Interleaved::empty(3, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);

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
        let mut block = Interleaved::empty(2, 4);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_exact(&view);

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
    #[rtsan::no_sanitize]
    fn test_copy_data_wrong_channels() {
        let mut block = Interleaved::empty(1, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block(&view);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_copy_data_wrong_frames() {
        let mut block = Interleaved::empty(3, 3);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_exact(&view);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_copy_data_exact_wrong_channels() {
        let mut block = Interleaved::empty(3, 4);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_exact(&view);
    }

    #[test]
    #[should_panic]
    #[rtsan::no_sanitize]
    fn test_copy_data_exact_wrong_frames() {
        let mut block = Interleaved::empty(2, 5);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2, 4);
        block.copy_from_block_exact(&view);
    }

    #[test]
    fn test_for_each() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2, 4);

        let mut i = 0;
        block.for_each(|v| {
            assert_eq!(*v, i as f32);
            i += 1;
        });
    }

    #[test]
    fn test_gain() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2, 4);

        block.gain(2.0);

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
