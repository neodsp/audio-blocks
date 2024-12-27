use rtsan::nonblocking;

use crate::{BlockRead, Sample};

pub struct SequentialView<'a, S: Sample> {
    data: &'a [S],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<'a, S: Sample> SequentialView<'a, S> {
    #[nonblocking]
    pub fn from_slice(data: &'a [S], num_channels: u16, num_frames: usize) -> Self {
        assert_eq!(data.len(), num_channels as usize * num_frames);
        Self {
            data,
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a [S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            num_channels_allocated as usize * num_frames_allocated
        );
        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }

    /// Creates a new `SequentialView` from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_raw(ptr: *const S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: std::slice::from_raw_parts(ptr, num_channels as usize * num_frames),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new `SequentialView` from raw parts with a limited amount of channels and/or frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid memory containing at least `num_channels_available * num_frames_available` elements
    /// - The memory referenced by `ptr` must be valid for the lifetime of the returned `SequentialView`
    /// - The memory must not be mutated through other pointers while this view exists
    #[nonblocking]
    pub unsafe fn from_raw_limited(
        ptr: *const S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: std::slice::from_raw_parts(
                ptr,
                num_channels_allocated as usize * num_frames_allocated,
            ),
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }
}

impl<'a, S: Sample> BlockRead<S> for SequentialView<'a, S> {
    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn num_channels_allocated(&self) -> u16 {
        self.num_channels_allocated
    }

    #[nonblocking]
    fn num_frames_allocated(&self) -> usize {
        self.num_frames_allocated
    }

    #[nonblocking]
    fn sample(&self, channel: u16, frame: usize) -> S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            *self
                .data
                .get_unchecked(channel as usize * self.num_frames_allocated + frame)
        }
    }

    #[nonblocking]
    fn channel(&self, channel: u16) -> impl Iterator<Item = S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize * self.num_frames_allocated)
            .take(self.num_frames)
            .copied()
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame)
            .step_by(self.num_frames_allocated)
            .take(self.num_channels as usize)
            .copied()
    }

    #[nonblocking]
    fn view(&self) -> impl BlockRead<S> {
        SequentialView::from_slice_limited(
            self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Sequential
    }

    #[nonblocking]
    fn raw_data(&self, ch: Option<u16>) -> &[S] {
        assert!(ch.is_none());
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);

        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                assert_eq!(
                    block.sample(ch, f),
                    (ch as usize * block.num_frames() + f) as f32
                );
            }
        }
    }

    #[test]
    fn test_channels() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);

        let channel = block.channel(0).collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = block.channel(1).collect::<Vec<_>>();
        assert_eq!(channel, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_frames() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);

        let channel = block.frame(0).collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 5.0]);
        let channel = block.frame(1).collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 6.0]);
        let channel = block.frame(2).collect::<Vec<_>>();
        assert_eq!(channel, vec![2.0, 7.0]);
        let channel = block.frame(3).collect::<Vec<_>>();
        assert_eq!(channel, vec![3.0, 8.0]);
        let channel = block.frame(4).collect::<Vec<_>>();
        assert_eq!(channel, vec![4.0, 9.0]);
    }

    #[test]
    fn test_from_slice() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
        assert_eq!(
            block.channel(0).collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel(1).collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(block.frame(0).collect::<Vec<_>>(), vec![0.0, 5.0]);
        assert_eq!(block.frame(1).collect::<Vec<_>>(), vec![1.0, 6.0]);
        assert_eq!(block.frame(2).collect::<Vec<_>>(), vec![2.0, 7.0]);
        assert_eq!(block.frame(3).collect::<Vec<_>>(), vec![3.0, 8.0]);
        assert_eq!(block.frame(4).collect::<Vec<_>>(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_view() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);
        let view = block.view();
        assert_eq!(
            view.channel(0).collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            view.channel(1).collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_limited() {
        let data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let block = SequentialView::from_slice_limited(&data, 2, 3, 3, 4);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
        }
    }

    #[test]
    fn test_from_raw() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = unsafe { SequentialView::<f32>::from_raw(data.as_mut_ptr(), 2, 5) };
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated, 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated, 5);
        assert_eq!(
            block.channel(0).collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel(1).collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(block.frame(0).collect::<Vec<_>>(), vec![0.0, 5.0]);
        assert_eq!(block.frame(1).collect::<Vec<_>>(), vec![1.0, 6.0]);
        assert_eq!(block.frame(2).collect::<Vec<_>>(), vec![2.0, 7.0]);
        assert_eq!(block.frame(3).collect::<Vec<_>>(), vec![3.0, 8.0]);
        assert_eq!(block.frame(4).collect::<Vec<_>>(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_from_raw_limited() {
        let data = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        let block = unsafe { SequentialView::from_raw_limited(data.as_ptr(), 2, 3, 3, 4) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
        }
    }

    #[test]
    fn test_raw_data() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let block = SequentialView::<f32>::from_slice(&data, 2, 5);

        assert_eq!(block.layout(), crate::BlockLayout::Sequential);

        assert_eq!(
            block.raw_data(None),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }
}
