use std::mem::MaybeUninit;

use rtsan::nonblocking;

use crate::{BlockRead, Sample};

pub struct StackedView<'a, S: Sample, const C: usize> {
    pub(super) data: [MaybeUninit<&'a [S]>; C],
    pub(super) num_channels: u16,
    pub(super) num_frames: usize,
    pub(super) num_channels_allocated: u16,
    pub(super) num_frames_allocated: usize,
}

impl<'a, S: Sample, const C: usize> StackedView<'a, S, C> {
    #[nonblocking]
    pub fn from_slices(data: &'a [&'a [S]]) -> Self {
        let num_frames_available = if data.is_empty() { 0 } else { data[0].len() };
        Self::from_slices_limited(data, data.len() as u16, num_frames_available)
    }

    #[nonblocking]
    pub fn from_slices_limited(
        data: &'a [&'a [S]],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_available = data.len();
        assert!(num_channels_available <= C);
        let num_frames_available = if num_channels_available == 0 {
            0
        } else {
            data[0].len()
        };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter().enumerate() {
            slice[i] = MaybeUninit::new(*item);
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available as u16,
            num_frames_allocated: num_frames_available,
        }
    }

    #[nonblocking]
    pub fn from_vec(data: &'a Vec<Vec<S>>) -> Self {
        let num_frames_available = if data.is_empty() { 0 } else { data[0].len() };
        Self::from_vec_limited(data, data.len() as u16, num_frames_available)
    }

    #[nonblocking]
    pub fn from_vec_limited(
        data: &'a Vec<Vec<S>>,
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_available = data.len();
        assert!(num_channels_available <= C);
        let num_frames_available = if num_channels_available == 0 {
            0
        } else {
            data[0].len()
        };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter().enumerate() {
            slice[i] = MaybeUninit::new(item.as_slice());
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available as u16,
            num_frames_allocated: num_frames_available,
        }
    }

    #[nonblocking]
    pub unsafe fn from_raw(data: *const *const S, num_channels: u16, num_frames: usize) -> Self {
        Self::from_raw_limited(data, num_channels, num_frames, num_channels, num_frames)
    }

    #[nonblocking]
    pub unsafe fn from_raw_limited(
        data: *const *const S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_available: u16,
        num_frames_available: usize,
    ) -> Self {
        assert!(num_channels_available as usize <= C);
        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        let data = unsafe { std::slice::from_raw_parts(data, num_channels_available as usize) };

        // Fill the slice with the provided data
        for i in 0..num_channels_available as usize {
            slice[i] = MaybeUninit::new(std::slice::from_raw_parts(data[i], num_frames_available))
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels_available as usize..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available,
            num_frames_allocated: num_frames_available,
        }
    }
}

impl<'a, S: Sample, const C: usize> BlockRead<S> for StackedView<'a, S, C> {
    #[nonblocking]
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[nonblocking]
    fn num_channels(&self) -> u16 {
        self.num_channels
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
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data[channel as usize]
                .assume_init_ref()
                .iter()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { &channel_data.assume_init_read()[frame] })
    }

    #[nonblocking]
    fn view(&self) -> impl BlockRead<S> {
        let mut out = [MaybeUninit::<&'a [S]>::uninit(); C];
        for (i, ch) in self.data.iter().enumerate() {
            out[i].write(unsafe { &*ch.assume_init_ref() });
        }
        StackedView {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_channels_allocated: self.num_channels_allocated,
            num_frames_allocated: self.num_frames_allocated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channels() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::<f32, 16>::from_slices(&data);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_frames() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::<f32, 16>::from_slices(&data);

        let channel = block.frame(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = block.frame(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![2.0, 3.0]);
        let channel = block.frame(2).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![4.0, 5.0]);
        let channel = block.frame(3).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![6.0, 7.0]);
        let channel = block.frame(4).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![8.0, 9.0]);
    }

    #[test]
    fn test_from_vec() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::<f32, 16>::from_vec(&mut vec);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(block.frame(0).copied().collect::<Vec<_>>(), vec![0.0, 1.0]);
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
        assert_eq!(block.frame(2).copied().collect::<Vec<_>>(), vec![4.0, 5.0]);
        assert_eq!(block.frame(3).copied().collect::<Vec<_>>(), vec![6.0, 7.0]);
        assert_eq!(block.frame(4).copied().collect::<Vec<_>>(), vec![8.0, 9.0]);
    }

    #[test]
    fn test_view() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::<f32, 16>::from_vec(&mut vec);
        let view = block.view();
        assert_eq!(
            view.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            view.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_limited() {
        let data = vec![vec![0.0; 4]; 3];

        let block = StackedView::<f32, 16>::from_vec_limited(&data, 2, 3);

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
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = [ch1.as_ptr(), ch2.as_ptr()];
        let block = unsafe { StackedView::<f32, 16>::from_raw(data.as_ptr(), 2, 5) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(
            block.channel(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(block.frame(0).copied().collect::<Vec<_>>(), vec![0.0, 1.0]);
        assert_eq!(block.frame(1).copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
        assert_eq!(block.frame(2).copied().collect::<Vec<_>>(), vec![4.0, 5.0]);
        assert_eq!(block.frame(3).copied().collect::<Vec<_>>(), vec![6.0, 7.0]);
        assert_eq!(block.frame(4).copied().collect::<Vec<_>>(), vec![8.0, 9.0]);
    }

    #[test]
    fn test_from_raw_limited() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let ch3 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = [ch1.as_ptr(), ch2.as_ptr(), ch3.as_ptr()];

        let block = unsafe { StackedView::<_, 16>::from_raw_limited(data.as_ptr(), 2, 3, 3, 5) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 5);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame(i).count(), 2);
        }
    }
}
