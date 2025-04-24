use rtsan_standalone::nonblocking;
use std::marker::PhantomData;

use crate::{AudioBlock, Sample};

pub struct StackedView<'a, S: Sample, V: AsRef<[S]>> {
    data: &'a [V],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
    _phantom: PhantomData<S>,
}

impl<'a, S: Sample, V: AsRef<[S]>> StackedView<'a, S, V> {
    #[nonblocking]
    pub fn from_slices(data: &'a [V]) -> Self {
        let num_frames_available = if data.is_empty() {
            0
        } else {
            data[0].as_ref().len()
        };
        Self::from_slices_limited(data, data.len() as u16, num_frames_available)
    }

    #[nonblocking]
    pub fn from_slices_limited(
        data: &'a [V],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_allocated = data.len() as u16;
        let num_frames_allocated = if num_channels_allocated == 0 {
            0
        } else {
            data[0].as_ref().len()
        };
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        data.iter()
            .for_each(|v| assert_eq!(v.as_ref().len(), num_frames_allocated));

        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
            _phantom: PhantomData,
        }
    }
}

impl<S: Sample, V: AsRef<[S]>> AudioBlock<S> for StackedView<'_, S, V> {
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
    fn sample(&self, channel: u16, frame: usize) -> S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            *self
                .data
                .get_unchecked(channel as usize)
                .as_ref()
                .get_unchecked(frame)
        }
    }

    #[nonblocking]
    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked(channel as usize)
                .as_ref()
                .iter()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_frames = self.num_frames; // Capture num_frames for the closure
        self.data
            .iter()
            // Limit to the active number of channels
            .take(self.num_channels as usize)
            // For each channel slice, create an iterator over its samples
            .map(move |channel_data| channel_data.as_ref().iter().take(num_frames))
    }

    #[nonblocking]
    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.as_ref().get_unchecked(frame) })
    }

    #[nonblocking]
    fn view(&self) -> impl AudioBlock<S> {
        StackedView::<S, V>::from_slices_limited(self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Stacked
    }

    #[nonblocking]
    fn raw_data(&self, stacked_ch: Option<u16>) -> &[S] {
        let ch = stacked_ch.expect("For stacked layout channel needs to be provided!");
        assert!(ch < self.num_channels_allocated);
        unsafe { self.data.get_unchecked(ch as usize).as_ref() }
    }
}

pub struct StackedPtrAdapter<'a, S: Sample, const MAX_CHANNELS: usize> {
    data: [&'a [S]; MAX_CHANNELS],
    num_channels: u16,
}

impl<'a, S: Sample, const MAX_CHANNELS: usize> StackedPtrAdapter<'a, S, MAX_CHANNELS> {
    /// Creates new StackedPtrAdapter from raw pointers.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to an array of pointers
    /// - The array must contain at least `num_channels` valid pointers
    /// - Each pointer in the array must point to a valid array of samples with `num_frames` length
    /// - The pointed memory must remain valid for the lifetime of the returned adapter
    /// - The data must not be modified through other pointers for the lifetime of the returned adapter
    #[nonblocking]
    pub unsafe fn new(ptr: *const *const S, num_channels: u16, num_frames: usize) -> Self {
        assert!(num_channels as usize <= MAX_CHANNELS);

        let mut data: [std::mem::MaybeUninit<&[S]>; MAX_CHANNELS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        let ptr_slice: &[*const S] =
            unsafe { std::slice::from_raw_parts(ptr, num_channels as usize) };

        for ch in 0..num_channels as usize {
            data[ch] = std::mem::MaybeUninit::new(unsafe {
                std::slice::from_raw_parts(ptr_slice[ch], num_frames)
            });
        }

        Self {
            data: unsafe { std::mem::transmute_copy(&data) },
            num_channels,
        }
    }

    #[nonblocking]
    pub fn stacked_view(&self) -> StackedView<'a, S, &[S]> {
        StackedView::from_slices(&self.data[..self.num_channels as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples() {
        let ch1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ch2 = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let data = [ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slices(&data);

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
    fn test_channel() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slices(&data);

        let channel = block.channel(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        let channel = block.channel(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_channels() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slices(&data);

        let mut channels_iter = block.channels();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        assert!(channels_iter.next().is_none());
    }

    #[test]
    fn test_frames() {
        let ch1 = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let ch2 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let data = vec![ch1.as_slice(), ch2.as_slice()];
        let block = StackedView::from_slices(&data);

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
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slices(&vec);
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
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slices(&vec);
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

        let block = StackedView::from_slices_limited(&data, 2, 3);

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
    fn test_pointer() {
        unsafe {
            let num_channels = 2;
            let num_frames = 5;
            let mut vec = [vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];

            let ptr_vec: Vec<*const f32> =
                vec.iter_mut().map(|inner_vec| inner_vec.as_ptr()).collect();
            let ptr = ptr_vec.as_ptr();

            let adapter = StackedPtrAdapter::<_, 16>::new(ptr, num_channels, num_frames);
            let stacked = adapter.stacked_view();

            assert_eq!(
                stacked.channel(0).copied().collect::<Vec<_>>(),
                vec![0.0, 2.0, 4.0, 6.0, 8.0]
            );

            assert_eq!(
                stacked.channel(1).copied().collect::<Vec<_>>(),
                vec![1.0, 3.0, 5.0, 7.0, 9.0]
            );
        }
    }

    #[test]
    fn test_raw_data() {
        let vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = StackedView::from_slices(&vec);

        assert_eq!(block.layout(), crate::BlockLayout::Stacked);

        assert_eq!(block.raw_data(Some(0)), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(block.raw_data(Some(1)), &[1.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
