use std::mem::MaybeUninit;

use crate::{BlockRead, Sample};

pub struct StackedView<'a, S: Sample, const C: usize> {
    pub(super) data: [MaybeUninit<&'a [S]>; C],
    pub(super) num_channels: u16,
    pub(super) num_frames: usize,
}

impl<'a, S: Sample, const C: usize> StackedView<'a, S, C> {
    pub fn from_slices(data: &'a [&'a [S]]) -> Self {
        let num_channels = data.len();
        assert!(num_channels <= C);
        let num_frames = if num_channels == 0 { 0 } else { data[0].len() };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter().enumerate() {
            slice[i] = MaybeUninit::new(*item);
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels as u16,
            num_frames,
        }
    }

    pub fn from_vec(data: &'a Vec<Vec<S>>) -> Self {
        let num_channels = data.len();
        assert!(num_channels <= C);
        let num_frames = if num_channels == 0 { 0 } else { data[0].len() };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter().enumerate() {
            slice[i] = MaybeUninit::new(item.as_slice());
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels as u16,
            num_frames,
        }
    }

    pub unsafe fn from_raw(data: *const *const S, num_channels: u16, num_frames: usize) -> Self {
        assert!(num_channels as usize <= C);
        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a [S]>; C] = unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for i in 0..num_channels as usize {
            slice[i] = MaybeUninit::new(std::slice::from_raw_parts(
                *data.add(num_channels as usize),
                num_frames,
            ))
        }

        // Fill the remaining entries with uninitialized values
        for i in num_channels as usize..C {
            slice[i] = MaybeUninit::new(&mut []);
        }

        Self {
            data: slice,
            num_channels: num_channels as u16,
            num_frames,
        }
    }
}

impl<'a, S: Sample, const C: usize> BlockRead<S> for StackedView<'a, S, C> {
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data[channel as usize]
                .assume_init_ref()
                .iter()
                .take(self.num_frames)
        }
    }

    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { &channel_data.assume_init_read()[frame] })
    }

    fn view(&self) -> impl BlockRead<S> {
        let mut out = [MaybeUninit::<&'a [S]>::uninit(); C];
        for (i, ch) in self.data.iter().enumerate() {
            out[i].write(unsafe { &*ch.assume_init_ref() });
        }
        StackedView {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
        }
    }
}
