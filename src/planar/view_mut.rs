use std::mem::MaybeUninit;

use crate::{BlockRead, BlockWrite, Sample};

use super::PlanarView;

pub struct PlanarViewMut<'a, S: Sample, const C: usize> {
    pub(super) data: [MaybeUninit<&'a mut [S]>; C],
    pub(super) num_channels: u16,
    pub(super) num_frames: usize,
}

impl<'a, S: Sample, const C: usize> PlanarViewMut<'a, S, C> {
    pub fn from_slices(data: &'a mut [&'a mut [S]]) -> Self {
        let num_channels = data.len();
        assert!(num_channels <= C);
        let num_frames = if num_channels == 0 { 0 } else { data[0].len() };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter_mut().enumerate() {
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

    pub fn from_vec(data: &'a mut Vec<Vec<S>>) -> Self {
        let num_channels = data.len();
        assert!(num_channels <= C);
        let num_frames = if num_channels == 0 { 0 } else { data[0].len() };

        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for (i, item) in data.iter_mut().enumerate() {
            slice[i] = MaybeUninit::new(item.as_mut_slice());
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

    pub unsafe fn from_raw(data: *const *mut S, num_channels: u16, num_frames: usize) -> Self {
        assert!(num_channels as usize <= C);
        // Create an uninitialized array
        let mut slice: [MaybeUninit<&'a mut [S]>; C] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Fill the slice with the provided data
        for i in 0..num_channels as usize {
            slice[i] = MaybeUninit::new(std::slice::from_raw_parts_mut(
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

impl<'a, S: Sample, const C: usize> BlockRead<S> for PlanarViewMut<'a, S, C> {
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
        PlanarView {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
        }
    }
}

impl<'a, S: Sample, const C: usize> BlockWrite<S> for PlanarViewMut<'a, S, C> {
    fn channel_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data[channel as usize]
                .assume_init_mut()
                .iter_mut()
                .take(self.num_frames)
        }
    }

    fn frame_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { &mut channel_data.assume_init_mut()[frame] })
    }

    fn view_mut(&mut self) -> impl BlockWrite<S> {
        let mut out: [MaybeUninit<&'_ mut [S]>; C] = std::array::from_fn(|_| MaybeUninit::uninit());
        for (i, ch) in self.data.iter_mut().enumerate() {
            out[i].write(unsafe { &mut *ch.assume_init_mut() });
        }
        PlanarViewMut {
            data: out,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
        }
    }
}
