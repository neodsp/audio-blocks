use crate::{BlockRead, Sample};

#[derive(Clone)]
pub struct InterleavedView<'a, S: Sample> {
    data: &'a [S],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<'a, S: Sample> InterleavedView<'a, S> {
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

    pub fn from_slice_limited(
        data: &'a [S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_available: u16,
        num_frames_available: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            num_channels_available as usize * num_frames_available
        );
        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available,
            num_frames_allocated: num_frames_available,
        }
    }

    pub unsafe fn from_raw(ptr: *const S, num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: std::slice::from_raw_parts(ptr, num_channels as usize * num_frames),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    pub unsafe fn from_raw_limited(
        ptr: *const S,
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_available: u16,
        num_frames_available: usize,
    ) -> Self {
        Self {
            data: std::slice::from_raw_parts(
                ptr,
                num_channels_available as usize * num_frames_available,
            ),
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_available,
            num_frames_allocated: num_frames_available,
        }
    }
}

impl<'a, S: Sample> BlockRead<S> for InterleavedView<'a, S> {
    fn num_frames(&self) -> usize {
        self.num_frames
    }

    fn num_channels(&self) -> u16 {
        self.num_channels
    }

    fn channel(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize)
            .step_by(self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    fn frame(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame * self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    fn view(&self) -> impl BlockRead<S> {
        InterleavedView::from_slice_limited(
            &self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }
}
