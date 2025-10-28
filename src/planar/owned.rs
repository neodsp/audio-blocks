use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

use crate::{AudioBlock, AudioBlockMut, Sample};

use super::{view::AudioBlockPlanarView, view_mut::AudioBlockPlanarViewMut};

/// A planar / seperate-channel audio block that owns its data.
///
/// * **Layout:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
/// * **Interpretation:** Each channel has its own separate buffer or array.
/// * **Terminology:** Also described as “planar” or “channels first” though more specifically it’s channel-isolated buffers.
/// * **Usage:** Very common in real-time DSP, as it simplifies memory access and can improve SIMD/vectorization efficiency.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let block = AudioBlockPlanar::new(2, 3);
/// let mut block = AudioBlockPlanar::from_block(&block);
///
/// block.channel_mut(0).fill(0.0);
/// block.channel_mut(1).fill(1.0);
///
/// assert_eq!(block.channel(0), &[0.0, 0.0, 0.0]);
/// assert_eq!(block.channel(1), &[1.0, 1.0, 1.0]);
/// ```
pub struct AudioBlockPlanar<S: Sample> {
    data: Box<[Box<[S]>]>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample> AudioBlockPlanar<S> {
    /// Creates a new audio block with the specified dimensions.
    ///
    /// Allocates memory for a new planar audio block with exactly the specified
    /// number of channels and frames. The block is initialized with the default value
    /// for the sample type.
    ///
    /// Do not use in real-time processes!
    ///
    /// # Arguments
    ///
    /// * `num_channels` - The number of audio channels
    /// * `num_frames` - The number of frames per channel
    ///
    /// # Panics
    ///
    /// Panics if the multiplication of `num_channels` and `num_frames` would overflow a usize.
    #[blocking]
    pub fn new(num_channels: u16, num_frames: usize) -> Self {
        Self {
            data: vec![vec![S::zero(); num_frames].into_boxed_slice(); num_channels as usize]
                .into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new audio block by copying data from another [`AudioBlock`].
    ///
    /// Converts any [`AudioBlock`] implementation to a planar format by iterating
    /// through each channel of the source block and copying its samples. The new block
    /// will have the same dimensions as the source block.
    ///
    /// # Warning
    ///
    /// This function allocates memory and should not be used in real-time audio processing contexts.
    ///
    /// # Arguments
    ///
    /// * `block` - The source audio block to copy data from
    #[blocking]
    pub fn from_block(block: &impl AudioBlock<S>) -> Self {
        let data: Vec<Box<[S]>> = block
            .channels_iter()
            .map(|c| c.copied().collect())
            .collect();
        Self {
            data: data.into_boxed_slice(),
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }

    /// Returns a slice for a single channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    #[nonblocking]
    pub fn channel(&self, channel: u16) -> &[S] {
        assert!(channel < self.num_channels);
        &self.data[channel as usize].as_ref()[..self.num_frames]
    }

    /// Returns a mutable slice for a single channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    #[nonblocking]
    pub fn channel_mut(&mut self, channel: u16) -> &mut [S] {
        assert!(channel < self.num_channels);
        &mut self.data[channel as usize].as_mut()[..self.num_frames]
    }

    /// Returns an iterator over all channels in the block.
    ///
    /// Each channel is represented as a slice of samples.
    #[nonblocking]
    pub fn channels(&self) -> impl Iterator<Item = &[S]> {
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(|channel_data| &channel_data.as_ref()[..self.num_frames])
    }

    /// Returns a mutable iterator over all channels in the block.
    ///
    /// Each channel is represented as a mutable slice of samples.
    #[nonblocking]
    pub fn channels_mut(&mut self) -> impl Iterator<Item = &mut [S]> {
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(|channel_data| &mut channel_data.as_mut()[..self.num_frames])
    }

    /// Provides direct access to the underlying memory.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the active range.
    #[nonblocking]
    pub fn raw_data(&self) -> &[Box<[S]>] {
        &self.data
    }

    /// Provides direct access to the underlying memory.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the active range.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [Box<[S]>] {
        &mut self.data
    }

    #[nonblocking]
    pub fn view(&self) -> AudioBlockPlanarView<'_, S, Box<[S]>> {
        AudioBlockPlanarView::from_slice_limited(&self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> AudioBlockPlanarViewMut<'_, S, Box<[S]>> {
        AudioBlockPlanarViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
        )
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockPlanar<S> {
    type PlanarView = Box<[S]>;

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
    fn layout(&self) -> crate::BlockLayout {
        crate::BlockLayout::Planar
    }

    #[nonblocking]
    fn sample(&self, channel: u16, frame: usize) -> S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            *self
                .data
                .get_unchecked(channel as usize)
                .get_unchecked(frame)
        }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked(channel as usize)
                .iter()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_frames = self.num_frames; // Capture num_frames for the closure
        self.data
            .iter()
            // Limit to the active number of channels
            .take(self.num_channels as usize)
            // For each channel slice, create an iterator over its samples
            .map(move |channel_data| channel_data.as_ref().iter().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.get_unchecked(frame) })
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        // Get an immutable slice of the channel boxes: `&[Box<[S]>]`
        let data_slice: &[Box<[S]>] = &self.data;

        // Assumes the struct guarantees that for all `chan` in `0..num_channels`,
        // `self.data[chan].len() >= num_frames`.

        (0..num_frames).map(move |frame_idx| {
            // For each frame index, create an iterator over the relevant channel boxes.
            // `data_slice` is captured immutably, which is allowed by nested closures.
            data_slice[..num_channels]
                .iter() // Yields `&'a Box<[S]>`
                .map(move |channel_slice_box| {
                    // Get the immutable slice `&[S]` from the box.
                    let channel_slice: &[S] = channel_slice_box;
                    // Access the sample immutably using safe indexing.
                    // Assumes frame_idx is valid based on outer loop and struct invariants.
                    &channel_slice[frame_idx]
                    // For max performance (if bounds are absolutely guaranteed):
                    // unsafe { channel_slice.get_unchecked(frame_idx) }
                })
        })
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }

    #[nonblocking]
    fn as_planar_view(&self) -> Option<AudioBlockPlanarView<'_, S, Self::PlanarView>> {
        Some(self.view())
    }
}

impl<S: Sample> AudioBlockMut<S> for AudioBlockPlanar<S> {
    type PlanarViewMut = Box<[S]>;

    #[nonblocking]
    fn set_active_num_channels(&mut self, num_channels: u16) {
        assert!(num_channels <= self.num_channels_allocated);
        self.num_channels = num_channels;
    }

    #[nonblocking]
    fn set_active_num_frames(&mut self, num_frames: usize) {
        assert!(num_frames <= self.num_frames_allocated);
        self.num_frames = num_frames;
    }

    #[nonblocking]
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            self.data
                .get_unchecked_mut(channel as usize)
                .get_unchecked_mut(frame)
        }
    }

    #[nonblocking]
    fn channel_iter_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked_mut(channel as usize)
                .iter_mut()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_frames = self.num_frames;
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| channel_data.as_mut().iter_mut().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.get_unchecked_mut(frame) })
    }

    #[nonblocking]
    fn frames_iter_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let data_slice: &mut [Box<[S]>] = &mut self.data;
        let data_ptr: *mut [Box<[S]>] = data_slice;

        (0..num_frames).map(move |frame_idx| {
            // Re-borrow mutably inside the closure via the raw pointer.
            // Safety: Safe because the outer iterator executes this sequentially per frame.
            let current_channel_boxes: &mut [Box<[S]>] = unsafe { &mut *data_ptr };

            // Iterate over the relevant channel boxes up to num_channels
            current_channel_boxes[..num_channels]
                .iter_mut() // Yields `&'a mut Box<[S]>`
                .map(move |channel_slice_box| {
                    // Get the mutable slice `&mut [S]` from the box.
                    let channel_slice: &mut [S] = channel_slice_box;
                    // Access the sample for the current channel at the current frame index.
                    // Safety: Relies on `frame_idx < channel_slice.len()`.
                    unsafe { channel_slice.get_unchecked_mut(frame_idx) }
                })
        })
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }

    #[nonblocking]
    fn as_planar_view_mut(
        &mut self,
    ) -> Option<AudioBlockPlanarViewMut<'_, S, Self::PlanarViewMut>> {
        Some(self.view_mut())
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for AudioBlockPlanar<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "AudioBlockPlanar {{")?;
        writeln!(f, "  num_channels: {}", self.num_channels)?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(
            f,
            "  num_channels_allocated: {}",
            self.num_channels_allocated
        )?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  channels:")?;

        for (i, channel) in self.channels().enumerate() {
            writeln!(f, "    {}: {:?}", i, channel)?;
        }

        writeln!(f, "  raw_data: {:?}", self.raw_data())?;
        writeln!(f, "}}")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interleaved::AudioBlockInterleavedView;
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_member_functions() {
        let mut block = AudioBlockPlanar::<f32>::new(3, 4);
        block.channel_mut(0).copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        block.channel_mut(1).copy_from_slice(&[4.0, 5.0, 6.0, 7.0]);

        block.set_active_size(2, 3);

        // single frame
        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[4.0, 5.0, 6.0]);

        assert_eq!(block.channel_mut(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel_mut(1), &[4.0, 5.0, 6.0]);

        // all frames
        let mut channels = block.channels();
        assert_eq!(channels.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(channels.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(channels.next(), None);
        drop(channels);

        let mut channels = block.channels_mut();
        assert_eq!(channels.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(channels.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(channels.next(), None);
        drop(channels);

        // raw data
        assert_eq!(block.raw_data()[0].as_ref(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(block.raw_data()[1].as_ref(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(block.raw_data()[2].as_ref(), &[0.0, 0.0, 0.0, 0.0]);

        assert_eq!(block.raw_data_mut()[0].as_ref(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(block.raw_data_mut()[1].as_ref(), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(block.raw_data_mut()[2].as_ref(), &[0.0, 0.0, 0.0, 0.0]);

        // views
        let view = block.view();
        assert_eq!(view.num_channels(), block.num_channels());
        assert_eq!(view.num_frames(), block.num_frames());
        assert_eq!(
            view.num_channels_allocated(),
            block.num_channels_allocated()
        );
        assert_eq!(view.num_frames_allocated(), block.num_frames_allocated());
        assert_eq!(view.raw_data(), block.raw_data());

        let num_channels = block.num_channels();
        let num_frames = block.num_frames();
        let num_channels_allocated = block.num_channels_allocated();
        let num_frames_allocated = block.num_frames_allocated();
        let data = block.raw_data().to_vec();
        let view = block.view_mut();
        assert_eq!(view.num_channels(), num_channels);
        assert_eq!(view.num_frames(), num_frames);
        assert_eq!(view.num_channels_allocated(), num_channels_allocated);
        assert_eq!(view.num_frames_allocated(), num_frames_allocated);
        assert_eq!(view.raw_data(), &data);
    }

    #[test]
    fn test_samples() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 5);

        let num_frames = block.num_frames();
        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                *block.sample_mut(ch, f) = (ch as usize * num_frames + f) as f32;
            }
        }

        for ch in 0..block.num_channels() {
            for f in 0..block.num_frames() {
                assert_eq!(block.sample(ch, f), (ch as usize * num_frames + f) as f32);
            }
        }

        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(block.channel(1), &[5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_channel_iter() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 5);

        let channel = block.channel_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = block.channel_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        block
            .channel_iter_mut(0)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        block
            .channel_iter_mut(1)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);

        let channel = block.channel_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = block.channel_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channel_iters() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 5);

        let mut channels_iter = block.channels_iter();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        let mut channels_iter = block.channels_iter_mut();
        channels_iter
            .next()
            .unwrap()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);
        channels_iter
            .next()
            .unwrap()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        let mut channels_iter = block.channels_iter();
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let channel = channels_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);
    }

    #[test]
    fn test_frame_iter() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 5);

        for i in 0..block.num_frames() {
            let frame = block.frame_iter(i).copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }

        for i in 0..block.num_frames() {
            let add = i as f32 * 10.0;
            block
                .frame_iter_mut(i)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }

        let channel = block.frame_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![0.0, 1.0]);
        let channel = block.frame_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![10.0, 11.0]);
        let channel = block.frame_iter(2).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![20.0, 21.0]);
        let channel = block.frame_iter(3).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![30.0, 31.0]);
        let channel = block.frame_iter(4).copied().collect::<Vec<_>>();
        assert_eq!(channel, vec![40.0, 41.0]);
    }

    #[test]
    fn test_frame_iters() {
        let mut block = AudioBlockPlanar::<f32>::new(3, 6);
        block.set_active_size(2, 5);

        let num_frames = block.num_frames;
        let mut frames_iter = block.frames_iter();
        for _ in 0..num_frames {
            let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        let mut frames_iter = block.frames_iter_mut();
        for i in 0..num_frames {
            let add = i as f32 * 10.0;
            frames_iter
                .next()
                .unwrap()
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        let mut frames_iter = block.frames_iter();
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![0.0, 1.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![10.0, 11.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![20.0, 21.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![30.0, 31.0]);
        let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![40.0, 41.0]);
        assert!(frames_iter.next().is_none());
    }

    #[test]
    fn test_from_block() {
        let block = AudioBlockPlanar::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            2,
            5,
        ));
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(
            block.frame_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0]
        );
        assert_eq!(
            block.frame_iter(1).copied().collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
        assert_eq!(
            block.frame_iter(2).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0]
        );
        assert_eq!(
            block.frame_iter(3).copied().collect::<Vec<_>>(),
            vec![6.0, 7.0]
        );
        assert_eq!(
            block.frame_iter(4).copied().collect::<Vec<_>>(),
            vec![8.0, 9.0]
        );
    }

    #[test]
    fn test_view() {
        let block = AudioBlockPlanar::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            2,
            5,
        ));

        assert!(block.as_interleaved_view().is_none());
        assert!(block.as_planar_view().is_some());
        assert!(block.as_sequential_view().is_none());

        let view = block.view();
        assert_eq!(
            view.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            view.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_view_mut() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 5);
        assert!(block.as_interleaved_view().is_none());
        assert!(block.as_planar_view().is_some());
        assert!(block.as_sequential_view().is_none());
        {
            let mut view = block.view_mut();
            view.channel_iter_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32);
            view.channel_iter_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + 10.0);
        }

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0, 13.0, 14.0]
        );
    }

    #[test]
    fn test_resize() {
        let mut block = AudioBlockPlanar::<f32>::new(3, 10);
        assert_eq!(block.num_channels(), 3);
        assert_eq!(block.num_frames(), 10);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel_iter(i).count(), 10);
            assert_eq!(block.channel_iter_mut(i).count(), 10);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame_iter(i).count(), 3);
            assert_eq!(block.frame_iter_mut(i).count(), 3);
        }

        block.set_active_size(3, 10);
        block.set_active_size(2, 5);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 10);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel_iter(i).count(), 5);
            assert_eq!(block.channel_iter_mut(i).count(), 5);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame_iter(i).count(), 2);
            assert_eq!(block.frame_iter_mut(i).count(), 2);
        }
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_channels() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(3, 10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(2, 11);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(1, 10);
        let _ = block.channel_iter(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(2, 5);
        let _ = block.frame_iter(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel_mut() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(1, 10);
        let _ = block.channel_iter_mut(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame_mut() {
        let mut block = AudioBlockPlanar::<f32>::new(2, 10);
        block.set_active_size(2, 5);
        let _ = block.frame_iter_mut(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds() {
        let mut block = AudioBlockPlanar::<f32>::new(3, 4);
        block.set_active_size(2, 3);

        block.channel(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds_mut() {
        let mut block = AudioBlockPlanar::<f32>::new(3, 4);
        block.set_active_size(2, 3);

        block.channel_mut(2);
    }
}
