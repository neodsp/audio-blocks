use rtsan_standalone::{blocking, nonblocking};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{boxed::Box, vec, vec::Vec};
use core::{marker::PhantomData, ptr::NonNull};
#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::{boxed::Box, vec, vec::Vec};
#[cfg(all(feature = "std", feature = "alloc"))]
use std::{boxed::Box, vec, vec::Vec};

use super::{view::AudioBlockInterleavedView, view_mut::AudioBlockInterleavedViewMut};
use crate::{
    AudioBlock, AudioBlockMut, Sample,
    iter::{StridedSampleIter, StridedSampleIterMut},
};

/// An interleaved audio block that owns its data.
///
/// * **Layout:** `[ch0, ch1, ch0, ch1, ch0, ch1]`
/// * **Interpretation:** Each group of channel samples represents a frame. So, this layout stores frames one after another.
/// * **Terminology:** Described as “packed” or “frames first” because each time step is grouped and processed as a unit (a frame).
/// * **Usage:** Often used in APIs or hardware-level interfaces, where synchronized playback across channels is crucial.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let block = AudioBlockInterleaved::new(2, 3);
/// let mut block = AudioBlockInterleaved::from_block(&block);
///
/// block.frame_mut(0).fill(0.0);
/// block.frame_mut(1).fill(1.0);
/// block.frame_mut(2).fill(2.0);
///
/// assert_eq!(block.raw_data(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
/// ```
pub struct AudioBlockInterleaved<S: Sample> {
    data: Box<[S]>,
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
}

impl<S: Sample + Default> AudioBlockInterleaved<S> {
    /// Creates a new interleaved audio block with the specified dimensions.
    ///
    /// Allocates memory for a new interleaved audio block with exactly the specified
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
        let total_samples = (num_channels as usize)
            .checked_mul(num_frames)
            .expect("Multiplication overflow: num_channels * num_frames is too large");

        Self {
            data: vec![S::default(); total_samples].into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }
}

impl<S: Sample> AudioBlockInterleaved<S> {
    /// Creates a new interleaved audio block by copying the data from a slice of interleaved audio data.
    ///
    /// # Parameters
    /// * `data` - The slice containing interleaved audio samples
    /// * `num_channels` - Number of audio channels in the data
    /// * `num_frames` - Number of audio frames in the data
    ///
    /// # Panics
    /// Panics if the length of `data` doesn't equal `num_channels * num_frames`.
    #[blocking]
    pub fn from_slice(data: &[S], num_channels: u16, num_frames: usize) -> Self {
        assert_eq!(data.len(), num_channels as usize * num_frames);
        Self {
            data: data.to_vec().into_boxed_slice(),
            num_channels,
            num_frames,
            num_channels_allocated: num_channels,
            num_frames_allocated: num_frames,
        }
    }

    /// Creates a new interleaved audio block by copying the data from a slice of interleaved audio data with limited visibility.
    ///
    /// This function allows creating a block that exposes only a subset of the allocated channels
    /// and frames, which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The slice containing interleaved audio samples
    /// * `num_channels_visible` - Number of audio channels to expose
    /// * `num_frames_visible` - Number of audio frames to expose
    /// * `num_channels_allocated` - Total number of channels allocated in the data buffer
    /// * `num_frames_allocated` - Total number of frames allocated in the data buffer
    ///
    /// # Panics
    /// * Panics if the length of `data` doesn't equal `num_channels_allocated * num_frames_allocated`
    /// * Panics if `num_channels_visible` exceeds `num_channels_allocated`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    #[blocking]
    pub fn from_slice_limited(
        data: &[S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_channels_allocated: u16,
        num_frames_allocated: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            num_channels_allocated as usize * num_frames_allocated
        );
        assert!(num_channels_visible <= num_channels_allocated);
        assert!(num_frames_visible <= num_frames_allocated);
        Self {
            data: data.to_vec().into_boxed_slice(),
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated,
            num_frames_allocated,
        }
    }

    /// Creates a new audio block by copying data from another [`AudioBlock`].
    ///
    /// Converts any [`AudioBlock`] implementation to an interleaved format by iterating
    /// through each frame of the source block and copying its samples. The new block
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
        let mut data = Vec::with_capacity(block.num_channels() as usize * block.num_frames());
        block
            .frames_iter()
            .for_each(|f| f.for_each(|&v| data.push(v)));
        Self {
            data: data.into_boxed_slice(),
            num_channels: block.num_channels(),
            num_frames: block.num_frames(),
            num_channels_allocated: block.num_channels(),
            num_frames_allocated: block.num_frames(),
        }
    }

    /// Returns a slice for a single frame.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    #[nonblocking]
    pub fn frame(&self, frame: usize) -> &[S] {
        assert!(frame < self.num_frames);
        let start = frame * self.num_channels_allocated as usize;
        let end = start + self.num_channels as usize;
        &self.data[start..end]
    }

    /// Returns a mutable slice for a single frame.
    ///
    /// # Panics
    ///
    /// Panics if frame index is out of bounds.
    #[nonblocking]
    pub fn frame_mut(&mut self, frame: usize) -> &mut [S] {
        assert!(frame < self.num_frames);
        let start = frame * self.num_channels_allocated as usize;
        let end = start + self.num_channels as usize;
        &mut self.data[start..end]
    }

    /// Returns an iterator over all frames in the block.
    ///
    /// Each frame is represented as a slice of samples.
    #[nonblocking]
    pub fn frames(&self) -> impl Iterator<Item = &[S]> {
        self.data
            .chunks(self.num_channels_allocated as usize)
            .take(self.num_frames)
            .map(move |frame| &frame[..self.num_channels as usize])
    }

    /// Returns a mutable iterator over all frames in the block.
    ///
    /// Each frame is represented as a mutable slice of samples.
    #[nonblocking]
    pub fn frames_mut(&mut self) -> impl Iterator<Item = &mut [S]> {
        let num_channels = self.num_channels as usize;
        self.data
            .chunks_mut(self.num_channels_allocated as usize)
            .take(self.num_frames)
            .map(move |frame| &mut frame[..num_channels])
    }

    /// Provides direct access to the underlying memory as an interleaved slice.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data(&self) -> &[S] {
        &self.data
    }

    /// Provides direct mutable access to the underlying memory as an interleaved slice.
    ///
    /// This function gives mutable access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [S] {
        &mut self.data
    }

    #[nonblocking]
    pub fn view(&self) -> AudioBlockInterleavedView<'_, S> {
        AudioBlockInterleavedView::from_slice_limited(
            &self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> AudioBlockInterleavedViewMut<'_, S> {
        AudioBlockInterleavedViewMut::from_slice_limited(
            &mut self.data,
            self.num_channels,
            self.num_frames,
            self.num_channels_allocated,
            self.num_frames_allocated,
        )
    }
}

impl<S: Sample> AudioBlock<S> for AudioBlockInterleaved<S> {
    type PlanarView = [S; 0];

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
        crate::BlockLayout::Interleaved
    }

    #[nonblocking]
    fn sample(&self, channel: u16, frame: usize) -> S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            *self
                .data
                .get_unchecked(frame * self.num_channels_allocated as usize + channel as usize)
        }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl Iterator<Item = &S> {
        assert!(channel < self.num_channels);
        self.data
            .iter()
            .skip(channel as usize)
            .step_by(self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_channels_allocated as usize;

        // If the data slice is empty (num_channels or num_frames was 0),
        // the effective number of frames any iterator should yield is 0.
        let effective_num_frames = if self.data.is_empty() { 0 } else { num_frames };

        // Get base pointer. If data is empty, it's dangling, but this is fine
        // because effective_num_frames will be 0, preventing its use in next().
        let data_ptr = self.data.as_ptr();

        (0..num_channels).map(move |channel_idx| {
            // Calculate start pointer for the channel.
            // Safety: If data is not empty, data_ptr is valid and channel_idx is
            // within bounds [0, num_channels), which is <= num_channels_allocated.
            // Pointer arithmetic is contained within the allocation.
            // If data is empty, data_ptr is dangling, but add(0) is okay.
            // If channel_idx > 0 and data is empty, this relies on num_channels being 0
            // (so this closure doesn't run) or effective_num_frames being 0
            // (so the resulting iterator is a no-op).
            // We rely on effective_num_frames == 0 when data is empty.
            let start_ptr = unsafe { data_ptr.add(channel_idx) };

            StridedSampleIter::<'_, S> {
                // Safety: Cast to *mut S for NonNull::new.
                // If effective_num_frames is 0, ptr can be dangling (NonNull::dangling()).
                // If effective_num_frames > 0, data is not empty, start_ptr is valid and non-null.
                ptr: NonNull::new(start_ptr as *mut S).unwrap_or(NonNull::dangling()),
                stride,
                remaining: effective_num_frames, // Use the safe frame count
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl Iterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .skip(frame * self.num_channels_allocated as usize)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl Iterator<Item = impl Iterator<Item = &S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_channels_allocated = self.num_channels_allocated as usize;
        self.data
            .chunks(num_channels_allocated)
            .take(self.num_frames)
            .map(move |channel_chunk| channel_chunk.iter().take(num_channels))
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }

    #[nonblocking]
    fn as_interleaved_view(&self) -> Option<AudioBlockInterleavedView<'_, S>> {
        Some(self.view())
    }
}

impl<S: Sample> AudioBlockMut<S> for AudioBlockInterleaved<S> {
    type PlanarViewMut = [S; 0];

    #[nonblocking]
    fn set_num_channels_visible(&mut self, num_channels: u16) {
        assert!(num_channels <= self.num_channels_allocated);
        self.num_channels = num_channels;
    }

    #[nonblocking]
    fn set_num_frames_visible(&mut self, num_frames: usize) {
        assert!(num_frames <= self.num_frames_allocated);
        self.num_frames = num_frames;
    }

    #[nonblocking]
    fn sample_mut(&mut self, channel: u16, frame: usize) -> &mut S {
        assert!(channel < self.num_channels);
        assert!(frame < self.num_frames);
        unsafe {
            self.data
                .get_unchecked_mut(frame * self.num_channels_allocated as usize + channel as usize)
        }
    }

    #[nonblocking]
    fn channel_iter_mut(&mut self, channel: u16) -> impl Iterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        self.data
            .iter_mut()
            .skip(channel as usize)
            .step_by(self.num_channels_allocated as usize)
            .take(self.num_frames)
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        let stride = self.num_channels_allocated as usize;

        // Ensure iterator is empty if underlying data is empty.
        let effective_num_frames = if self.data.is_empty() { 0 } else { num_frames };

        // Get base mutable pointer.
        let data_ptr = self.data.as_mut_ptr();

        (0..num_channels).map(move |channel_idx| {
            // Calculate start pointer.
            // Safety: Same reasoning as the immutable version applies.
            let start_ptr = unsafe { data_ptr.add(channel_idx) };

            StridedSampleIterMut::<'_, S> {
                // Safety: Same reasoning as the immutable version applies.
                ptr: NonNull::new(start_ptr).unwrap_or(NonNull::dangling()),
                stride,
                remaining: effective_num_frames, // Use the safe frame count
                _marker: PhantomData,
            }
        })
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl Iterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .skip(frame * self.num_channels_allocated as usize)
            .take(self.num_channels as usize)
    }

    #[nonblocking]
    fn frames_iter_mut(&mut self) -> impl Iterator<Item = impl Iterator<Item = &mut S> + '_> + '_ {
        let num_channels = self.num_channels as usize;
        let num_channels_allocated = self.num_channels_allocated as usize;
        self.data
            .chunks_mut(num_channels_allocated)
            .take(self.num_frames)
            .map(move |channel_chunk| channel_chunk.iter_mut().take(num_channels))
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }

    #[nonblocking]
    fn as_interleaved_view_mut(&mut self) -> Option<AudioBlockInterleavedViewMut<'_, S>> {
        Some(self.view_mut())
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for AudioBlockInterleaved<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "AudioBlockInterleaved {{")?;
        writeln!(f, "  num_channels: {}", self.num_channels)?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(
            f,
            "  num_channels_allocated: {}",
            self.num_channels_allocated
        )?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  frames:")?;

        for (i, frame) in self.frames().enumerate() {
            writeln!(f, "    {}: {:?}", i, frame)?;
        }

        writeln!(f, "  raw_data: {:?}", self.raw_data())?;
        writeln!(f, "}}")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::AudioBlockSequentialView;
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_member_functions() {
        let mut block = AudioBlockInterleaved::<f32>::new(4, 3);
        block.frame_mut(0).copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        block.frame_mut(1).copy_from_slice(&[4.0, 5.0, 6.0, 7.0]);

        block.set_visible(3, 2);

        // single frame
        assert_eq!(block.frame(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.frame(1), &[4.0, 5.0, 6.0]);

        assert_eq!(block.frame_mut(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.frame_mut(1), &[4.0, 5.0, 6.0]);

        // all frames
        let mut frames = block.frames();
        assert_eq!(frames.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(frames.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(frames.next(), None);
        drop(frames);

        let mut frames = block.frames_mut();
        assert_eq!(frames.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(frames.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(frames.next(), None);
        drop(frames);

        // raw data
        assert_eq!(
            block.raw_data(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
        );

        assert_eq!(
            block.raw_data_mut(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0]
        );

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
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);

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

        assert_eq!(
            block.raw_data(),
            &[0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0]
        );
    }

    #[test]
    fn test_channel_iter() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);

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
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);

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
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);

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

        let frame = block.frame_iter(0).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![0.0, 1.0]);
        let frame = block.frame_iter(1).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![10.0, 11.0]);
        let frame = block.frame_iter(2).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![20.0, 21.0]);
        let frame = block.frame_iter(3).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![30.0, 31.0]);
        let frame = block.frame_iter(4).copied().collect::<Vec<_>>();
        assert_eq!(frame, vec![40.0, 41.0]);
    }

    #[test]
    fn test_frame_iters() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);
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
    fn test_view() {
        let block =
            AudioBlockInterleaved::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                2,
                5,
            ));

        assert!(block.as_interleaved_view().is_some());
        assert!(block.as_planar_view().is_none());
        assert!(block.as_sequential_view().is_none());

        let view = block.as_view();
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
        let mut block = AudioBlockInterleaved::<f32>::new(2, 5);
        assert!(block.as_interleaved_view_mut().is_some());
        assert!(block.as_planar_view_mut().is_none());
        assert!(block.as_sequential_view_mut().is_none());

        {
            let mut view = block.as_view_mut();
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
    fn test_from_slice() {
        let block =
            AudioBlockInterleaved::<f32>::from_block(&AudioBlockInterleavedView::from_slice(
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
    fn test_from_block() {
        let block = AudioBlockSequentialView::<f32>::from_slice(
            &[0.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 9.0],
            2,
            5,
        );

        let block = AudioBlockInterleaved::<f32>::from_block(&block);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn test_resize() {
        let mut block = AudioBlockInterleaved::<f32>::new(3, 10);
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

        block.set_visible(2, 5);

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
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(3, 10);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(2, 11);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(1, 10);
        let _ = block.channel_iter(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(2, 5);
        let _ = block.frame_iter(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_channel_mut() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(1, 10);
        let _ = block.channel_iter_mut(1);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_frame_mut() {
        let mut block = AudioBlockInterleaved::<f32>::new(2, 10);
        block.set_visible(2, 5);
        let _ = block.frame_iter_mut(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds() {
        let mut block = AudioBlockInterleaved::<f32>::new(3, 6);
        block.set_visible(2, 5);
        block.frame(5);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds_mut() {
        let mut block = AudioBlockInterleaved::<f32>::new(3, 6);
        block.set_visible(2, 5);
        block.frame_mut(5);
    }
}
