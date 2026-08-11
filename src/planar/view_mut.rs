use rtsan_standalone::nonblocking;
use std::marker::PhantomData;

use crate::{AudioBlock, AudioBlockMut, Sample};

use super::PlanarView;

/// A mutable view of planar / separate-channel audio data.
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
/// let mut data = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
///
/// let block = PlanarViewMut::from_slice(&mut data);
///
/// assert_eq!(block.channel(0), &[0.0, 0.0, 0.0]);
/// assert_eq!(block.channel(1), &[1.0, 1.0, 1.0]);
/// ```
pub struct PlanarViewMut<'a, S: Sample, V: AsMut<[S]> + AsRef<[S]>> {
    data: &'a mut [V],
    num_channels: u16,
    num_frames: usize,
    num_channels_allocated: u16,
    num_frames_allocated: usize,
    _phantom: PhantomData<S>,
}

impl<'a, S: Sample, V: AsMut<[S]> + AsRef<[S]>> PlanarViewMut<'a, S, V> {
    /// Creates a new audio block from a mutable slice of planar audio data.
    ///
    /// # Parameters
    /// * `data` - The mutable slice containing planar audio samples (one slice per channel)
    ///
    /// # Panics
    /// Panics if the channel slices have different lengths.
    #[nonblocking]
    pub fn from_slice(data: &'a mut [V]) -> Self {
        let num_frames_allocated = if data.is_empty() {
            0
        } else {
            data[0].as_ref().len()
        };
        Self::from_slice_limited(data, data.len() as u16, num_frames_allocated)
    }

    /// Creates a new audio block from a mutable slice with limited visibility.
    ///
    /// This function allows creating a view that exposes only a subset of the allocated channels
    /// and frames, which is useful for working with a logical section of a larger buffer.
    ///
    /// # Parameters
    /// * `data` - The mutable slice containing planar audio samples (one slice per channel)
    /// * `num_channels_visible` - Number of audio channels to expose in the view
    /// * `num_frames_visible` - Number of audio frames to expose in the view
    ///
    /// # Panics
    /// * Panics if `num_channels_visible` exceeds the number of channels in `data`
    /// * Panics if `num_frames_visible` exceeds the length of any channel buffer
    /// * Panics if channel slices have different lengths
    #[nonblocking]
    pub fn from_slice_limited(
        data: &'a mut [V],
        num_channels_visible: u16,
        num_frames_visible: usize,
    ) -> Self {
        let num_channels_allocated = data.len();
        let num_frames_allocated = if num_channels_allocated == 0 {
            0
        } else {
            data[0].as_ref().len()
        };
        assert!(num_channels_visible <= num_channels_allocated as u16);
        assert!(num_frames_visible <= num_frames_allocated);
        data.iter()
            .for_each(|v| assert_eq!(v.as_ref().len(), num_frames_allocated));

        Self {
            data,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_channels_allocated: num_channels_allocated as u16,
            num_frames_allocated,
            _phantom: PhantomData,
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
    pub fn channels(&self) -> impl ExactSizeIterator<Item = &[S]> {
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(|channel_data| &channel_data.as_ref()[..self.num_frames])
    }

    /// Returns a mutable iterator over all channels in the block.
    ///
    /// Each channel is represented as a mutable slice of samples.
    #[nonblocking]
    pub fn channels_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [S]> {
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(|channel_data| &mut channel_data.as_mut()[..self.num_frames])
    }

    /// Provides direct access to the underlying memory.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data(&self) -> &[V] {
        self.data
    }

    /// Provides direct access to the underlying memory.
    ///
    /// This function gives access to all allocated data, including any reserved capacity
    /// beyond the visible range.
    #[nonblocking]
    pub fn raw_data_mut(&mut self) -> &mut [V] {
        self.data
    }

    #[nonblocking]
    pub fn view(&self) -> PlanarView<'_, S, V> {
        PlanarView::from_slice_limited(self.data, self.num_channels, self.num_frames)
    }

    #[nonblocking]
    pub fn view_mut(&mut self) -> PlanarViewMut<'_, S, V> {
        PlanarViewMut::from_slice_limited(self.data, self.num_channels, self.num_frames)
    }
}

impl<S: Sample, V: AsMut<[S]> + AsRef<[S]>> AudioBlock<S> for PlanarViewMut<'_, S, V> {
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
                .as_ref()
                .get_unchecked(frame)
        }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
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
    fn channels_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        let num_frames = self.num_frames; // Capture num_frames for the closure
        self.data
            .iter()
            // Limit to the visible number of channels
            .take(self.num_channels as usize)
            // For each channel slice, create an iterator over its samples
            .map(move |channel_data| channel_data.as_ref().iter().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.data
            .iter()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.as_ref().get_unchecked(frame) })
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &'_ S>> {
        let num_channels = self.num_channels as usize;
        let num_frames = self.num_frames;
        // `self.data` is the field `&'data [V]`. We get `&'a [V]` from `&'a self`.
        let data_slice: &[V] = self.data;

        // Assumes the struct/caller guarantees that for all `chan` in `0..num_channels`,
        // `self.data[chan].as_ref().len() >= num_frames`.

        (0..num_frames).map(move |frame_idx| {
            // For each frame index, create an iterator over the relevant channel views.
            data_slice[..num_channels]
                .iter() // Yields `&'a V`
                .map(move |channel_view: &V| {
                    // Get the immutable slice `&[S]` from the view using AsRef.
                    let channel_slice: &[S] = channel_view.as_ref();
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
        PlanarView::from_slice_limited(self.data, self.num_channels, self.num_frames)
    }
}

impl<S: Sample, V: AsMut<[S]> + AsRef<[S]>> AudioBlockMut<S> for PlanarViewMut<'_, S, V> {
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
                .get_unchecked_mut(channel as usize)
                .as_mut()
                .get_unchecked_mut(frame)
        }
    }

    #[nonblocking]
    fn channel_iter_mut(&mut self, channel: u16) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(channel < self.num_channels);
        unsafe {
            self.data
                .get_unchecked_mut(channel as usize)
                .as_mut()
                .iter_mut()
                .take(self.num_frames)
        }
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>> {
        let num_frames = self.num_frames;
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| channel_data.as_mut().iter_mut().take(num_frames))
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.data
            .iter_mut()
            .take(self.num_channels as usize)
            .map(move |channel_data| unsafe { channel_data.as_mut().get_unchecked_mut(frame) })
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }

    #[nonblocking]
    fn for_each_allocated(&mut self, mut f: impl FnMut(&mut S)) {
        self.raw_data_mut()
            .iter_mut()
            .for_each(|c| c.as_mut().iter_mut().for_each(&mut f));
    }

    #[nonblocking]
    fn enumerate_allocated(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        self.raw_data_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(channel, channel_data)| {
                channel_data
                    .as_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(frame, sample)| f(channel as u16, frame, sample))
            });
    }
}

impl<S: Sample + core::fmt::Debug, V: AsMut<[S]> + AsRef<[S]> + core::fmt::Debug> core::fmt::Debug
    for PlanarViewMut<'_, S, V>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "audio_blocks::PlanarViewMut {{")?;
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
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_member_functions() {
        let mut data = [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let mut block = PlanarViewMut::from_slice_limited(&mut data, 2, 3);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);

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
        let mut ch1 = vec![0.0; 5];
        let mut ch2 = vec![0.0; 5];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);

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
    fn test_frame_iter_mut_covers_every_frame() {
        // Planar views mutate frame-major one frame at a time via `&mut self`.
        let mut ch0 = vec![0.0f32; 4];
        let mut ch1 = vec![0.0f32; 4];
        let mut ch2 = vec![0.0f32; 4];
        let mut data = vec![ch0.as_mut_slice(), ch1.as_mut_slice(), ch2.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);

        for f in 0..block.num_frames() {
            for (c, sample) in block.frame_iter_mut(f).enumerate() {
                *sample = (c * 10 + f) as f32;
            }
        }

        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(block.channel(1), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(block.channel(2), &[20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn test_frame_iters() {
        let mut ch1 = vec![0.0; 10];
        let mut ch2 = vec![0.0; 10];
        let mut ch3 = vec![0.0; 10];
        let mut data = vec![ch1.as_mut_slice(), ch2.as_mut_slice(), ch3.as_mut_slice()];
        let mut block = PlanarViewMut::from_slice(&mut data);
        block.set_visible(2, 5);

        let num_frames = block.num_frames;
        let mut frames_iter = block.frames_iter();
        for _ in 0..num_frames {
            let frame = frames_iter.next().unwrap().copied().collect::<Vec<_>>();
            assert_eq!(frame, vec![0.0, 0.0]);
        }
        assert!(frames_iter.next().is_none());
        drop(frames_iter);

        for i in 0..num_frames {
            let add = i as f32 * 10.0;
            block
                .frame_iter_mut(i)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + add);
        }

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
    fn test_from_vec() {
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = PlanarViewMut::from_slice(&mut vec);
        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 5);
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
        let mut vec = vec![vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let block = PlanarViewMut::from_slice(&mut vec);

        assert_eq!(block.layout(), crate::BlockLayout::Planar);
        assert_eq!(block.raw_data().len(), 2);
        assert_eq!(block.channel(0), &[0.0, 2.0, 4.0, 6.0, 8.0]);

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
        let mut data = vec![vec![0.0; 5]; 2];
        let mut block = PlanarViewMut::from_slice(&mut data);
        assert_eq!(block.layout(), crate::BlockLayout::Planar);
        assert_eq!(block.raw_data().len(), 2);
        assert_eq!(block.channel(0), &[0.0; 5]);
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
    fn test_limited() {
        let mut data = vec![vec![0.0; 4]; 3];

        let mut block = PlanarViewMut::from_slice_limited(&mut data, 2, 3);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated, 3);
        assert_eq!(block.num_frames_allocated, 4);

        for i in 0..block.num_channels() {
            assert_eq!(block.channel_iter(i).count(), 3);
            assert_eq!(block.channel_iter_mut(i).count(), 3);
        }
        for i in 0..block.num_frames() {
            assert_eq!(block.frame_iter(i).count(), 2);
            assert_eq!(block.frame_iter_mut(i).count(), 2);
        }
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds() {
        let mut data = [[0.0; 4]; 3];
        let block = PlanarViewMut::from_slice_limited(&mut data, 2, 3);

        block.channel(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_slice_out_of_bounds_mut() {
        let mut data = [[0.0; 4]; 3];
        let mut block = PlanarViewMut::from_slice_limited(&mut data, 2, 3);

        block.channel_mut(2);
    }
}
