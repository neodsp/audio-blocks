use rtsan_standalone::nonblocking;

use crate::{
    AudioBlock, AudioBlockMut, BlockLayout, Sample,
    mono::{MonoView, MonoViewMut},
};

pub trait AudioBlockOps<S: Sample> {
    /// Mix all channels to mono by averaging them.
    /// Only processes `min(src_frames, dst_frames)` frames.
    /// Returns `None` if all frames were processed (exact match),
    /// or `Some(frames_processed)` if a partial mix occurred.
    fn mix_to_mono(&self, dest: &mut MonoViewMut<S>) -> Option<usize>
    where
        S: std::ops::AddAssign + std::ops::Div<Output = S> + From<u16>;
    /// Mix all channels to mono by averaging them.
    /// Panics if source and destination don't have the same number of frames.
    fn mix_to_mono_exact(&self, dest: &mut MonoViewMut<S>)
    where
        S: std::ops::AddAssign + std::ops::Div<Output = S> + From<u16>;
    /// Copy a specific channel to a mono buffer.
    /// Only copies `min(src_frames, dst_frames)` frames.
    /// Returns `None` if all frames were copied (exact match),
    /// or `Some(frames_copied)` if a partial copy occurred.
    fn copy_channel_to_mono(&self, dest: &mut MonoViewMut<S>, channel: u16) -> Option<usize>;
    /// Copy a specific channel to a mono buffer.
    /// Panics if source and destination don't have the same number of frames.
    fn copy_channel_to_mono_exact(&self, dest: &mut MonoViewMut<S>, channel: u16);
}

pub trait AudioBlockOpsMut<S: Sample> {
    /// Copy samples from source block into destination.
    /// Only copies `min(src, dst)` channels and frames.
    /// Returns `None` if the entire block was copied (exact match),
    /// or `Some((channels_copied, frames_copied))` if a partial copy occurred.
    /// Never panics - safely handles mismatched block sizes.
    fn copy_from_block(&mut self, block: &impl AudioBlock<S>) -> Option<(u16, usize)>;
    /// Copy samples from source block, requiring exact size match.
    /// Panics if source and destination don't have identical channels and frames.
    fn copy_from_block_exact(&mut self, block: &impl AudioBlock<S>);
    /// Copy a mono block to all channels of this block.
    /// Only copies `min(src_frames, dst_frames)` frames.
    /// Returns `None` if all frames were copied (exact match),
    /// or `Some(frames_copied)` if a partial copy occurred.
    fn copy_mono_to_all_channels(&mut self, mono: &MonoView<S>) -> Option<usize>;
    /// Copy a mono block to all channels of this block.
    /// Panics if blocks don't have the same number of frames.
    fn copy_mono_to_all_channels_exact(&mut self, mono: &MonoView<S>);
    /// Gives access to all samples in the block.
    fn for_each(&mut self, f: impl FnMut(&mut S));
    /// Gives access to all samples in the block while supplying the information
    /// about which channel and frame number the sample is stored in.
    fn enumerate(&mut self, f: impl FnMut(u16, usize, &mut S));
    /// Iterate over all allocated samples using fast linear buffer iteration.
    ///
    /// This iterates over `num_frames_allocated()` samples, not just `num_frames()`.
    /// It uses cache-friendly linear access over the underlying storage, which is
    /// significantly faster than [`for_each`] for large buffers when the visible
    /// window is close to the allocated size.
    ///
    /// # Performance Note
    ///
    /// Only use this when `num_frames()` is close to `num_frames_allocated()`.
    /// If the buffer has been resized dramatically (e.g., `set_visible()` to half
    /// the allocation), [`for_each`] will be faster as it respects the visible window.
    fn for_each_allocated(&mut self, f: impl FnMut(&mut S));
    /// Iterate over all allocated samples with indices using fast linear buffer iteration.
    ///
    /// Like [`for_each_allocated`], this iterates over the entire allocated buffer
    /// for cache-friendly linear access. Only faster than [`enumerate`] when the
    /// visible window is close to the allocated size.
    fn enumerate_allocated(&mut self, f: impl FnMut(u16, usize, &mut S));
    /// Sets all samples in the block to the specified value.
    /// Iterates over the entire allocated buffer for efficiency.
    fn fill_with(&mut self, sample: S);
    /// Sets all samples in the block to the default value (zero for numeric types).
    /// Iterates over the entire allocated buffer for efficiency.
    fn clear(&mut self)
    where
        S: Default;
    /// Applies gain to all samples by multiplying each sample.
    /// Iterates over the entire allocated buffer for efficiency.
    fn gain(&mut self, gain: S)
    where
        S: std::ops::Mul<Output = S> + Copy;
}

impl<S: Sample, B: AudioBlock<S>> AudioBlockOps<S> for B {
    #[nonblocking]
    fn mix_to_mono(&self, dest: &mut MonoViewMut<S>) -> Option<usize>
    where
        S: std::ops::AddAssign + std::ops::Div<Output = S> + From<u16>,
    {
        let frames = self.num_frames().min(dest.num_frames());
        let num_channels = S::from(self.num_channels());

        for frame in 0..frames {
            let mut sum = *self.frame_iter(frame).next().unwrap();
            for sample in self.frame_iter(frame).skip(1) {
                sum += *sample;
            }
            *dest.sample_mut(frame) = sum / num_channels;
        }

        if frames == self.num_frames() {
            None
        } else {
            Some(frames)
        }
    }

    #[nonblocking]
    fn mix_to_mono_exact(&self, dest: &mut MonoViewMut<S>)
    where
        S: std::ops::AddAssign + std::ops::Div<Output = S> + From<u16>,
    {
        assert_eq!(self.num_frames(), dest.num_frames());

        let num_channels = S::from(self.num_channels());

        for frame in 0..self.num_frames() {
            let mut sum = *self.frame_iter(frame).next().unwrap();
            for sample in self.frame_iter(frame).skip(1) {
                sum += *sample;
            }
            *dest.sample_mut(frame) = sum / num_channels;
        }
    }

    #[nonblocking]
    fn copy_channel_to_mono(&self, dest: &mut MonoViewMut<S>, channel: u16) -> Option<usize> {
        let frames = self.num_frames().min(dest.num_frames());
        for (o, i) in dest
            .raw_data_mut()
            .iter_mut()
            .take(frames)
            .zip(self.channel_iter(channel).take(frames))
        {
            *o = *i;
        }

        if frames == self.num_frames() {
            None
        } else {
            Some(frames)
        }
    }

    #[nonblocking]
    fn copy_channel_to_mono_exact(&self, dest: &mut MonoViewMut<S>, channel: u16) {
        assert_eq!(self.num_frames(), dest.num_frames());
        for (o, i) in dest
            .raw_data_mut()
            .iter_mut()
            .zip(self.channel_iter(channel))
        {
            *o = *i;
        }
    }
}

impl<S: Sample, B: AudioBlockMut<S>> AudioBlockOpsMut<S> for B {
    #[nonblocking]
    fn copy_from_block(&mut self, block: &impl AudioBlock<S>) -> Option<(u16, usize)> {
        let channels = self.num_channels().min(block.num_channels());
        let frames = self.num_frames().min(block.num_frames());

        for (this_channel, other_channel) in self.channels_iter_mut().zip(block.channels_iter()) {
            for (sample_mut, sample) in this_channel.zip(other_channel) {
                *sample_mut = *sample;
            }
        }

        if channels == block.num_channels() && frames == block.num_frames() {
            None
        } else {
            Some((channels, frames))
        }
    }

    #[nonblocking]
    fn copy_from_block_exact(&mut self, block: &impl AudioBlock<S>) {
        assert_eq!(block.num_channels(), self.num_channels());
        assert_eq!(block.num_frames(), self.num_frames());
        for ch in 0..self.num_channels() {
            for (sample_mut, sample) in self.channel_iter_mut(ch).zip(block.channel_iter(ch)) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn copy_mono_to_all_channels(&mut self, mono: &MonoView<S>) -> Option<usize> {
        let frames = mono.num_frames().min(self.num_frames());
        for channel in self.channels_iter_mut() {
            for (sample_mut, sample) in channel.take(frames).zip(mono.samples().iter().take(frames))
            {
                *sample_mut = *sample;
            }
        }

        if frames == mono.num_frames() {
            None
        } else {
            Some(frames)
        }
    }

    #[nonblocking]
    fn copy_mono_to_all_channels_exact(&mut self, mono: &MonoView<S>) {
        assert_eq!(mono.num_frames(), self.num_frames());
        for channel in self.channels_iter_mut() {
            for (sample_mut, sample) in channel.zip(mono.samples()) {
                *sample_mut = *sample;
            }
        }
    }

    #[nonblocking]
    fn for_each(&mut self, mut f: impl FnMut(&mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for channel in self.channels_iter_mut() {
                channel.for_each(&mut f);
            }
        } else {
            match self.layout() {
                BlockLayout::Sequential | BlockLayout::Planar => {
                    for channel in self.channels_iter_mut() {
                        channel.for_each(&mut f);
                    }
                }
                BlockLayout::Interleaved => {
                    for frame in 0..self.num_frames() {
                        self.frame_iter_mut(frame).for_each(&mut f);
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn enumerate(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        // below 8 channels it is faster to always go per channel
        if self.num_channels() < 8 {
            for (ch, channel) in self.channels_iter_mut().enumerate() {
                for (fr, sample) in channel.enumerate() {
                    f(ch as u16, fr, sample)
                }
            }
        } else {
            match self.layout() {
                BlockLayout::Interleaved => {
                    for (fr, frame) in self.frames_iter_mut().enumerate() {
                        for (ch, sample) in frame.enumerate() {
                            f(ch as u16, fr, sample)
                        }
                    }
                }
                BlockLayout::Planar | BlockLayout::Sequential => {
                    for (ch, channel) in self.channels_iter_mut().enumerate() {
                        for (fr, sample) in channel.enumerate() {
                            f(ch as u16, fr, sample)
                        }
                    }
                }
            }
        }
    }

    #[nonblocking]
    fn for_each_allocated(&mut self, mut f: impl FnMut(&mut S)) {
        match self.layout() {
            BlockLayout::Interleaved => self
                .as_interleaved_view_mut()
                .expect("Layout is interleaved")
                .raw_data_mut()
                .iter_mut()
                .for_each(&mut f),
            BlockLayout::Planar => self
                .as_planar_view_mut()
                .expect("Layout is planar")
                .raw_data_mut()
                .iter_mut()
                .for_each(|c| c.as_mut().iter_mut().for_each(&mut f)),
            BlockLayout::Sequential => self
                .as_sequential_view_mut()
                .expect("Layout is sequential")
                .raw_data_mut()
                .iter_mut()
                .for_each(&mut f),
        }
    }

    #[nonblocking]
    fn enumerate_allocated(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        match self.layout() {
            BlockLayout::Interleaved => {
                let num_frames = self.num_frames_allocated();
                self.as_interleaved_view_mut()
                    .expect("Layout is interleaved")
                    .raw_data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i % num_frames;
                        let frame = i / num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
            BlockLayout::Planar => self
                .as_planar_view_mut()
                .expect("Layout is planar")
                .raw_data_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(ch, v)| {
                    v.as_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(frame, sample)| f(ch as u16, frame, sample))
                }),
            BlockLayout::Sequential => {
                let num_frames = self.num_frames_allocated();
                self.as_sequential_view_mut()
                    .expect("Layout is sequential")
                    .raw_data_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, sample)| {
                        let channel = i / num_frames;
                        let frame = i % num_frames;
                        f(channel as u16, frame, sample)
                    });
            }
        }
    }

    #[nonblocking]
    fn fill_with(&mut self, sample: S) {
        self.for_each_allocated(|v| *v = sample);
    }

    #[nonblocking]
    fn clear(&mut self)
    where
        S: Default,
    {
        self.fill_with(S::default());
    }

    #[nonblocking]
    fn gain(&mut self, gain: S)
    where
        S: std::ops::Mul<Output = S> + Copy,
    {
        self.for_each_allocated(|v| *v = *v * gain);
    }
}

#[cfg(test)]
mod tests {
    use rtsan_standalone::no_sanitize_realtime;

    use crate::{
        interleaved::InterleavedViewMut,
        sequential::{SequentialView, SequentialViewMut},
    };

    use super::*;

    #[test]
    fn test_copy_from_block_dest_larger() {
        // Destination is larger than source - source is fully copied
        let mut data = [0.0; 15];
        let mut block = InterleavedViewMut::from_slice(&mut data, 3);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);

        // Source fits entirely, so returns None (exact match from source perspective)
        let result = block.copy_from_block(&view);
        assert_eq!(result, None);

        // Buffer size should NOT change
        assert_eq!(block.num_channels(), 3);
        assert_eq!(block.num_frames(), 5);

        // First 2 channels should have the data
        // Only check first 4 frames that were copied
        assert_eq!(
            block.channel_iter(0).take(4).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel_iter(1).take(4).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_copy_from_block_exact_match_returns_none() {
        // Source and destination are the same size - should return None
        let mut data = [0.0; 8];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);

        let result = block.copy_from_block(&view);
        assert_eq!(result, None); // Exact match returns None
    }

    #[test]
    fn test_copy_from_block_clamps_to_dest_size() {
        // Source is larger than destination - partial copy
        let mut data = [0.0; 4]; // 2 channels, 2 frames
        let mut block = SequentialViewMut::from_slice(&mut data, 2);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);

        let result = block.copy_from_block(&view);
        assert_eq!(result, Some((2, 2))); // Only 2 frames copied
        // Only first 2 frames copied from each channel
        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0]
        );
    }

    #[test]
    fn test_copy_from_block_source_larger_channels() {
        // Source has more channels than destination
        let mut data = [0.0; 4]; // 1 channel, 4 frames
        let mut block = SequentialViewMut::from_slice(&mut data, 1);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);

        let result = block.copy_from_block(&view);
        assert_eq!(result, Some((1, 4))); // Only 1 channel copied

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_copy_from_block_exact() {
        let mut data = [0.0; 8];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);
        block.copy_from_block_exact(&view);

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 4);

        // First 2 channels should have the data
        assert_eq!(
            block.channel_iter(0).take(4).copied().collect::<Vec<_>>(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            block.channel_iter(1).take(4).copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0, 7.0]
        );
        // Third channel should still be zeros (only visible frames matter for iterator)
        // Note: channel_iter respects num_visible, so we check allocated directly
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_from_block_exact_wrong_channels() {
        let mut data = [0.0; 12];
        let mut block = InterleavedViewMut::from_slice(&mut data, 3);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);
        block.copy_from_block_exact(&view);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_from_block_exact_wrong_frames() {
        let mut data = [0.0; 10];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2);
        let view = SequentialView::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2);
        block.copy_from_block_exact(&view);
    }

    #[test]
    fn test_for_each() {
        // Test that for_each processes all visible samples
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2);

        let mut count = 0;
        block.for_each(|v| {
            *v *= 2.0;
            count += 1;
        });
        assert_eq!(count, 8); // All 8 samples processed
        assert_eq!(data, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);

        // Test enumerate provides valid channel/frame indices
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = InterleavedViewMut::from_slice(&mut data, 2);

        let mut seen = vec![];
        block.enumerate(|c, f, v| {
            seen.push((c, f, *v));
        });

        // Should have seen all 8 samples
        assert_eq!(seen.len(), 8);

        // Verify each sample appears exactly once with valid indices
        let mut values: Vec<f32> = seen.iter().map(|(_, _, v)| *v).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Verify channel indices are valid (0 or 1 for 2-channel block)
        for (c, _, _) in &seen {
            assert!(*c == 0 || *c == 1);
        }

        // Verify frame indices are valid (0-3 for 4-frame block)
        for (_, f, _) in &seen {
            assert!(*f < 4);
        }
    }

    #[test]
    fn test_for_each_allocated() {
        // Test that for_each_allocated iterates over the entire allocated buffer
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2);

        // Resize to half, but for_each_allocated should still process all
        block.set_visible(2, 2);

        let mut count = 0;
        block.for_each_allocated(|v| {
            *v *= 2.0;
            count += 1;
        });

        // Should have processed all 8 samples (4 frames * 2 channels), not just 4
        assert_eq!(count, 8);
        // All values should be doubled
        assert_eq!(data, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn test_enumerate_allocated() {
        // Test that enumerate_allocated provides correct indices for entire buffer
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2);

        // Resize to half
        block.set_visible(2, 2);

        let mut seen = vec![];
        block.enumerate_allocated(|c, f, v| {
            seen.push((c, f, *v));
            *v = 0.0;
        });

        // Should have seen all 8 samples with correct (channel, frame) indices
        assert_eq!(seen.len(), 8);
        assert_eq!(seen[0], (0, 0, 0.0));
        assert_eq!(seen[1], (0, 1, 1.0));
        assert_eq!(seen[2], (0, 2, 2.0));
        assert_eq!(seen[3], (0, 3, 3.0));
        assert_eq!(seen[4], (1, 0, 4.0));
        assert_eq!(seen[5], (1, 1, 5.0));
        assert_eq!(seen[6], (1, 2, 6.0));
        assert_eq!(seen[7], (1, 3, 7.0));

        // All values should be zeroed
        assert_eq!(data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clear() {
        let mut data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut block = SequentialViewMut::from_slice(&mut data, 2);

        block.fill_with(1.0);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0, 1.0]
        );

        block.clear();

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_mix_to_mono() {
        use crate::mono::MonoViewMut;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        block.mix_to_mono(&mut mono);

        assert_eq!(mono.num_frames(), 4);
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 5.0, 6.0] // (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        );
    }

    #[test]
    fn test_copy_mono_to_all_channels() {
        use crate::mono::MonoView;

        let mono_data = [1.0, 2.0, 3.0, 4.0];
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 12];
        let mut block = SequentialViewMut::from_slice(&mut data, 3);

        block.copy_mono_to_all_channels(&mono);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(2).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_mix_to_mono_flexible_dest_smaller() {
        use crate::mono::MonoViewMut;

        // Source has 4 frames, destination has 2 frames
        // Should only process 2 frames, returns Some(frames_processed)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 2];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        let result = block.mix_to_mono(&mut mono);
        assert_eq!(result, Some(2)); // Partial mix

        assert_eq!(mono.num_frames(), 2);
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0] // (1+5)/2, (2+6)/2 - only first 2 frames
        );
    }

    #[test]
    fn test_mix_to_mono_flexible_self_smaller() {
        use crate::mono::MonoViewMut;

        // Source has 2 frames, destination has 4 frames
        // Should only process 2 frames, returns None (exact match for source)
        let data = [1.0, 2.0, 3.0, 4.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [99.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        let result = block.mix_to_mono(&mut mono);
        assert_eq!(result, None); // Source was fully processed

        assert_eq!(mono.num_frames(), 4);
        // Only first 2 frames should be mixed
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![2.0, 3.0, 99.0, 99.0] // (1+3)/2, (2+4)/2, then unchanged
        );
    }

    #[test]
    fn test_mix_to_mono_exact() {
        use crate::mono::MonoViewMut;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        block.mix_to_mono_exact(&mut mono);

        assert_eq!(mono.num_frames(), 4);
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 5.0, 6.0] // (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_mix_to_mono_exact_wrong_frames() {
        use crate::mono::MonoViewMut;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 2]; // Wrong size - only 2 frames instead of 4
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        block.mix_to_mono_exact(&mut mono);
    }

    #[test]
    fn test_copy_channel_to_mono_flexible_dest_smaller() {
        use crate::mono::MonoViewMut;

        // Source has 4 frames, destination has 2 frames
        // Should only copy 2 frames, returns Some(frames_copied)
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 2];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        let result = block.copy_channel_to_mono(&mut mono, 0);
        assert_eq!(result, Some(2)); // Partial copy

        assert_eq!(mono.num_frames(), 2);
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0] // Only first 2 frames from channel 0
        );
    }

    #[test]
    fn test_copy_channel_to_mono_flexible_self_smaller() {
        use crate::mono::MonoViewMut;

        // Source has 2 frames, destination has 4 frames allocated
        // Should only copy 2 frames, returns None (source fully copied)
        let data = [1.0, 2.0, 3.0, 4.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        let result = block.copy_channel_to_mono(&mut mono, 1);
        assert_eq!(result, None); // Source fully copied

        assert_eq!(mono.num_frames(), 4);
        // Only first 2 frames should be copied from channel 1
        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_copy_channel_to_mono_exact_match_returns_none() {
        use crate::mono::MonoViewMut;

        // Source and destination have the same size
        // Should copy all frames and return None
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        let result = block.copy_channel_to_mono(&mut mono, 0);
        assert_eq!(result, None); // Exact match

        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_copy_channel_to_mono_exact() {
        use crate::mono::MonoViewMut;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 4];
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        block.copy_channel_to_mono_exact(&mut mono, 1);

        assert_eq!(
            mono.samples().iter().copied().collect::<Vec<_>>(),
            vec![5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_channel_to_mono_exact_wrong_frames() {
        use crate::mono::MonoViewMut;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let block = SequentialView::from_slice(&data, 2);

        let mut mono_data = [0.0; 2]; // Wrong size
        let mut mono = MonoViewMut::from_slice(&mut mono_data);

        block.copy_channel_to_mono_exact(&mut mono, 0);
    }

    #[test]
    fn test_copy_mono_to_all_channels_flexible_dest_smaller() {
        use crate::mono::MonoView;

        // Mono has 4 frames, destination has only 2 frames
        // Should only copy 2 frames, returns Some(frames_copied)
        let mono_data = [1.0, 2.0, 3.0, 4.0];
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 2]; // 1 channel, 2 frames
        let mut block = SequentialViewMut::from_slice(&mut data, 1);

        let result = block.copy_mono_to_all_channels(&mono);
        assert_eq!(result, Some(2)); // Partial copy

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0] // Only first 2 frames copied
        );
    }

    #[test]
    fn test_copy_mono_to_all_channels_flexible_self_smaller() {
        use crate::mono::MonoView;

        // Mono has 2 frames, destination has 4 frames
        // Should only copy 2 frames, returns None (mono fully copied)
        let mono_data = [1.0, 2.0];
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 8]; // 2 channels, 4 frames
        let mut block = SequentialViewMut::from_slice(&mut data, 2);

        let result = block.copy_mono_to_all_channels(&mono);
        assert_eq!(result, None); // Mono fully copied

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 0.0, 0.0] // Only first 2 frames copied
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 0.0, 0.0] // Only first 2 frames copied
        );
    }

    #[test]
    fn test_copy_mono_to_all_channels_exact_match_returns_none() {
        use crate::mono::MonoView;

        // Mono and destination have the same size
        // Should copy all frames and return None
        let mono_data = [1.0, 2.0, 3.0, 4.0];
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 12]; // 3 channels, 4 frames
        let mut block = SequentialViewMut::from_slice(&mut data, 3);

        let result = block.copy_mono_to_all_channels(&mono);
        assert_eq!(result, None); // Exact match

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_copy_mono_to_all_channels_exact() {
        use crate::mono::MonoView;

        let mono_data = [1.0, 2.0, 3.0, 4.0];
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 12];
        let mut block = SequentialViewMut::from_slice(&mut data, 3);

        block.copy_mono_to_all_channels_exact(&mono);

        assert_eq!(
            block.channel_iter(0).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            block.channel_iter(2).copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_copy_mono_to_all_channels_exact_wrong_frames() {
        use crate::mono::MonoView;

        let mono_data = [1.0, 2.0, 3.0, 4.0, 5.0]; // 5 frames
        let mono = MonoView::from_slice(&mono_data);

        let mut data = [0.0; 12]; // 3 channels, 4 frames - mismatch!
        let mut block = SequentialViewMut::from_slice(&mut data, 3);

        block.copy_mono_to_all_channels_exact(&mono);
    }
}
