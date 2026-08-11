use core::marker::PhantomData;
use rtsan_standalone::nonblocking;

use crate::{AudioBlock, AudioBlockMut, Sample};

/// A read-only planar audio block over a borrowed array of channel pointers.
///
/// * **Layout:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
/// * **Interpretation:** Each channel has its own separate buffer or array.
/// * **Usage:** Host APIs hand out `*const *const S`. This block borrows that
///   array instead of copying it, so there is no channel-count limit and no
///   per-channel storage.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let ch0 = [0.0f32, 1.0, 2.0];
/// let ch1 = [3.0f32, 4.0, 5.0];
/// let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
///
/// // Safety: both channels are valid for 3 frames and outlive the block.
/// let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 3) };
///
/// assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
/// assert_eq!(block.channel(1), &[3.0, 4.0, 5.0]);
/// ```
pub struct PlanarPtrs<'a, S: Sample> {
    ptrs: &'a [*const S],
    num_channels: u16,
    num_frames: usize,
    num_frames_allocated: usize,
    _marker: PhantomData<&'a S>,
}

impl<'a, S: Sample> PlanarPtrs<'a, S> {
    /// Creates a new audio block from a borrowed array of channel pointers.
    ///
    /// Every channel is visible and `num_frames` is both the visible and the
    /// allocated frame count.
    ///
    /// # Safety
    ///
    /// * Every pointer in `ptrs` must be valid and aligned for `num_frames`
    ///   elements of `S`.
    /// * That memory must stay valid for `'a`.
    /// * The samples must not be accessed through any other pointer for `'a`.
    ///
    /// Pointers may overlap, since the samples are only ever read.
    ///
    /// # Panics
    ///
    /// Panics if `ptrs.len()` exceeds `u16::MAX`.
    #[nonblocking]
    pub unsafe fn from_ptrs(ptrs: &'a [*const S], num_frames: usize) -> Self {
        assert!(ptrs.len() <= u16::MAX as usize);

        Self {
            ptrs,
            num_channels: ptrs.len() as u16,
            num_frames,
            num_frames_allocated: num_frames,
            _marker: PhantomData,
        }
    }

    /// Creates a new audio block from a borrowed array of channel pointers with
    /// limited visibility.
    ///
    /// Exposes only a subset of the allocated channels and frames, which is
    /// useful for working with a logical section of a larger buffer.
    ///
    /// # Safety
    ///
    /// * Every pointer in `ptrs` must be valid and aligned for
    ///   `num_frames_allocated` elements of `S`.
    /// * That memory must stay valid for `'a`.
    /// * The samples must not be accessed through any other pointer for `'a`.
    ///
    /// Pointers may overlap, since the samples are only ever read.
    ///
    /// # Panics
    ///
    /// * Panics if `num_channels_visible` exceeds `ptrs.len()`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    /// * Panics if `ptrs.len()` exceeds `u16::MAX`
    #[nonblocking]
    pub unsafe fn from_ptrs_limited(
        ptrs: &'a [*const S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(ptrs.len() <= u16::MAX as usize);
        assert!(num_channels_visible as usize <= ptrs.len());
        assert!(num_frames_visible <= num_frames_allocated);

        Self {
            ptrs,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_frames_allocated,
            _marker: PhantomData,
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
        // Safety: the assert keeps `channel` within the visible channels, which
        // the constructor bounded by `ptrs.len()`, and `num_frames` never
        // exceeds the `num_frames_allocated` every pointer is valid for.
        unsafe { self.channel_unchecked(channel, self.num_frames) }
    }

    /// Returns an iterator over all channels in the block.
    ///
    /// Each channel is represented as a slice of samples.
    #[nonblocking]
    pub fn channels(&self) -> impl ExactSizeIterator<Item = &[S]> {
        let num_frames = self.num_frames;
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: every visible pointer is valid for `num_frames_allocated`
            // elements and `num_frames` never exceeds that count.
            .map(move |&ptr| unsafe { core::slice::from_raw_parts(ptr, num_frames) })
    }

    /// Returns a copy of this block, borrowed for the lifetime of `self`.
    #[nonblocking]
    pub fn view(&self) -> PlanarPtrs<'_, S> {
        PlanarPtrs {
            ptrs: self.ptrs,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_frames_allocated: self.num_frames_allocated,
            _marker: PhantomData,
        }
    }

    /// Builds the slice of `channel`, covering `num_frames` samples.
    ///
    /// # Safety
    ///
    /// `channel` must be less than `self.ptrs.len()` and `num_frames` must not
    /// exceed `self.num_frames_allocated`.
    #[inline]
    unsafe fn channel_unchecked(&self, channel: u16, num_frames: usize) -> &[S] {
        // Safety: the constructor's contract makes every pointer valid and
        // aligned for `num_frames_allocated` elements, and the caller keeps
        // `channel` and `num_frames` in bounds.
        unsafe {
            core::slice::from_raw_parts(*self.ptrs.get_unchecked(channel as usize), num_frames)
        }
    }
}

impl<S: Sample> AudioBlock<S> for PlanarPtrs<'_, S> {
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
        self.ptrs.len() as u16
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
        // Safety: both asserts keep the indices inside the visible region, which
        // is contained in the region every pointer is valid for.
        unsafe { *self.ptrs.get_unchecked(channel as usize).add(frame) }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
        assert!(channel < self.num_channels);
        // Safety: see `channel`.
        unsafe { self.channel_unchecked(channel, self.num_frames).iter() }
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        self.channels().map(|channel| channel.iter())
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: `frame` is inside the visible frames, so inside the region
            // every pointer is valid for.
            .map(move |&ptr| unsafe { &*ptr.add(frame) })
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        let ptrs = &self.ptrs[..self.num_channels as usize];
        (0..self.num_frames).map(move |frame| {
            ptrs.iter()
                // Safety: `frame` is inside the visible frames, so inside the
                // region every pointer is valid for.
                .map(move |&ptr| unsafe { &*ptr.add(frame) })
        })
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for PlanarPtrs<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "audio_blocks::PlanarPtrs {{")?;
        writeln!(f, "  num_channels: {}", self.num_channels)?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(f, "  num_channels_allocated: {}", self.ptrs.len())?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  channels:")?;

        for (i, channel) in self.channels().enumerate() {
            writeln!(f, "    {}: {:?}", i, channel)?;
        }

        writeln!(f, "}}")?;

        Ok(())
    }
}

/// A mutable planar audio block over a borrowed array of channel pointers.
///
/// * **Layout:** `[[ch0, ch0, ch0], [ch1, ch1, ch1]]`
/// * **Interpretation:** Each channel has its own separate buffer or array.
/// * **Usage:** Host APIs hand out `*const *mut S`. This block borrows that
///   array instead of copying it, so there is no channel-count limit and no
///   per-channel storage.
///
/// # Example
///
/// ```
/// use audio_blocks::*;
///
/// let mut ch0 = [0.0f32; 3];
/// let mut ch1 = [0.0f32; 3];
/// let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
///
/// // Safety: the channels are distinct buffers, each valid for 3 frames, and
/// // are not touched elsewhere while the block lives.
/// let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 3) };
///
/// block.channel_mut(1).fill(1.0);
///
/// assert_eq!(block.channel(0), &[0.0, 0.0, 0.0]);
/// assert_eq!(block.channel(1), &[1.0, 1.0, 1.0]);
/// ```
pub struct PlanarPtrsMut<'a, S: Sample> {
    ptrs: &'a [*mut S],
    num_channels: u16,
    num_frames: usize,
    num_frames_allocated: usize,
    _marker: PhantomData<&'a mut S>,
}

impl<'a, S: Sample> PlanarPtrsMut<'a, S> {
    /// Creates a new audio block from a borrowed array of channel pointers.
    ///
    /// Every channel is visible and `num_frames` is both the visible and the
    /// allocated frame count.
    ///
    /// # Safety
    ///
    /// * Every pointer in `ptrs` must be valid and aligned for `num_frames`
    ///   elements of `S`.
    /// * The channel buffers must not overlap each other, since each one becomes
    ///   an exclusive `&mut [S]`.
    /// * That memory must stay valid for `'a`.
    /// * The samples must not be accessed through any other pointer for `'a`.
    ///
    /// # Panics
    ///
    /// Panics if `ptrs.len()` exceeds `u16::MAX`.
    #[nonblocking]
    pub unsafe fn from_ptrs(ptrs: &'a [*mut S], num_frames: usize) -> Self {
        assert!(ptrs.len() <= u16::MAX as usize);

        Self {
            ptrs,
            num_channels: ptrs.len() as u16,
            num_frames,
            num_frames_allocated: num_frames,
            _marker: PhantomData,
        }
    }

    /// Creates a new audio block from a borrowed array of channel pointers with
    /// limited visibility.
    ///
    /// Exposes only a subset of the allocated channels and frames, which is
    /// useful for working with a logical section of a larger buffer.
    ///
    /// # Safety
    ///
    /// * Every pointer in `ptrs` must be valid and aligned for
    ///   `num_frames_allocated` elements of `S`.
    /// * The channel buffers must not overlap each other, since each one becomes
    ///   an exclusive `&mut [S]`.
    /// * That memory must stay valid for `'a`.
    /// * The samples must not be accessed through any other pointer for `'a`.
    ///
    /// # Panics
    ///
    /// * Panics if `num_channels_visible` exceeds `ptrs.len()`
    /// * Panics if `num_frames_visible` exceeds `num_frames_allocated`
    /// * Panics if `ptrs.len()` exceeds `u16::MAX`
    #[nonblocking]
    pub unsafe fn from_ptrs_limited(
        ptrs: &'a [*mut S],
        num_channels_visible: u16,
        num_frames_visible: usize,
        num_frames_allocated: usize,
    ) -> Self {
        assert!(ptrs.len() <= u16::MAX as usize);
        assert!(num_channels_visible as usize <= ptrs.len());
        assert!(num_frames_visible <= num_frames_allocated);

        Self {
            ptrs,
            num_channels: num_channels_visible,
            num_frames: num_frames_visible,
            num_frames_allocated,
            _marker: PhantomData,
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
        // Safety: the assert keeps `channel` within the visible channels, which
        // the constructor bounded by `ptrs.len()`, and `num_frames` never
        // exceeds the `num_frames_allocated` every pointer is valid for.
        unsafe {
            core::slice::from_raw_parts(*self.ptrs.get_unchecked(channel as usize), self.num_frames)
        }
    }

    /// Returns a mutable slice for a single channel.
    ///
    /// # Panics
    ///
    /// Panics if channel index is out of bounds.
    #[nonblocking]
    pub fn channel_mut(&mut self, channel: u16) -> &mut [S] {
        assert!(channel < self.num_channels);
        // Safety: as in `channel`, plus `&mut self` guarantees no other slice of
        // this block is live, and the constructor's non-overlap contract makes
        // this channel disjoint from the others.
        unsafe {
            core::slice::from_raw_parts_mut(
                *self.ptrs.get_unchecked(channel as usize),
                self.num_frames,
            )
        }
    }

    /// Returns an iterator over all channels in the block.
    ///
    /// Each channel is represented as a slice of samples.
    #[nonblocking]
    pub fn channels(&self) -> impl ExactSizeIterator<Item = &[S]> {
        let num_frames = self.num_frames;
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: every visible pointer is valid for `num_frames_allocated`
            // elements and `num_frames` never exceeds that count.
            .map(move |&ptr| unsafe { core::slice::from_raw_parts(ptr, num_frames) })
    }

    /// Returns a mutable iterator over all channels in the block.
    ///
    /// Each channel is represented as a mutable slice of samples.
    #[nonblocking]
    pub fn channels_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [S]> {
        let num_frames = self.num_frames;
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: each slice is built from its own channel pointer, and the
            // constructor's non-overlap contract makes those regions disjoint,
            // so the yielded slices never alias. `&mut self` keeps them the only
            // live slices of this block.
            .map(move |&ptr| unsafe { core::slice::from_raw_parts_mut(ptr, num_frames) })
    }

    /// Returns a read-only copy of this block, borrowed for the lifetime of `self`.
    #[nonblocking]
    pub fn view(&self) -> PlanarPtrs<'_, S> {
        // Safety: `*mut S` and `*const S` have the same layout, so the borrowed
        // pointer array can be read as an array of const pointers. Every pointer
        // is valid for `num_frames_allocated` elements, and the shared borrow of
        // `self` keeps the block read-only while the view lives.
        let ptrs =
            unsafe { core::slice::from_raw_parts(self.ptrs.as_ptr().cast(), self.ptrs.len()) };
        PlanarPtrs {
            ptrs,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_frames_allocated: self.num_frames_allocated,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable copy of this block, borrowed for the lifetime of `self`.
    #[nonblocking]
    pub fn view_mut(&mut self) -> PlanarPtrsMut<'_, S> {
        PlanarPtrsMut {
            ptrs: self.ptrs,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            num_frames_allocated: self.num_frames_allocated,
            _marker: PhantomData,
        }
    }
}

impl<S: Sample> AudioBlock<S> for PlanarPtrsMut<'_, S> {
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
        self.ptrs.len() as u16
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
        // Safety: both asserts keep the indices inside the visible region, which
        // is contained in the region every pointer is valid for.
        unsafe { *self.ptrs.get_unchecked(channel as usize).add(frame) }
    }

    #[nonblocking]
    fn channel_iter(&self, channel: u16) -> impl ExactSizeIterator<Item = &S> {
        self.channel(channel).iter()
    }

    #[nonblocking]
    fn channels_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        self.channels().map(|channel| channel.iter())
    }

    #[nonblocking]
    fn frame_iter(&self, frame: usize) -> impl ExactSizeIterator<Item = &S> {
        assert!(frame < self.num_frames);
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: `frame` is inside the visible frames, so inside the region
            // every pointer is valid for.
            .map(move |&ptr| unsafe { &*ptr.add(frame) })
    }

    #[nonblocking]
    fn frames_iter(&self) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &S>> {
        let ptrs = &self.ptrs[..self.num_channels as usize];
        (0..self.num_frames).map(move |frame| {
            ptrs.iter()
                // Safety: `frame` is inside the visible frames, so inside the
                // region every pointer is valid for.
                .map(move |&ptr| unsafe { &*ptr.add(frame) })
        })
    }

    #[nonblocking]
    fn as_view(&self) -> impl AudioBlock<S> {
        self.view()
    }
}

impl<S: Sample> AudioBlockMut<S> for PlanarPtrsMut<'_, S> {
    #[nonblocking]
    fn set_num_channels_visible(&mut self, num_channels: u16) {
        assert!(num_channels as usize <= self.ptrs.len());
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
        // Safety: both asserts keep the indices inside the visible region, and
        // `&mut self` guarantees no other reference into this block is live.
        unsafe { &mut *self.ptrs.get_unchecked(channel as usize).add(frame) }
    }

    #[nonblocking]
    fn channel_iter_mut(&mut self, channel: u16) -> impl ExactSizeIterator<Item = &mut S> {
        self.channel_mut(channel).iter_mut()
    }

    #[nonblocking]
    fn channels_iter_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = impl ExactSizeIterator<Item = &mut S>> {
        self.channels_mut().map(|channel| channel.iter_mut())
    }

    #[nonblocking]
    fn frame_iter_mut(&mut self, frame: usize) -> impl ExactSizeIterator<Item = &mut S> {
        assert!(frame < self.num_frames);
        self.ptrs[..self.num_channels as usize]
            .iter()
            // Safety: each reference is derived from its own channel pointer at
            // the same frame, and the constructor's non-overlap contract makes
            // those samples distinct, so the yielded references never alias.
            .map(move |&ptr| unsafe { &mut *ptr.add(frame) })
    }

    #[nonblocking]
    fn as_view_mut(&mut self) -> impl AudioBlockMut<S> {
        self.view_mut()
    }

    #[nonblocking]
    fn for_each_allocated(&mut self, mut f: impl FnMut(&mut S)) {
        let num_frames = self.num_frames_allocated;
        for &ptr in self.ptrs {
            // Safety: every pointer is valid for `num_frames_allocated`
            // elements, the channels are disjoint, and only one channel slice is
            // live at a time.
            let channel = unsafe { core::slice::from_raw_parts_mut(ptr, num_frames) };
            channel.iter_mut().for_each(&mut f);
        }
    }

    #[nonblocking]
    fn enumerate_allocated(&mut self, mut f: impl FnMut(u16, usize, &mut S)) {
        let num_frames = self.num_frames_allocated;
        for (channel, &ptr) in self.ptrs.iter().enumerate() {
            // Safety: see `for_each_allocated`.
            let samples = unsafe { core::slice::from_raw_parts_mut(ptr, num_frames) };
            for (frame, sample) in samples.iter_mut().enumerate() {
                f(channel as u16, frame, sample);
            }
        }
    }
}

impl<S: Sample + core::fmt::Debug> core::fmt::Debug for PlanarPtrsMut<'_, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "audio_blocks::PlanarPtrsMut {{")?;
        writeln!(f, "  num_channels: {}", self.num_channels)?;
        writeln!(f, "  num_frames: {}", self.num_frames)?;
        writeln!(f, "  num_channels_allocated: {}", self.ptrs.len())?;
        writeln!(f, "  num_frames_allocated: {}", self.num_frames_allocated)?;
        writeln!(f, "  channels:")?;

        for (i, channel) in self.channels().enumerate() {
            writeln!(f, "    {}: {:?}", i, channel)?;
        }

        writeln!(f, "}}")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtsan_standalone::no_sanitize_realtime;

    #[test]
    fn test_pointer() {
        let data = [vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let ptrs: Vec<*const f32> = data.iter().map(|channel| channel.as_ptr()).collect();

        let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 5) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.layout(), crate::BlockLayout::Planar);

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
    fn test_pointer_mut() {
        let mut data = [vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![1.0, 3.0, 5.0, 7.0, 9.0]];
        let ptrs: Vec<*mut f32> = data
            .iter_mut()
            .map(|channel| channel.as_mut_ptr())
            .collect();

        let block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 5) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_channels_allocated(), 2);
        assert_eq!(block.num_frames(), 5);
        assert_eq!(block.num_frames_allocated(), 5);
        assert_eq!(block.layout(), crate::BlockLayout::Planar);

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
    fn test_member_functions() {
        let data = [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let ptrs: Vec<*const f32> = data.iter().map(|channel| channel.as_ptr()).collect();
        let block = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 2, 3, 4) };

        // single channel
        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[4.0, 5.0, 6.0]);

        // all channels
        let mut channels = block.channels();
        assert_eq!(channels.next().unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(channels.next().unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(channels.next(), None);
        drop(channels);

        // views
        let view = block.view();
        assert_eq!(view.num_channels(), block.num_channels());
        assert_eq!(view.num_frames(), block.num_frames());
        assert_eq!(
            view.num_channels_allocated(),
            block.num_channels_allocated()
        );
        assert_eq!(view.num_frames_allocated(), block.num_frames_allocated());
    }

    #[test]
    fn test_member_functions_mut() {
        let mut data = [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let ptrs: Vec<*mut f32> = data
            .iter_mut()
            .map(|channel| channel.as_mut_ptr())
            .collect();
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 2, 3, 4) };

        // single channel
        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[4.0, 5.0, 6.0]);

        assert_eq!(block.channel_mut(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel_mut(1), &[4.0, 5.0, 6.0]);

        // all channels
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

        // views
        {
            let view = block.view();
            assert_eq!(view.num_channels(), 2);
            assert_eq!(view.num_frames(), 3);
            assert_eq!(view.num_channels_allocated(), 3);
            assert_eq!(view.num_frames_allocated(), 4);
            assert_eq!(view.channel(1), &[4.0, 5.0, 6.0]);
        }

        {
            let mut view = block.view_mut();
            assert_eq!(view.num_channels(), 2);
            assert_eq!(view.num_frames(), 3);
            assert_eq!(view.num_channels_allocated(), 3);
            assert_eq!(view.num_frames_allocated(), 4);
            view.channel_mut(0)[0] = 100.0;
        }

        assert_eq!(block.channel(0), &[100.0, 1.0, 2.0]);
    }

    #[test]
    fn test_samples() {
        let ch0 = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ch1 = [5.0, 6.0, 7.0, 8.0, 9.0];
        let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
        let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 5) };

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
    fn test_samples_mut() {
        let mut ch0 = [0.0f32; 5];
        let mut ch1 = [0.0f32; 5];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 5) };

        let num_frames = block.num_frames();
        for ch in 0..block.num_channels() {
            for f in 0..num_frames {
                *block.sample_mut(ch, f) = (ch as usize * num_frames + f) as f32;
            }
        }

        for ch in 0..block.num_channels() {
            for f in 0..num_frames {
                assert_eq!(block.sample(ch, f), (ch as usize * num_frames + f) as f32);
            }
        }

        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(block.channel(1), &[5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_frame_iters() {
        let ch0 = [0.0, 2.0, 4.0, 6.0, 8.0];
        let ch1 = [1.0, 3.0, 5.0, 7.0, 9.0];
        let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
        let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 5) };

        for f in 0..block.num_frames() {
            assert_eq!(
                block.frame_iter(f).copied().collect::<Vec<_>>(),
                vec![(2 * f) as f32, (2 * f + 1) as f32]
            );
        }

        let mut frames_iter = block.frames_iter();
        for f in 0..5 {
            assert_eq!(
                frames_iter.next().unwrap().copied().collect::<Vec<_>>(),
                vec![(2 * f) as f32, (2 * f + 1) as f32]
            );
        }
        assert!(frames_iter.next().is_none());
    }

    #[test]
    fn test_frame_iters_mut() {
        let mut ch0 = [0.0f32; 4];
        let mut ch1 = [0.0f32; 4];
        let mut ch2 = [0.0f32; 4];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr(), ch2.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 4) };

        for f in 0..block.num_frames() {
            for (c, sample) in block.frame_iter_mut(f).enumerate() {
                *sample = (c * 10 + f) as f32;
            }
        }

        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(block.channel(1), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(block.channel(2), &[20.0, 21.0, 22.0, 23.0]);

        let mut frames_iter = block.frames_iter();
        assert_eq!(
            frames_iter.next().unwrap().copied().collect::<Vec<_>>(),
            vec![0.0, 10.0, 20.0]
        );
        assert_eq!(
            frames_iter.next().unwrap().copied().collect::<Vec<_>>(),
            vec![1.0, 11.0, 21.0]
        );
    }

    #[test]
    fn test_channel_iters() {
        let ch0 = [0.0, 2.0, 4.0, 6.0, 8.0];
        let ch1 = [1.0, 3.0, 5.0, 7.0, 9.0];
        let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
        let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 5) };

        let mut channels_iter = block.channels_iter();
        assert_eq!(
            channels_iter.next().unwrap().copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
        );
        assert_eq!(
            channels_iter.next().unwrap().copied().collect::<Vec<_>>(),
            vec![1.0, 3.0, 5.0, 7.0, 9.0]
        );
        assert!(channels_iter.next().is_none());
    }

    #[test]
    fn test_channel_iters_mut() {
        let mut ch0 = [0.0f32; 5];
        let mut ch1 = [0.0f32; 5];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 5) };

        block
            .channel_iter_mut(0)
            .enumerate()
            .for_each(|(i, v)| *v = i as f32);

        let mut channels_iter = block.channels_iter_mut();
        channels_iter.next().unwrap().for_each(|v| *v += 100.0);
        channels_iter
            .next()
            .unwrap()
            .enumerate()
            .for_each(|(i, v)| *v = i as f32 + 10.0);
        assert!(channels_iter.next().is_none());
        drop(channels_iter);

        assert_eq!(block.channel(0), &[100.0, 101.0, 102.0, 103.0, 104.0]);
        assert_eq!(block.channel(1), &[10.0, 11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channels_mut_are_disjoint() {
        // Two channel slices are held at once, which is only sound because each
        // one is derived from its own channel pointer.
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];

        {
            let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 3) };

            let mut channels = block.channels_mut();
            let first = channels.next().unwrap();
            let second = channels.next().unwrap();
            first.fill(1.0);
            second.fill(2.0);
            first[0] = 3.0;
            second[2] = 4.0;
        }

        assert_eq!(ch0, [3.0, 1.0, 1.0]);
        assert_eq!(ch1, [2.0, 2.0, 4.0]);
    }

    #[test]
    fn test_multiple_views_from_same_pointers() {
        let ch0 = [0.0, 1.0, 2.0];
        let ch1 = [3.0, 4.0, 5.0];
        let ptrs = [ch0.as_ptr(), ch1.as_ptr()];

        let first = unsafe { PlanarPtrs::from_ptrs(&ptrs, 3) };
        let second = unsafe { PlanarPtrs::from_ptrs(&ptrs, 3) };
        let third = first.view();

        assert_eq!(first.channel(1), &[3.0, 4.0, 5.0]);
        assert_eq!(second.channel(1), &[3.0, 4.0, 5.0]);
        assert_eq!(third.channel(1), &[3.0, 4.0, 5.0]);
        assert_eq!(first.sample(0, 2), second.sample(0, 2));
    }

    #[test]
    fn test_overlapping_pointers_are_fine_read_only() {
        // The read-only type makes no exclusivity claim, so the same buffer may
        // appear as several channels.
        let ch = [1.0, 2.0, 3.0];
        let ptrs = [ch.as_ptr(), ch.as_ptr(), unsafe { ch.as_ptr().add(1) }];
        let block = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 3, 2, 2) };

        assert_eq!(block.channel(0), &[1.0, 2.0]);
        assert_eq!(block.channel(1), &[1.0, 2.0]);
        assert_eq!(block.channel(2), &[2.0, 3.0]);
    }

    #[test]
    fn test_limited() {
        let data = [[0.0f32; 4], [1.0; 4], [2.0; 4]];
        let ptrs: Vec<*const f32> = data.iter().map(|channel| channel.as_ptr()).collect();
        let block = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 2, 3, 4) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 4);

        for c in 0..block.num_channels() {
            assert_eq!(block.channel_iter(c).count(), 3);
            assert_eq!(block.channel(c).len(), 3);
        }
        for f in 0..block.num_frames() {
            assert_eq!(block.frame_iter(f).count(), 2);
        }
        assert_eq!(block.channels().count(), 2);
        assert_eq!(block.frames_iter().count(), 3);
    }

    #[test]
    fn test_limited_mut() {
        let mut data = [[0.0f32; 4], [1.0; 4], [2.0; 4]];
        let ptrs: Vec<*mut f32> = data
            .iter_mut()
            .map(|channel| channel.as_mut_ptr())
            .collect();
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 2, 3, 4) };

        assert_eq!(block.num_channels(), 2);
        assert_eq!(block.num_frames(), 3);
        assert_eq!(block.num_channels_allocated(), 3);
        assert_eq!(block.num_frames_allocated(), 4);

        for c in 0..block.num_channels() {
            assert_eq!(block.channel_iter(c).count(), 3);
            assert_eq!(block.channel_iter_mut(c).count(), 3);
        }
        for f in 0..block.num_frames() {
            assert_eq!(block.frame_iter(f).count(), 2);
            assert_eq!(block.frame_iter_mut(f).count(), 2);
        }

        // The visible region can grow up to the allocated region.
        block.set_visible(3, 4);
        assert_eq!(block.channel_iter(2).count(), 4);
        assert_eq!(block.frame_iter(3).count(), 3);
    }

    #[test]
    fn test_mut_round_trip() {
        let mut ch0 = vec![0.0f32; 4];
        let mut ch1 = vec![0.0f32; 4];
        let mut ch2 = vec![0.0f32; 4];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr(), ch2.as_mut_ptr()];

        {
            let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 4) };
            for (c, channel) in block.channels_iter_mut().enumerate() {
                for (f, sample) in channel.enumerate() {
                    *sample = (c * 10 + f) as f32;
                }
            }
            block.gain(2.0);
        }

        assert_eq!(ch0, vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(ch1, vec![20.0, 22.0, 24.0, 26.0]);
        assert_eq!(ch2, vec![40.0, 42.0, 44.0, 46.0]);
    }

    #[test]
    fn test_for_each_and_enumerate_visible() {
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 1, 2, 3) };

        block.for_each(|v| *v = 1.0);
        assert_eq!(block.channel(0), &[1.0, 1.0]);

        let mut indices = Vec::new();
        block.enumerate(|c, f, v| {
            indices.push((c, f));
            *v = (c as usize * 10 + f) as f32;
        });
        assert_eq!(indices, vec![(0, 0), (0, 1)]);
        assert_eq!(block.channel(0), &[0.0, 1.0]);

        // The hidden channel and frame stayed untouched.
        block.set_visible(2, 3);
        assert_eq!(block.channel(0), &[0.0, 1.0, 0.0]);
        assert_eq!(block.channel(1), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_enumerate_allocated_order() {
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 1, 1, 3) };

        let mut indices = Vec::new();
        block.enumerate_allocated(|c, f, v| {
            indices.push((c, f));
            *v = (c as usize * 10 + f) as f32;
        });

        // Planar order is channel-major over the whole allocation.
        assert_eq!(
            indices,
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        );

        block.set_visible(2, 3);
        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_for_each_allocated_covers_hidden_region() {
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 1, 1, 3) };

        block.fill_with(5.0);

        block.set_visible(2, 3);
        assert_eq!(block.channel(0), &[5.0, 5.0, 5.0]);
        assert_eq!(block.channel(1), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_as_view() {
        let ch0 = [0.0, 1.0, 2.0];
        let ch1 = [3.0, 4.0, 5.0];
        let ptrs = [ch0.as_ptr(), ch1.as_ptr()];
        let block = unsafe { PlanarPtrs::from_ptrs(&ptrs, 3) };

        let view = block.as_view();
        assert_eq!(view.num_channels(), 2);
        assert_eq!(
            view.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_as_view_mut() {
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        let ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 3) };

        {
            let mut view = block.as_view_mut();
            view.channel_iter_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32);
            view.channel_iter_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i as f32 + 10.0);
        }

        assert_eq!(block.channel(0), &[0.0, 1.0, 2.0]);
        assert_eq!(block.channel(1), &[10.0, 11.0, 12.0]);

        let view = block.as_view();
        assert_eq!(
            view.channel_iter(1).copied().collect::<Vec<_>>(),
            vec![10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_debug() {
        let mut ch0 = [0.0f32; 2];
        let ptrs = [ch0.as_mut_ptr()];
        let block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 2) };
        assert!(format!("{block:?}").contains("PlanarPtrsMut"));

        let view = block.view();
        assert!(format!("{view:?}").contains("PlanarPtrs"));
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_channel_out_of_bounds() {
        let data = [[0.0f32; 4], [0.0; 4], [0.0; 4]];
        let ptrs: Vec<*const f32> = data.iter().map(|channel| channel.as_ptr()).collect();
        let block = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 2, 3, 4) };

        block.channel(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_channel_out_of_bounds_mut() {
        let mut data = [[0.0f32; 4], [0.0; 4], [0.0; 4]];
        let ptrs: Vec<*mut f32> = data
            .iter_mut()
            .map(|channel| channel.as_mut_ptr())
            .collect();
        let mut block = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 2, 3, 4) };

        block.channel_mut(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_frame_out_of_bounds() {
        let ch0 = [0.0f32; 4];
        let ptrs = [ch0.as_ptr()];
        let block = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 1, 3, 4) };

        block.sample(0, 3);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_too_many_channels_visible() {
        let ch0 = [0.0f32; 4];
        let ptrs = [ch0.as_ptr()];
        let _ = unsafe { PlanarPtrs::from_ptrs_limited(&ptrs, 2, 4, 4) };
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_too_many_frames_visible() {
        let mut ch0 = [0.0f32; 4];
        let ptrs = [ch0.as_mut_ptr()];
        let _ = unsafe { PlanarPtrsMut::from_ptrs_limited(&ptrs, 1, 5, 4) };
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_channels() {
        let mut ch0 = [0.0f32; 4];
        let ptrs = [ch0.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 4) };

        block.set_num_channels_visible(2);
    }

    #[test]
    #[should_panic]
    #[no_sanitize_realtime]
    fn test_wrong_resize_frames() {
        let mut ch0 = [0.0f32; 4];
        let ptrs = [ch0.as_mut_ptr()];
        let mut block = unsafe { PlanarPtrsMut::from_ptrs(&ptrs, 4) };

        block.set_num_frames_visible(5);
    }
}
