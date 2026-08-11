#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::Planar;
pub use view::{PlanarPtrAdapter, PlanarView};
pub use view_mut::{PlanarPtrAdapterMut, PlanarViewMut};

use core::marker::PhantomData;

/// Maximum number of channels supported by [`PlanarViewMut`]'s mutable frame
/// iteration.
///
/// Mutable frame iteration caches one pointer per channel (see
/// [`PlanarFrameIterMut`]) in a fixed-size array, which caps how many channels a
/// [`PlanarViewMut`] can hold. The owned [`Planar`] type keeps this cache on the
/// heap and has no such limit.
pub const MAX_PLANAR_CHANNELS: usize = 64;

/// A mutable iterator over the samples of a single frame of planar audio data.
///
/// Yielded by the frame iterator that [`AudioBlockMut::frames_iter_mut`] returns
/// for planar blocks. It walks a slice of per-channel base pointers, offsetting
/// each by a fixed frame index to reach that channel's sample for this frame.
///
/// [`AudioBlockMut::frames_iter_mut`]: crate::AudioBlockMut::frames_iter_mut
pub struct PlanarFrameIterMut<'a, S> {
    /// Iterator over one base pointer per channel, all belonging to the same
    /// exclusively-borrowed block.
    bases: core::slice::Iter<'a, *mut S>,
    /// Frame offset applied to every channel base pointer.
    frame: usize,
    _marker: PhantomData<&'a mut S>,
}

impl<'a, S> PlanarFrameIterMut<'a, S> {
    /// Creates a frame iterator over `bases`, yielding the sample at `frame`
    /// within each channel.
    ///
    /// # Safety
    ///
    /// - Each pointer in `bases` must point to a channel buffer that is valid
    ///   for reads and writes for at least `frame + 1` samples of `S`.
    /// - For the entire lifetime `'a`, the pointed-to samples must be
    ///   exclusively owned by this borrow (no other live reference aliases
    ///   them). Distinct channels must point to non-overlapping buffers.
    /// - Two `PlanarFrameIterMut` sharing the same `bases` must use distinct
    ///   `frame` values so the samples they yield never overlap.
    pub(crate) unsafe fn new(bases: core::slice::Iter<'a, *mut S>, frame: usize) -> Self {
        Self {
            bases,
            frame,
            _marker: PhantomData,
        }
    }
}

impl<'a, S> Iterator for PlanarFrameIterMut<'a, S> {
    type Item = &'a mut S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let base = *self.bases.next()?;
        // Safety: `new`'s contract guarantees `base` is valid for `frame` and
        // that the sample is exclusively borrowed for `'a`. Distinct channels
        // and distinct frames never overlap, so every reference handed out is
        // to disjoint memory.
        Some(unsafe { &mut *base.add(self.frame) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.bases.size_hint()
    }
}

impl<S> ExactSizeIterator for PlanarFrameIterMut<'_, S> {
    #[inline]
    fn len(&self) -> usize {
        self.bases.len()
    }
}
