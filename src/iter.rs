use core::{marker::PhantomData, ptr::NonNull};

use crate::Sample;

/// An iterator for traversing samples with a fixed stride pattern.
///
/// Used internally for two purposes:
/// 1. Iterating over channels when the underlying data is interleaved
/// 2. Iterating over frames when the underlying data is sequential
///
/// This allows efficient access to non-contiguous samples in memory.
#[derive(Clone)]
pub(crate) struct StridedSampleIter<'a, S: Sample> {
    pub ptr: NonNull<S>,  // Pointer to the current sample for this channel or frame
    pub stride: usize,    // How many elements to jump to get to the next frame or channel
    pub remaining: usize, // Number of frames left for this channel or frame
    pub _marker: PhantomData<&'a S>, // Links the output lifetime &'a S to the borrow in channels()/ frames()
}

impl<'a, S: Sample> Iterator for StridedSampleIter<'a, S> {
    type Item = &'a S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            unsafe {
                let current_ptr = self.ptr; // Pointer to the item to return
                self.remaining -= 1; // Decrement remaining count first

                // Only calculate the *next* pointer if there are more items after this one
                if self.remaining > 0 {
                    // This add operation should now always result in a pointer
                    // within the allocation or at most one-past-the-end,
                    // because we know `current_ptr` isn't the absolute last pointer
                    // this iterator instance will access.
                    self.ptr = NonNull::new_unchecked(current_ptr.as_ptr().add(self.stride));
                }
                // else: No need to advance self.ptr, iterator is finished.

                // Return reference to the *current* sample
                Some(&*current_ptr.as_ptr())
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<S: Sample> ExactSizeIterator for StridedSampleIter<'_, S> {}

/// An iterator for traversing samples with a fixed stride pattern.
///
/// Used internally for two purposes:
/// 1. Iterating over channels when the underlying data is interleaved
/// 2. Iterating over frames when the underlying data is sequential
///
/// This allows efficient access to non-contiguous samples in memory.
pub(crate) struct StridedSampleIterMut<'a, S: Sample> {
    pub ptr: NonNull<S>,
    pub stride: usize,
    pub remaining: usize,
    pub _marker: PhantomData<&'a mut S>, // Links the output lifetime &'a mut S
}

impl<'a, S: Sample> Iterator for StridedSampleIterMut<'a, S> {
    type Item = &'a mut S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            unsafe {
                // Get pointer to the current item. Need to cast to *mut for the return type.
                let current_mut_ptr = self.ptr.as_ptr();
                self.remaining -= 1; // Decrement remaining count first

                // Only calculate the *next* pointer if there are more items after this one
                if self.remaining > 0 {
                    // Same safety reasoning as the immutable version.
                    self.ptr = NonNull::new_unchecked(current_mut_ptr.add(self.stride));
                }
                // else: No need to advance self.ptr, iterator is finished.

                // Return mutable reference to the *current* sample
                Some(&mut *current_mut_ptr)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<S: Sample> ExactSizeIterator for StridedSampleIterMut<'_, S> {}
