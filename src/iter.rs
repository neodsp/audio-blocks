use core::{marker::PhantomData, ptr::NonNull};

use crate::Sample;

#[derive(Clone)] // Can be cloned if S is Copy
pub(crate) struct InterleavedDataIter<'a, S: Sample> {
    pub ptr: NonNull<S>,  // Pointer to the current sample for this channel or frame
    pub stride: usize,    // How many elements to jump to get to the next frame or channel
    pub remaining: usize, // Number of frames left for this channel or frame
    pub _marker: PhantomData<&'a S>, // Links the output lifetime &'a S to the borrow in channels()/ frames()
}

impl<'a, S: Sample> Iterator for InterleavedDataIter<'a, S> {
    type Item = &'a S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            unsafe {
                let current_ptr = self.ptr;
                // Advance pointer for the *next* call. Use NonNull::new unchecked or expect
                // as stride should not make a valid pointer null.
                self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(self.stride));
                self.remaining -= 1;
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

impl<S: Sample> ExactSizeIterator for InterleavedDataIter<'_, S> {}

pub(crate) struct InterleavedDataIterMut<'a, S: Sample> {
    pub ptr: NonNull<S>,
    pub stride: usize,
    pub remaining: usize,
    pub _marker: PhantomData<&'a mut S>, // Links the output lifetime &'a mut S
}

impl<'a, S: Sample> Iterator for InterleavedDataIterMut<'a, S> {
    type Item = &'a mut S;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            unsafe {
                let current_ptr = self.ptr;
                self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().add(self.stride));
                self.remaining -= 1;
                // Return mutable reference
                Some(&mut *current_ptr.as_ptr())
            }
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl<S: Sample> ExactSizeIterator for InterleavedDataIterMut<'_, S> {}
