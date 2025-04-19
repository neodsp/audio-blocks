#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::Stacked;
pub use view::{StackedPtrAdapter, StackedView};
pub use view_mut::{StackedPtrAdapterMut, StackedViewMut};
