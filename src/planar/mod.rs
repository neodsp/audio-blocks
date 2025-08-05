#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::AudioBlockPlanar;
pub use view::{AudioBlockPlanarView, PlanarPtrAdapter};
pub use view_mut::{AudioBlockPlanarViewMut, PlanarPtrAdapterMut};
