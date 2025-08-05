#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::AudioBlockInterleaved;
pub use view::AudioBlockInterleavedView;
pub use view_mut::AudioBlockInterleavedViewMut;
