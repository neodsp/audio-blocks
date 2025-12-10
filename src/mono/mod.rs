#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::AudioBlockMono;
pub use view::AudioBlockMonoView;
pub use view_mut::AudioBlockMonoViewMut;
