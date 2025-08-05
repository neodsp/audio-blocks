#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::AudioBlockSequential;
pub use view::AudioBlockSequentialView;
pub use view_mut::AudioBlockSequentialViewMut;
