#[cfg(any(feature = "std", feature = "alloc"))]
mod owned;
mod view;
mod view_mut;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use owned::Sequential;
pub use view::SequentialView;
pub use view_mut::SequentialViewMut;
