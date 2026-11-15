// ! SHE Client module for UE
#[cfg(feature = "demo-d443a-edge-budget")]
pub mod d443a;
pub mod task;
pub use task::SheClientTask;
