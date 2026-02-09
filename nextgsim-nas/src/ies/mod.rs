//! NAS Information Elements (IEs)
//!
//! This module contains implementations of NAS Information Elements
//! as defined in 3GPP TS 24.501.
//!
//! ## IE Types
//!
//! - Type 1: Half-octet (4 bits) - [`ie1`]
//! - Type 2: Single octet (8 bits) - to be implemented
//! - Type 3: Fixed length - [`ie3`]
//! - Type 4: Variable length (TLV) - [`ie4`]
//! - Type 6: Variable length (TLV-E) - [`ie6`]

pub mod ie1;
pub mod ie3;
pub mod ie4;
pub mod ie6;

pub use ie1::*;
pub use ie3::*;
pub use ie4::*;
pub use ie6::*;
