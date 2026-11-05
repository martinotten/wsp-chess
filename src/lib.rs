//! Chess engine using the hand-written network (weight-space programming) as its evaluation.
//!
//! - [`eval`] – sparse, incremental evaluation of the network
//! - [`exchange`] – material values and static exchange evaluation (SEE)
//! - [`search`] – alpha-beta search with switchable techniques
//! - [`tt`] – transposition table
//! - [`uci`] – UCI protocol

pub mod board;
pub mod eval;
pub mod exchange;
pub mod search;
pub mod tt;
pub mod uci;
