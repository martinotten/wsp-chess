//! Sparse evaluation of the hand-written network (export: `generator/export.py`).
//!
//! Targets `T = [z1 | z2 | z3 | out]`. Every source (768 inputs, then h1, h2, h3) has a list of
//! (target index, weight) pairs into later layers. Because only ~20 h1, ~24 h2 and ~2 h3 neurons are
//! active per position, only their rows are summed. Result `s` in pawns, from the side to move.
//!
//! The input part (biases plus the contributions of all set input bits to all targets) is called the
//! accumulator. [`Net::update`] carries it over from one board to another incrementally.

use crate::board::{Color, Piece, Position, Square};
use cozy_chess::Board;
use std::fmt;

/// Number of input bits (12 piece kinds × 64 squares).
pub const N_INPUTS: usize = 768;
const MAGIC: &[u8; 4] = b"WSPN";
const VERSION: u32 = 1;
/// Maximum number of changed features up to which [`Net::update`] updates instead of recomputing.
pub const MAX_CHANGES: usize = 12;

/// Error while loading the weights file.
#[derive(Debug)]
pub enum LoadError {
    /// File not readable.
    Io(std::io::Error),
    /// Content does not match the format.
    Format(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(err) => write!(out, "weights file not readable: {err}"),
            LoadError::Format(reason) => write!(out, "weights file invalid: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// The network in sparse form (one list of targets per source).
///
/// Sources are the input features and the neurons h1, h2, h3; targets are the pre-activations z1,
/// z2, z3 and the output, stored consecutively in one vector. The weights of source `src` are in
/// `targets[starts[src]..starts[src + 1]]` and `weights[…]` (same range).
pub struct Net {
    /// Sizes of the layers h1, h2, h3.
    n1: usize,
    n2: usize,
    n3: usize,
    bias: Vec<f32>,
    starts: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<f32>,
}

/// Input index as `xi` in `generator/constants.py`: `(side*6 + kind)*64 + square`, from the view of
/// `persp` (side 0 = pieces of `persp`; if `persp` is Black, the board is mirrored vertically).
#[must_use]
pub fn feature(persp: Color, color: Color, role: Piece, sq: Square) -> usize {
    let side = usize::from(color != persp);
    let seen_sq = if persp == Color::Black { sq.flip_rank() } else { sq };
    (side * 6 + role as usize) * 64 + seen_sq as usize
}

/// Calls `visit(index, activation)` for all positive values; blocks of 16 values are checked as a
/// whole first, since almost all neurons are inactive.
///
/// The pre-check uses the bit pattern as `i32`: a positive number has a pattern `> 0`, whereas `0.0`,
/// `-0.0` and negative numbers do not. An integer maximum vectorizes reliably. The pre-check never
/// skips a positive value; the final decision is still made with `value > 0.0`.
#[inline]
fn for_active(values: &[f32], mut visit: impl FnMut(usize, f32)) {
    for (block_index, block) in values.chunks(16).enumerate() {
        let max_bits = block
            .iter()
            .fold(i32::MIN, |max, value| max.max(value.to_bits().cast_signed()));
        if max_bits > 0 {
            for (offset, &value) in block.iter().enumerate() {
                if value > 0.0 {
                    visit(block_index * 16 + offset, value);
                }
            }
        }
    }
}

/// Reads fixed-width little-endian values in sequence.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8], LoadError> {
        if self.bytes.len() < len {
            return Err(LoadError::Format("file too short"));
        }
        let (head, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        Ok(head)
    }

    fn u32s(&mut self, count: usize) -> Result<Vec<u32>, LoadError> {
        Ok(self
            .take(4 * count)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect())
    }

    fn f32s(&mut self, count: usize) -> Result<Vec<f32>, LoadError> {
        Ok(self
            .take(4 * count)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect())
    }

    fn usizes(&mut self, count: usize) -> Result<Vec<usize>, LoadError> {
        self.u32s(count)?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| LoadError::Format("index too large")))
            .collect()
    }
}

impl Net {
    /// Loads the weights from a file in the format written by `generator/export.py`.
    ///
    /// # Errors
    /// If the file is missing or does not match the format.
    pub fn load(path: &str) -> Result<Net, LoadError> {
        Net::from_bytes(&std::fs::read(path).map_err(LoadError::Io)?)
    }

    /// Reads the weights from bytes and validates the structure: every source feeds only later
    /// layers. This guarantees that [`Net::score_from`] and [`Net::update`] only write within the
    /// vectors.
    ///
    /// # Errors
    /// If the bytes do not match the format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, LoadError> {
        let mut reader = Reader { bytes };
        if reader.take(4)? != MAGIC {
            return Err(LoadError::Format("magic WSPN missing"));
        }
        if reader.u32s(1)? != [VERSION] {
            return Err(LoadError::Format("unknown version"));
        }
        let sizes = reader.usizes(3)?;
        let (n1, n2, n3) = (sizes[0], sizes[1], sizes[2]);
        let n_targets = n1 + n2 + n3 + 1;
        let n_sources = N_INPUTS + n1 + n2 + n3;
        let bias = reader.f32s(n_targets)?;
        let starts = reader.usizes(n_sources + 1)?;
        let nnz = starts[n_sources];
        let targets = reader.usizes(nnz)?;
        let weights = reader.f32s(nnz)?;
        if !reader.bytes.is_empty() {
            return Err(LoadError::Format("trailing data after the end"));
        }
        if starts[0] != 0 || starts.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(LoadError::Format("row starts not ascending"));
        }
        // First allowed target per source: inputs -> all, h1 -> from z2, h2 -> from z3, h3 -> out only
        for src in 0..n_sources {
            let min = if src < N_INPUTS {
                0
            } else if src < N_INPUTS + n1 {
                n1
            } else if src < N_INPUTS + n1 + n2 {
                n1 + n2
            } else {
                n1 + n2 + n3
            };
            if targets[starts[src]..starts[src + 1]]
                .iter()
                .any(|&target| target < min || target >= n_targets)
            {
                return Err(LoadError::Format("weight into an earlier layer"));
            }
        }
        Ok(Net {
            n1,
            n2,
            n3,
            bias,
            starts,
            targets,
            weights,
        })
    }

    /// Length of an accumulator.
    #[must_use]
    pub fn n_targets(&self) -> usize {
        self.n1 + self.n2 + self.n3 + 1
    }

    /// Length of the work vector for [`Net::score_from`] (z2, z3, out).
    #[must_use]
    pub fn n_work(&self) -> usize {
        self.n2 + self.n3 + 1
    }

    /// Targets and weights of source `src` in stored order (reference for tests).
    #[cfg(test)]
    fn row(&self, src: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let row = self.starts[src]..self.starts[src + 1];
        self.targets[row.clone()]
            .iter()
            .copied()
            .zip(self.weights[row].iter().copied())
    }

    /// Adds `scale` × the weights of source `src` to `dest`, which starts at target `offset`.
    #[inline]
    fn add_row(&self, src: usize, scale: f32, dest: &mut [f32], offset: usize) {
        let row = self.starts[src]..self.starts[src + 1];
        for (&target, &weight) in self.targets[row.clone()].iter().zip(&self.weights[row]) {
            dest[target - offset] += weight * scale;
        }
    }

    /// Computes the accumulator for one perspective from scratch.
    pub fn refresh(&self, board: &Board, persp: Color, acc: &mut [f32]) {
        acc.copy_from_slice(&self.bias);
        // ascending squares: fixed summation order so that values are bit-identically reproducible
        for sq in board.occupied() {
            if let (Some(color), Some(role)) = (board.color_on(sq), board.piece_on(sq)) {
                self.add_row(feature(persp, color, role, sq), 1.0, acc, 0);
            }
        }
    }

    /// Updates the accumulator `acc` (perspective `persp`) from board `from` to board `to`.
    /// Returns `false` (without changes) if more than [`MAX_CHANGES`] features differ; a full
    /// recomputation with [`Net::refresh`] is cheaper then. `buf` is scratch space.
    pub fn update(
        &self,
        from: &Board,
        to: &Board,
        persp: Color,
        acc: &mut [f32],
        buf: &mut Vec<(usize, f32)>,
    ) -> bool {
        buf.clear();
        for color in Color::ALL {
            for role in Piece::ALL {
                let (before, after) = (from.colored_pieces(color, role), to.colored_pieces(color, role));
                buf.extend(
                    (before & !after)
                        .into_iter()
                        .map(|sq| (feature(persp, color, role, sq), -1.0)),
                );
                buf.extend(
                    (after & !before)
                        .into_iter()
                        .map(|sq| (feature(persp, color, role, sq), 1.0)),
                );
                if buf.len() > MAX_CHANGES {
                    return false;
                }
            }
        }
        for &(input, sign) in buf.iter() {
            self.add_row(input, sign, acc, 0);
        }
        true
    }

    /// Evaluation `s` (pawns) from the accumulator of the side to move's perspective.
    /// `work` must have length [`Net::n_work`].
    pub fn score_from(&self, acc: &[f32], work: &mut [f32]) -> f32 {
        let (n1, n2, n3) = (self.n1, self.n2, self.n3);
        work.copy_from_slice(&acc[n1..]);
        for_active(&acc[..n1], |neuron, act| {
            self.add_row(N_INPUTS + neuron, act, work, n1);
        });
        let (h2, rest) = work.split_at_mut(n2);
        for_active(h2, |neuron, act| {
            self.add_row(N_INPUTS + n1 + neuron, act, rest, n1 + n2);
        });
        let (h3, out) = rest.split_at_mut(n3);
        for_active(h3, |neuron, act| {
            self.add_row(N_INPUTS + n1 + n2 + neuron, act, out, n1 + n2 + n3);
        });
        out[0]
    }

    /// Evaluation without an accumulator (reference for tests and the `eval` command).
    #[must_use]
    pub fn score(&self, pos: &Position) -> f32 {
        let mut acc = vec![0.0; self.n_targets()];
        let mut work = vec![0.0; self.n_work()];
        self.refresh(pos.board(), pos.turn(), &mut acc);
        self.score_from(&acc, &mut work)
    }
}

#[cfg(test)]
mod tests;
