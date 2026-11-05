//! Search window and node data of the search.

use super::INF;
use crate::board::Move;

/// Search window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Window {
    pub(super) alpha: i32,
    pub(super) beta: i32,
}

impl Window {
    pub(super) const FULL: Window = Window {
        alpha: -INF,
        beta: INF,
    };

    /// Window from the opponent's point of view.
    pub(super) fn flip(self) -> Window {
        Window {
            alpha: -self.beta,
            beta: -self.alpha,
        }
    }

    /// Null window: only tests whether a score lies above `alpha`.
    pub(super) fn null(alpha: i32) -> Window {
        Window {
            alpha,
            beta: alpha + 1,
        }
    }

    /// Window around an expected score.
    pub(super) fn around(score: i32, delta: i32) -> Window {
        Window {
            alpha: score - delta,
            beta: score + delta,
        }
    }

    /// Wider than a null window: principal variation node.
    pub(super) fn is_pv(self) -> bool {
        self.beta - self.alpha > 1
    }
}

/// Location of a node in the search tree.
#[derive(Clone, Copy, Debug)]
pub(super) struct Node {
    /// Distance from the root in plies.
    pub(super) ply: usize,
    /// Zobrist key of the position.
    pub(super) key: u64,
    /// Null move allowed (not at the root, not twice in a row).
    pub(super) can_null: bool,
}

impl Node {
    pub(super) fn root(key: u64) -> Node {
        Node {
            ply: 0,
            key,
            can_null: false,
        }
    }

    pub(super) fn child(self, key: u64) -> Node {
        Node {
            ply: self.ply + 1,
            key,
            can_null: true,
        }
    }

    pub(super) fn after_null_move(self, key: u64) -> Node {
        Node {
            ply: self.ply + 1,
            key,
            can_null: false,
        }
    }

    pub(super) fn is_root(self) -> bool {
        self.ply == 0
    }
}

/// Location of a quiescence search node.
#[derive(Clone, Copy, Debug)]
pub(super) struct QNode {
    /// Distance from the root in plies.
    pub(super) ply: usize,
    /// Ply within the quiescence search (0 = first level).
    pub(super) qdepth: u32,
}

impl QNode {
    pub(super) fn entry(ply: usize) -> QNode {
        QNode { ply, qdepth: 0 }
    }

    pub(super) fn child(self) -> QNode {
        QNode {
            ply: self.ply + 1,
            qdepth: self.qdepth + 1,
        }
    }
}

/// What stays the same while searching the moves of a node.
pub(super) struct Plan {
    pub(super) node: Node,
    pub(super) depth: i32,
    pub(super) in_check: bool,
    /// Hopeless node: static evaluation far below alpha, quiet moves cannot make up for it
    /// (futility).
    pub(super) futile: bool,
    pub(super) tt_move: u16,
}

/// A move in a node's move list.
pub(super) struct Candidate {
    pub(super) mv: Move,
    pub(super) code: u16,
    /// Position in the sorted list (0 = first).
    pub(super) index: usize,
    pub(super) quiet: bool,
    pub(super) gives_check: bool,
}

impl Candidate {
    /// Quiet move without check, not the first: can no longer rescue the score of a hopeless node
    /// (futility, see [`Plan::futile`]). The first move is always searched so that the node gets a
    /// score.
    pub(super) fn cannot_rescue_hopeless_node(&self) -> bool {
        self.index > 0 && self.quiet && !self.gives_check
    }
}

/// Transposition table result for a node.
pub(super) enum TtProbe {
    /// Stored score suffices, the node is done.
    Cutoff(i32),
    /// Hints for the search: best move (0 = none) and static evaluation.
    Hint { mv: u16, eval: Option<i32> },
}

/// Best move of a node.
pub(super) struct Best {
    pub(super) score: i32,
    pub(super) code: u16,
}
