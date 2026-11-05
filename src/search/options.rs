//! Switches for the search techniques, limits of a search and its result.

use crate::board::Move;
use std::time::Duration;

/// Search techniques that can be switched off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    /// Principal Variation Search (null window after the first move).
    Pvs,
    /// Search window around the score of the previous iteration.
    Aspiration,
    /// Null-move pruning.
    NullMove,
    /// Late Move Reductions.
    Lmr,
    /// Futility pruning at depth 1–2.
    Futility,
    /// Skip captures with a losing exchange in the quiescence search.
    SeePrune,
    /// Delta pruning in the quiescence search.
    Delta,
    /// Extend checks by one ply.
    CheckExt,
    /// Killer moves in move ordering.
    Killers,
    /// History heuristic in move ordering.
    History,
    /// Quiet checks at the first level of the quiescence search (default: off).
    QChecks,
}

impl Feature {
    /// All techniques with their UCI name.
    pub const ALL: [(Feature, &'static str); 11] = [
        (Feature::Pvs, "PVS"),
        (Feature::Aspiration, "Aspiration"),
        (Feature::NullMove, "NullMove"),
        (Feature::Lmr, "LMR"),
        (Feature::Futility, "Futility"),
        (Feature::SeePrune, "SEEPrune"),
        (Feature::Delta, "Delta"),
        (Feature::CheckExt, "CheckExt"),
        (Feature::Killers, "Killers"),
        (Feature::History, "History"),
        (Feature::QChecks, "QChecks"),
    ];

    /// Technique for a UCI name (case-insensitive).
    #[must_use]
    pub fn by_name(name: &str) -> Option<Feature> {
        Feature::ALL
            .iter()
            .find(|(_, feature_name)| feature_name.eq_ignore_ascii_case(name))
            .map(|&(feature, _)| feature)
    }
}

/// On/off switch per [`Feature`]; default: all on except [`Feature::QChecks`].
#[derive(Clone, Copy, Debug)]
pub struct Opts([bool; Feature::ALL.len()]);

impl Default for Opts {
    fn default() -> Self {
        let mut opts = Opts([true; Feature::ALL.len()]);
        opts.set(Feature::QChecks, false);
        opts
    }
}

impl Opts {
    /// Is the technique enabled?
    #[must_use]
    pub fn on(&self, feature: Feature) -> bool {
        self.0[feature as usize]
    }

    /// Switches the technique on or off.
    pub fn set(&mut self, feature: Feature, value: bool) {
        self.0[feature as usize] = value;
    }
}

/// Limits of a search.
#[derive(Clone, Copy, Default, Debug)]
pub struct Limits {
    /// Thinking time.
    pub movetime: Option<Duration>,
    /// Maximum iteration depth.
    pub depth: Option<i32>,
    /// Maximum number of nodes.
    pub nodes: Option<u64>,
    /// Search until stopped from outside.
    pub infinite: bool,
}

/// Result of a completed iteration.
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// Iteration depth.
    pub depth: i32,
    /// Maximum ply reached (including quiescence search).
    pub seldepth: usize,
    /// Score in cp or mate score.
    pub score: i32,
    /// Nodes visited.
    pub nodes: u64,
    /// Network evaluations.
    pub evals: u64,
    /// Search time so far.
    pub time: Duration,
    /// Principal variation.
    pub pv: Vec<Move>,
    /// Transposition table usage in per mille.
    pub hashfull: usize,
}
