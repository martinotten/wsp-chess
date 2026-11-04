//! Suchfenster und Knotendaten der Suche.

use super::INF;
use shakmaty::Move;

/// Suchfenster.
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

    /// Fenster aus Sicht des Gegners.
    pub(super) fn flip(self) -> Window {
        Window {
            alpha: -self.beta,
            beta: -self.alpha,
        }
    }

    /// Null-Fenster: prüft nur, ob ein Wert über `alpha` liegt.
    pub(super) fn null(alpha: i32) -> Window {
        Window {
            alpha,
            beta: alpha + 1,
        }
    }

    /// Fenster um einen erwarteten Wert.
    pub(super) fn around(score: i32, delta: i32) -> Window {
        Window {
            alpha: score - delta,
            beta: score + delta,
        }
    }

    /// Breiter als ein Null-Fenster: Knoten der Hauptvariante.
    pub(super) fn is_pv(self) -> bool {
        self.beta - self.alpha > 1
    }
}

/// Lage eines Knotens im Suchbaum.
#[derive(Clone, Copy, Debug)]
pub(super) struct Node {
    /// Abstand zur Wurzel in Halbzügen.
    pub(super) ply: usize,
    /// Zobrist-Schlüssel der Stellung.
    pub(super) key: u64,
    /// Nullzug erlaubt (nicht an der Wurzel, nicht zweimal hintereinander).
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

/// Lage eines Knotens der Ruhesuche.
#[derive(Clone, Copy, Debug)]
pub(super) struct QNode {
    /// Abstand zur Wurzel in Halbzügen.
    pub(super) ply: usize,
    /// Höhe innerhalb der Ruhesuche (0 = erste Ebene).
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

/// Was beim Durchsuchen der Züge eines Knotens gleich bleibt.
pub(super) struct Plan {
    pub(super) node: Node,
    pub(super) depth: i32,
    pub(super) in_check: bool,
    /// Aussichtsloser Knoten: statische Bewertung weit unter alpha, ruhige Züge holen das nicht auf (Futility).
    pub(super) futile: bool,
    pub(super) tt_move: u16,
}

/// Ein Zug in der Zugliste eines Knotens.
pub(super) struct Candidate {
    pub(super) mv: Move,
    pub(super) code: u16,
    /// Position in der sortierten Liste (0 = erster).
    pub(super) index: usize,
    pub(super) quiet: bool,
    pub(super) gives_check: bool,
}

impl Candidate {
    /// Ruhiger Zug ohne Schach, nicht der erste: kann den Wert eines aussichtslosen Knotens nicht
    /// mehr retten (Futility, siehe [`Plan::futile`]). Der erste Zug wird immer durchsucht, damit
    /// der Knoten einen Wert bekommt.
    pub(super) fn cannot_rescue_hopeless_node(&self) -> bool {
        self.index > 0 && self.quiet && !self.gives_check
    }
}

/// Ergebnis der Transpositionstabelle für einen Knoten.
pub(super) enum TtProbe {
    /// Gespeicherter Wert reicht, der Knoten ist erledigt.
    Cutoff(i32),
    /// Hinweise für die Suche: bester Zug (0 = keiner) und statische Bewertung.
    Hint { mv: u16, eval: Option<i32> },
}

/// Bester Zug eines Knotens.
pub(super) struct Best {
    pub(super) score: i32,
    pub(super) code: u16,
}
