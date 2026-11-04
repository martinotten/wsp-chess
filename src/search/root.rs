//! Wurzel: iterative Vertiefung mit Aspiration-Fenstern, Rückfallzug und Hauptvariante.

use super::node::{Node, Window};
use super::{enc, hash, Feature, Info, Limits, Searcher, INF, MATE, MATE_BOUND, MAX_PLY};
use shakmaty::{Chess, Move, Position};
use std::time::Instant;

/// Größte Iterationstiefe.
const MAX_DEPTH: i32 = 100;

/// Aspiration: erste halbe Fensterbreite, ab dieser Breite volles Fenster, Mindesttiefe.
const ASPIRATION_DELTA: i32 = 40;
const ASPIRATION_MAX_DELTA: i32 = 1000;
const ASPIRATION_MIN_DEPTH: i32 = 4;

/// History: Teiler beim Start einer neuen Suche (ältere Erfahrungen verblassen).
const HISTORY_DECAY: i32 = 8;

impl Searcher {
    /// Legaler Zug zur Kodierung `e` (siehe [`enc`]); `None` für 0 oder unbekannte Züge.
    fn find(pos: &Chess, e: u16) -> Option<Move> {
        if e == 0 {
            return None;
        }
        pos.legal_moves().into_iter().find(|&m| enc(m) == e)
    }

    /// Hauptvariante aus der Transpositionstabelle, höchstens `n` Züge, ohne Wiederholungen.
    fn pv(&self, pos: &Chess, first: Move, n: usize) -> Vec<Move> {
        let mut line = vec![first];
        let mut p = pos.clone();
        p.play_unchecked(first);
        let mut seen = vec![hash(pos)];
        while line.len() < n {
            let h = hash(&p);
            if seen.contains(&h) {
                break;
            }
            seen.push(h);
            let Some(m) = self.tt.probe(h).and_then(|e| Self::find(&p, e.mv)) else {
                break;
            };
            line.push(m);
            p.play_unchecked(m);
        }
        line
    }

    fn start_search(&mut self, lim: &Limits) {
        self.start = Instant::now();
        self.deadline = lim.movetime.map(|t| self.start + t);
        self.max_nodes = lim.nodes;
        self.nodes = 0;
        self.evals = 0;
        self.aborted = false;
        self.seldepth = 0;
        self.tt.next_age();
        self.killers = [[0; 2]; MAX_PLY];
        for v in &mut self.history {
            *v /= HISTORY_DECAY;
        }
    }

    /// Rückfall, falls nicht einmal Tiefe 1 fertig wird: bester Zug nach Netz (ein Halbzug).
    fn one_ply_best(&mut self, pos: &Chess) -> Option<(Move, i32)> {
        let mut best = None;
        for m in pos.legal_moves() {
            let mut c = pos.clone();
            c.play_unchecked(m);
            let v = if c.is_checkmate() {
                MATE - 1
            } else {
                -self.eval(&c, 1)
            };
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((m, v));
            }
        }
        best
    }

    /// Eine Iteration mit Aspiration-Fenster um `prev`; `None` bei Abbruch. Fällt der Wert aus dem
    /// Fenster, wird es auf dieser Seite verdoppelt und ab [`ASPIRATION_MAX_DELTA`] ganz geöffnet.
    fn iterate(&mut self, pos: &Chess, d: i32, prev: i32, root_hash: u64) -> Option<i32> {
        let mut delta = ASPIRATION_DELTA;
        let narrow =
            self.opts.on(Feature::Aspiration) && d >= ASPIRATION_MIN_DEPTH && prev.abs() < MATE_BOUND;
        let mut w = if narrow {
            Window::around(prev, delta)
        } else {
            Window::FULL
        };
        loop {
            let score = self.negamax(pos, d, w, Node::root(root_hash));
            if self.aborted {
                return None;
            }
            if score <= w.alpha && w.alpha > -INF {
                w.alpha = (w.alpha - delta).max(-INF);
            } else if score >= w.beta && w.beta < INF {
                w.beta = (w.beta + delta).min(INF);
            } else {
                return Some(score);
            }
            delta *= 2;
            if delta > ASPIRATION_MAX_DELTA {
                w = Window::FULL;
            }
        }
    }

    /// Iterative Vertiefung. `game` = Schlüssel aller früheren Stellungen der Partie (ohne die
    /// aktuelle), für Wiederholungen. `on_depth` wird nach jeder fertigen Iteration aufgerufen.
    pub fn search(
        &mut self,
        pos: &Chess,
        game: &[u64],
        lim: Limits,
        mut on_depth: impl FnMut(&Info),
    ) -> (Option<Move>, Info) {
        self.start_search(&lim);
        let root_hash = hash(pos);
        self.path.clear();
        self.path.extend_from_slice(game);
        self.path.push(root_hash);

        let Some((fallback, fallback_score)) = self.one_ply_best(pos) else {
            return (None, Info::default());
        };
        let mut best = fallback;
        let mut info = Info {
            score: fallback_score,
            pv: vec![fallback],
            ..Info::default()
        };
        let mut prev = 0;
        for d in 1..=lim.depth.unwrap_or(MAX_DEPTH).min(MAX_DEPTH) {
            self.root_depth = d;
            self.root_best = None;
            let Some(score) = self.iterate(pos, d, prev, root_hash) else {
                self.take_partial_depth_one(d, &mut best, &mut info);
                break;
            };
            if let Some((m, _)) = self.root_best {
                best = m;
            }
            prev = score;
            info = self.iteration_info(pos, d, score, best);
            on_depth(&info);
            if Self::mate_found(score, d) || Self::time_for_next_depth_used(&lim, self.start) {
                break;
            }
        }
        info.nodes = self.nodes;
        info.evals = self.evals;
        info.time = self.start.elapsed();
        (Some(best), info)
    }

    /// Abbruch in Tiefe 1: Der zuerst durchsuchte Zug ist der Netzbeste und ersetzt den Rückfall.
    fn take_partial_depth_one(&self, d: i32, best: &mut Move, info: &mut Info) {
        if d != 1 {
            return;
        }
        if let Some((m, s)) = self.root_best {
            *best = m;
            info.score = s;
            info.pv = vec![m];
        }
    }

    fn iteration_info(&self, pos: &Chess, d: i32, score: i32, best: Move) -> Info {
        Info {
            depth: d,
            seldepth: self.seldepth,
            score,
            nodes: self.nodes,
            evals: self.evals,
            time: self.start.elapsed(),
            pv: self.pv(pos, best, usize::try_from(d).unwrap_or(0) + 4),
            hashfull: self.tt.hashfull(),
        }
    }

    /// Matt gefunden, das innerhalb der Iterationstiefe liegt: tiefer suchen ändert nichts.
    fn mate_found(score: i32, d: i32) -> bool {
        score.abs() >= MATE_BOUND && d >= MATE - score.abs()
    }

    /// Mehr als die halbe Bedenkzeit ist verbraucht: Die nächste Tiefe würde nicht fertig.
    fn time_for_next_depth_used(lim: &Limits, start: Instant) -> bool {
        lim.movetime
            .is_some_and(|t| !lim.infinite && start.elapsed() > t / 2)
    }
}
