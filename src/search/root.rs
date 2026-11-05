//! Wurzel: iterative Vertiefung mit Aspiration-Fenstern, Rückfallzug und Hauptvariante.

use super::node::{Node, Window};
use super::{hash, move_code, Feature, Info, Limits, Searcher, Team, INF, MATE, MATE_BOUND, MAX_PLY};
use shakmaty::{Chess, Move, Position};

/// Größte Iterationstiefe.
const MAX_DEPTH: i32 = 100;

/// Aspiration: erste halbe Fensterbreite, ab dieser Breite volles Fenster, Mindesttiefe.
const ASPIRATION_DELTA: i32 = 40;
const ASPIRATION_MAX_DELTA: i32 = 1000;
const ASPIRATION_MIN_DEPTH: i32 = 4;

/// History: Teiler beim Start einer neuen Suche (ältere Erfahrungen verblassen).
const HISTORY_DECAY: i32 = 8;

/// Stand der iterativen Vertiefung zwischen zwei Tiefen.
struct Deepening {
    /// Bester Zug der letzten fertigen Tiefe (anfangs der Rückfallzug).
    best: Move,
    /// Bericht der letzten fertigen Tiefe; sein Wert ist die Mitte des nächsten Aspiration-Fensters.
    info: Info,
}

impl Searcher {
    /// Iterative Vertiefung. `game` = Schlüssel aller früheren Stellungen der Partie (ohne die
    /// aktuelle), für Wiederholungen. `on_depth` wird nach jeder fertigen Iteration aufgerufen.
    pub fn search(
        &mut self,
        pos: &Chess,
        game: &[u64],
        lim: Limits,
        on_depth: impl FnMut(&Info),
    ) -> (Option<Move>, Info) {
        let team = self.begin_team();
        self.search_main(pos, game, lim, &team, on_depth)
    }

    /// Neue Suche dieses Suchers als Hauptthread: Generation weiterzählen, gemeinsamer Zustand ab jetzt.
    pub(super) fn begin_team(&mut self) -> Team {
        self.age = self.age.wrapping_add(1);
        Team::new(self.age)
    }

    /// Iterative Vertiefung des Hauptthreads (liefert Zug und Bericht).
    pub(super) fn search_main(
        &mut self,
        pos: &Chess,
        game: &[u64],
        lim: Limits,
        team: &Team,
        mut on_depth: impl FnMut(&Info),
    ) -> (Option<Move>, Info) {
        self.start_search(&lim, team, false);
        let root_key = self.begin_path(pos, game);
        let Some(mut state) = self.fallback(pos) else {
            return (None, Info::default());
        };
        for depth in 1..=Self::depth_limit(&lim) {
            if !self.complete_depth(pos, depth, root_key, &mut state) {
                break;
            }
            on_depth(&state.info);
            if self.deeper_search_pointless(&lim, &state.info) {
                break;
            }
        }
        self.finish(state)
    }

    /// Iterative Vertiefung eines Helfers (Nummer `index` ≥ 1): ungerade Nummern beginnen eine Tiefe
    /// tiefer, damit nicht alle Threads dieselbe Tiefe rechnen. Ergebnis nur über die Tabelle.
    pub(super) fn search_helper(
        &mut self,
        pos: &Chess,
        game: &[u64],
        lim: Limits,
        team: &Team,
        index: usize,
    ) {
        self.start_search(&lim, team, true);
        let root_key = self.begin_path(pos, game);
        let Some(mut state) = self.fallback(pos) else {
            return;
        };
        let first = 1 + i32::from(index % 2 == 1);
        for depth in first..=Self::depth_limit(&lim) {
            if !self.complete_depth(pos, depth, root_key, &mut state)
                || Self::mate_found(state.info.score, depth)
            {
                break;
            }
        }
    }

    fn start_search(&mut self, lim: &Limits, team: &Team, helper: bool) {
        self.team = team.clone();
        self.helper = helper;
        self.age = team.age;
        self.start = team.start;
        self.deadline = lim.movetime.map(|movetime| self.start + movetime);
        self.max_nodes = lim.nodes;
        self.nodes = 0;
        self.evals = 0;
        self.aborted = false;
        self.seldepth = 0;
        self.killers = [[0; 2]; MAX_PLY];
        for score in &mut self.history {
            *score /= HISTORY_DECAY;
        }
    }

    /// Suchpfad für die Wiederholungserkennung: frühere Stellungen der Partie, dann die Wurzel.
    fn begin_path(&mut self, pos: &Chess, game: &[u64]) -> u64 {
        let root_key = hash(pos);
        self.path.clear();
        self.path.extend_from_slice(game);
        self.path.push(root_key);
        root_key
    }

    /// Startstand mit dem Rückfallzug; `None`, wenn es keinen legalen Zug gibt.
    fn fallback(&mut self, pos: &Chess) -> Option<Deepening> {
        let (best, score) = self.one_ply_best(pos)?;
        let info = Info {
            score,
            pv: vec![best],
            ..Info::default()
        };
        Some(Deepening { best, info })
    }

    /// Rückfall, falls nicht einmal Tiefe 1 fertig wird: bester Zug nach Netz (ein Halbzug). Nach dem
    /// ersten Zug ist ein legaler Zug gesichert; danach wird vor jeder Bewertung der Stopp geprüft.
    fn one_ply_best(&mut self, pos: &Chess) -> Option<(Move, i32)> {
        let mut best: Option<(Move, i32)> = None;
        for mv in pos.legal_moves() {
            if best.is_some() && self.should_stop(self.team_nodes()) {
                break;
            }
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = if child.is_checkmate() {
                MATE - 1
            } else {
                -self.eval(&child, 1)
            };
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((mv, score));
            }
        }
        best
    }

    fn depth_limit(lim: &Limits) -> i32 {
        lim.depth.unwrap_or(MAX_DEPTH).min(MAX_DEPTH)
    }

    /// Durchsucht eine Tiefe und übernimmt ihr Ergebnis; `false`, wenn die Suche abgebrochen wurde.
    fn complete_depth(&mut self, pos: &Chess, depth: i32, root_key: u64, state: &mut Deepening) -> bool {
        self.root_depth = depth;
        self.root_best = None;
        let Some(score) = self.iterate(pos, depth, state.info.score, root_key) else {
            if depth == 1 {
                self.take_partial_first_depth(state);
            }
            return false;
        };
        if let Some((mv, _)) = self.root_best {
            state.best = mv;
        }
        state.info = self.iteration_info(pos, depth, score, state.best);
        true
    }

    /// Eine Iteration mit Aspiration-Fenster um `expected` (Wert der vorigen Tiefe); `None` bei
    /// Abbruch. Fällt der Wert aus dem Fenster, wird es auf dieser Seite verdoppelt und ab
    /// [`ASPIRATION_MAX_DELTA`] ganz geöffnet.
    fn iterate(&mut self, pos: &Chess, depth: i32, expected: i32, root_key: u64) -> Option<i32> {
        let mut delta = ASPIRATION_DELTA;
        let narrow =
            self.opts.on(Feature::Aspiration) && depth >= ASPIRATION_MIN_DEPTH && expected.abs() < MATE_BOUND;
        let mut win = if narrow {
            Window::around(expected, delta)
        } else {
            Window::FULL
        };
        loop {
            let score = self.negamax(pos, depth, win, Node::root(root_key));
            if self.aborted {
                return None;
            }
            if score <= win.alpha && win.alpha > -INF {
                win.alpha = (win.alpha - delta).max(-INF);
            } else if score >= win.beta && win.beta < INF {
                win.beta = (win.beta + delta).min(INF);
            } else {
                return Some(score);
            }
            delta *= 2;
            if delta > ASPIRATION_MAX_DELTA {
                win = Window::FULL;
            }
        }
    }

    /// Abbruch in Tiefe 1: Der zuerst durchsuchte Zug ist der Netzbeste und ersetzt den Rückfall.
    fn take_partial_first_depth(&self, state: &mut Deepening) {
        if let Some((mv, score)) = self.root_best {
            state.best = mv;
            state.info.score = score;
            state.info.pv = vec![mv];
        }
    }

    fn iteration_info(&self, pos: &Chess, depth: i32, score: i32, best: Move) -> Info {
        Info {
            depth,
            seldepth: self.seldepth,
            score,
            nodes: self.team_nodes(),
            evals: self.evals,
            time: self.start.elapsed(),
            pv: self.pv(pos, best, usize::try_from(depth).unwrap_or(0) + 4),
            hashfull: self.tt.hashfull(self.age),
        }
    }

    /// Hauptvariante aus der Transpositionstabelle, höchstens `max_len` Züge, ohne Wiederholungen.
    fn pv(&self, pos: &Chess, first: Move, max_len: usize) -> Vec<Move> {
        let mut line = vec![first];
        let mut current = pos.clone();
        current.play_unchecked(first);
        let mut seen = vec![hash(pos)];
        while line.len() < max_len {
            let key = hash(&current);
            if seen.contains(&key) {
                break;
            }
            seen.push(key);
            let Some(mv) = self
                .tt
                .probe(key)
                .and_then(|entry| Self::find(&current, entry.mv))
            else {
                break;
            };
            line.push(mv);
            current.play_unchecked(mv);
        }
        line
    }

    /// Legaler Zug zur Kodierung `code` (siehe [`move_code`]); `None` für 0 oder unbekannte Züge.
    fn find(pos: &Chess, code: u16) -> Option<Move> {
        if code == 0 {
            return None;
        }
        pos.legal_moves().into_iter().find(|&mv| move_code(mv) == code)
    }

    /// Eine weitere Tiefe bringt nichts: Matt gefunden oder die Zeit reicht nicht mehr.
    fn deeper_search_pointless(&self, lim: &Limits, done: &Info) -> bool {
        Self::mate_found(done.score, done.depth) || self.time_for_next_depth_used(lim)
    }

    /// Matt gefunden, das innerhalb der Iterationstiefe liegt: tiefer suchen ändert nichts.
    fn mate_found(score: i32, depth: i32) -> bool {
        score.abs() >= MATE_BOUND && depth >= MATE - score.abs()
    }

    /// Mehr als die halbe Bedenkzeit ist verbraucht: Die nächste Tiefe würde nicht fertig.
    fn time_for_next_depth_used(&self, lim: &Limits) -> bool {
        lim.movetime
            .is_some_and(|movetime| !lim.infinite && self.start.elapsed() > movetime / 2)
    }

    /// Endbericht mit den Zählern der ganzen Suche.
    fn finish(&self, state: Deepening) -> (Option<Move>, Info) {
        let mut info = state.info;
        info.nodes = self.team_nodes();
        info.evals = self.evals;
        info.time = self.start.elapsed();
        (Some(state.best), info)
    }
}
