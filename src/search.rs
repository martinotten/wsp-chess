//! Alpha-Beta-Suche mit dem Handnetz als Bewertung.
//!
//! Werte in Centipawn (`100 · s`, `s` = Netzausgabe in Bauern), Sicht des Spielers am Zug.
//! Matt = ±(`MATE` − ply); in der Transpositionstabelle auf die Knotenhöhe normalisiert.
//! Jede Suchtechnik ist über [`Feature`] abschaltbar (UCI-Optionen), damit sie einzeln gemessen
//! werden kann.

use crate::eval::Net;
use crate::tt::{Bound, Tt};
use shakmaty::uci::UciMove;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Board, CastlingMode, Chess, Color, EnPassantMode, File, Move, Position, Role, Square};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Wert für „matt in 0 Halbzügen“.
pub const MATE: i32 = 30_000;
/// Werte ab hier (Betrag) sind Mattwerte.
pub const MATE_BOUND: i32 = 29_000;
/// Größer als jeder Wert.
pub const INF: i32 = 32_000;
/// Größte Suchhöhe.
pub const MAX_PLY: usize = 128;
const EVAL_CLAMP: f32 = 20_000.0;
/// Höchstzahl geänderter Figuren, bis zu der der Akkumulator nachgeführt statt neu berechnet wird.
const MAX_DIFF: usize = 12;

/// Abschaltbare Suchtechniken.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    /// Principal Variation Search (Null-Fenster nach dem ersten Zug).
    Pvs,
    /// Suchfenster um den Wert der vorigen Iteration.
    Aspiration,
    /// Nullzug-Pruning.
    NullMove,
    /// Late Move Reductions.
    Lmr,
    /// Futility-Pruning in Tiefe 1–2.
    Futility,
    /// Schläge mit negativem Abtausch in der Ruhesuche überspringen.
    SeePrune,
    /// Delta-Pruning in der Ruhesuche.
    Delta,
    /// Schachgebote um einen Halbzug verlängern.
    CheckExt,
    /// Killerzüge in der Zugsortierung.
    Killers,
    /// History-Heuristik in der Zugsortierung.
    History,
    /// Ruhige Schachgebote in der ersten Ebene der Ruhesuche (Standard: aus).
    QChecks,
}

impl Feature {
    /// Alle Techniken mit ihrem UCI-Namen.
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

    /// Technik zum UCI-Namen (ohne Beachtung der Groß-/Kleinschreibung).
    #[must_use]
    pub fn by_name(name: &str) -> Option<Feature> {
        Feature::ALL
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .map(|&(f, _)| f)
    }
}

/// Ein-/Ausschalter je [`Feature`]; Standard: alle an außer [`Feature::QChecks`].
#[derive(Clone, Copy, Debug)]
pub struct Opts([bool; Feature::ALL.len()]);

impl Default for Opts {
    fn default() -> Self {
        let mut o = Opts([true; Feature::ALL.len()]);
        o.set(Feature::QChecks, false);
        o
    }
}

impl Opts {
    /// Ist die Technik eingeschaltet?
    #[must_use]
    pub fn on(&self, f: Feature) -> bool {
        self.0[f as usize]
    }

    /// Schaltet die Technik ein oder aus.
    pub fn set(&mut self, f: Feature, value: bool) {
        self.0[f as usize] = value;
    }
}

/// Grenzen einer Suche.
#[derive(Clone, Copy, Default, Debug)]
pub struct Limits {
    /// Bedenkzeit.
    pub movetime: Option<Duration>,
    /// Höchste Iterationstiefe.
    pub depth: Option<i32>,
    /// Höchstzahl Knoten.
    pub nodes: Option<u64>,
    /// Bis zum Abbruch von außen suchen.
    pub infinite: bool,
}

/// Ergebnis einer abgeschlossenen Iteration.
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// Iterationstiefe.
    pub depth: i32,
    /// Größte erreichte Höhe (mit Ruhesuche).
    pub seldepth: usize,
    /// Wert in cp bzw. Mattwert.
    pub score: i32,
    /// Besuchte Knoten.
    pub nodes: u64,
    /// Netzbewertungen.
    pub evals: u64,
    /// Bisherige Suchzeit.
    pub time: Duration,
    /// Hauptvariante.
    pub pv: Vec<Move>,
    /// Belegung der Transpositionstabelle in Promille.
    pub hashfull: usize,
}

/// Materialwert in cp (für Sortierung, Abtausch und Delta-Pruning).
#[must_use]
pub fn value(r: Role) -> i32 {
    match r {
        Role::Pawn => 100,
        Role::Knight | Role::Bishop => 300,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20_000,
    }
}

/// Zobrist-Schlüssel der Stellung.
#[must_use]
pub fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Zug als 16 Bit: von | nach << 6 | Umwandlung << 12 (Rochade: Zielfeld des Königs). 0 = kein Zug.
#[must_use]
pub fn enc(m: Move) -> u16 {
    let (from, to) = match m {
        Move::Castle { king, rook } => {
            let f = if rook.file() > king.file() {
                File::G
            } else {
                File::C
            };
            (king, Square::from_coords(f, king.rank()))
        }
        _ => (m.from().unwrap_or(m.to()), m.to()),
    };
    let promo = m.promotion().map_or(0, u16::from);
    u16::from(from) | u16::from(to) << 6 | promo << 12
}

/// Zug in UCI-Schreibweise (Rochade als Königszug).
#[must_use]
pub fn uci(m: Move) -> String {
    UciMove::from_move(m, CastlingMode::Standard).to_string()
}

fn ply_i32(ply: usize) -> i32 {
    i32::try_from(ply).unwrap_or(i32::MAX)
}

#[expect(clippy::cast_possible_truncation, reason = "Wert vorher auf ±20000 begrenzt")]
fn to_cp(s: f32) -> i32 {
    (s * 100.0).round().clamp(-EVAL_CLAMP, EVAL_CLAMP) as i32
}

fn non_pawn_material(pos: &Chess) -> bool {
    let b = pos.board();
    (b.by_color(pos.turn()) & !(b.pawns() | b.kings())).any()
}

fn victim_value(m: Move) -> i32 {
    if m.is_en_passant() {
        100
    } else {
        m.capture().map_or(0, value)
    }
}

fn gives_check(pos: &Chess, m: Move) -> bool {
    let mut child = pos.clone();
    child.play_unchecked(m);
    child.is_check()
}

/// Ruhiger Zug: steht die Figur auf dem Zielfeld ungedeckt angegriffen, gilt sie als verloren (−Wert).
fn see_quiet(pos: &Chess, m: Move) -> i32 {
    let b = pos.board();
    let Some(from) = m.from() else { return 0 };
    let occ = b.occupied().without(from);
    let them = !pos.turn();
    let attacked = (b.attacks_to(m.to(), them, occ) & b.by_color(them)).any();
    let defended = (b.attacks_to(m.to(), pos.turn(), occ) & b.by_color(pos.turn()) & occ).any();
    if attacked && !defended {
        -value(m.role())
    } else {
        0
    }
}

/// Statischer Abtausch auf dem Zielfeld (ohne Fesselungen) aus Sicht des Ziehenden, in cp.
#[must_use]
pub fn see(pos: &Chess, m: Move) -> i32 {
    let (Some(from), victim) = (m.from(), victim_value(m)) else {
        return 0;
    };
    if victim == 0 {
        return 0;
    }
    let b = pos.board();
    let target = m.to();
    let mut gain = [0i32; 32];
    gain[0] = victim;
    let mut attacker = m.role();
    let mut occ = b.occupied().without(from);
    if m.is_en_passant() {
        occ = occ.without(Square::from_coords(target.file(), from.rank()));
    }
    let mut side = !pos.turn();
    let mut d = 0;
    loop {
        let atk = b.attacks_to(target, side, occ) & occ & b.by_color(side);
        let next = Role::ALL
            .iter()
            .find_map(|&r| (atk & b.by_role(r)).first().map(|sq| (r, sq)));
        let Some((r, sq)) = next else { break };
        if d + 1 >= gain.len() {
            break;
        }
        d += 1;
        gain[d] = value(attacker) - gain[d - 1];
        if attacker == Role::King {
            // Der König hat geschlagen und wird nun selbst geschlagen: dieser Schlag war unzulässig
            d -= 1;
            break;
        }
        attacker = r;
        occ = occ.without(sq);
        side = !side;
    }
    while d > 0 {
        gain[d - 1] = -(-gain[d - 1]).max(gain[d]);
        d -= 1;
    }
    gain[0]
}

/// Akkumulator je Suchhöhe für die inkrementelle Bewertung.
struct Slot {
    board: Option<Board>,
    persp: Color,
    acc: Vec<f32>,
}

/// Suchfenster.
#[derive(Clone, Copy)]
struct Window {
    alpha: i32,
    beta: i32,
}

impl Window {
    fn flip(self) -> Window {
        Window {
            alpha: -self.beta,
            beta: -self.alpha,
        }
    }
    fn null(alpha: i32) -> Window {
        Window {
            alpha,
            beta: alpha + 1,
        }
    }
}

/// Zustand der Suche (Tabellen bleiben zwischen Zügen erhalten).
pub struct Searcher {
    net: Arc<Net>,
    tt: Tt,
    /// Eingeschaltete Techniken.
    pub opts: Opts,
    stop: Arc<AtomicBool>,
    nodes: u64,
    evals: u64,
    start: Instant,
    deadline: Option<Instant>,
    max_nodes: Option<u64>,
    aborted: bool,
    seldepth: usize,
    root_depth: i32,
    killers: [[u16; 2]; MAX_PLY],
    history: Vec<i32>,
    path: Vec<u64>,
    root_best: Option<(Move, i32)>,
    slots: Vec<Slot>,
    work: Vec<f32>,
    diff: Vec<(usize, f32)>,
    /// Akkumulator nachführen statt neu berechnen.
    pub incremental: bool,
    /// Test: jede Bewertung mit Neuberechnung vergleichen und die größte Abweichung (Bauern) merken.
    pub check_incremental: Option<f32>,
}

fn history_idx(c: Color, e: u16) -> usize {
    usize::from(c == Color::Black) * 4096 + usize::from(e & 0xfff)
}

impl Searcher {
    /// Neuer Sucher mit Transpositionstabelle von `hash_mb` MB; `stop` bricht eine Suche von außen ab.
    #[must_use]
    pub fn new(net: Arc<Net>, hash_mb: usize, stop: Arc<AtomicBool>) -> Searcher {
        let nt = net.n_targets();
        let nw = net.n_work();
        Searcher {
            net,
            tt: Tt::new(hash_mb),
            opts: Opts::default(),
            stop,
            nodes: 0,
            evals: 0,
            start: Instant::now(),
            deadline: None,
            max_nodes: None,
            aborted: false,
            seldepth: 0,
            root_depth: 0,
            killers: [[0; 2]; MAX_PLY],
            history: vec![0; 2 * 4096],
            path: Vec::with_capacity(1024),
            root_best: None,
            slots: (0..MAX_PLY + 2)
                .map(|_| Slot {
                    board: None,
                    persp: Color::White,
                    acc: vec![0.0; nt],
                })
                .collect(),
            work: vec![0.0; nw],
            diff: Vec::with_capacity(32),
            incremental: true,
            check_incremental: None,
        }
    }

    /// Das Netz der Bewertung.
    #[must_use]
    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Neue Partie: Tabellen leeren.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.killers = [[0; 2]; MAX_PLY];
        self.history.fill(0);
    }

    /// Statische Bewertung in cp, Sicht des Spielers am Zug. Der Akkumulator wird vom zuletzt auf
    /// dieser Höhe bewerteten Brett (meist ein Geschwisterknoten) oder vom zwei Halbzüge höheren
    /// (gleiche Perspektive) nachgeführt und nur bei großem Unterschied neu berechnet.
    pub fn eval(&mut self, pos: &Chess, ply: usize) -> i32 {
        self.evals += 1;
        let c = pos.turn();
        let ply = ply.min(MAX_PLY + 1);
        let net = &self.net;
        let (lo, hi) = self.slots.split_at_mut(ply);
        let slot = &mut hi[0];
        let mut done = false;
        if self.incremental {
            if let (Some(b), true) = (&slot.board, slot.persp == c) {
                done = net.update(b, pos.board(), c, &mut slot.acc, MAX_DIFF, &mut self.diff);
            }
            if !done && ply >= 2 {
                let g = &lo[ply - 2];
                if let (Some(b), true) = (&g.board, g.persp == c) {
                    slot.acc.copy_from_slice(&g.acc);
                    done = net.update(b, pos.board(), c, &mut slot.acc, MAX_DIFF, &mut self.diff);
                }
            }
        }
        if !done {
            net.refresh(pos.board(), c, &mut slot.acc);
        }
        slot.board = Some(pos.board().clone());
        slot.persp = c;
        let v = net.score_from(&slot.acc, &mut self.work);
        if let Some(worst) = self.check_incremental.as_mut() {
            *worst = worst.max((v - net.score(pos)).abs());
        }
        to_cp(v)
    }

    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes.trailing_zeros() >= 10
            && (self.stop.load(Ordering::Relaxed)
                || self.deadline.is_some_and(|d| Instant::now() >= d)
                || self.max_nodes.is_some_and(|n| self.nodes >= n))
        {
            self.aborted = true;
        }
    }

    /// Gleiche Stellung früher im Pfad (gleiche Seite am Zug), höchstens bis zum letzten
    /// irreversiblen Zug; 0 im Pfad markiert einen Nullzug (Grenze).
    fn is_repetition(&self, h: u64, halfmoves: u32) -> bool {
        let window = usize::try_from(halfmoves).unwrap_or(usize::MAX);
        self.path
            .iter()
            .rev()
            .skip(2)
            .step_by(2)
            .take(window / 2)
            .take_while(|&&x| x != 0)
            .any(|&x| x == h)
    }

    fn to_tt(s: i32, ply: usize) -> i32 {
        let p = ply_i32(ply);
        if s >= MATE_BOUND {
            s + p
        } else if s <= -MATE_BOUND {
            s - p
        } else {
            s
        }
    }

    fn from_tt(s: i32, ply: usize) -> i32 {
        let p = ply_i32(ply);
        if s >= MATE_BOUND {
            s - p
        } else if s <= -MATE_BOUND {
            s + p
        } else {
            s
        }
    }

    fn order_score(&self, turn: Color, m: Move, tt_move: u16, ply: usize) -> i32 {
        let e = enc(m);
        if e == tt_move {
            return 10_000_000;
        }
        if m.is_capture() || m.is_promotion() {
            let promo = m.promotion().map_or(0, value);
            return 1_000_000 + 10 * (victim_value(m) + promo) - value(m.role()) / 100;
        }
        if self.opts.on(Feature::Killers) {
            if self.killers[ply][0] == e {
                return 900_000;
            }
            if self.killers[ply][1] == e {
                return 800_000;
            }
        }
        if self.opts.on(Feature::History) {
            return self.history[history_idx(turn, e)];
        }
        0
    }

    fn sorted(
        &self,
        pos: &Chess,
        moves: impl IntoIterator<Item = Move>,
        tt_move: u16,
        ply: usize,
    ) -> Vec<Move> {
        let turn = pos.turn();
        let mut v: Vec<(i32, Move)> = moves
            .into_iter()
            .map(|m| (self.order_score(turn, m, tt_move, ply), m))
            .collect();
        v.sort_unstable_by_key(|&(score, _)| std::cmp::Reverse(score));
        v.into_iter().map(|(_, m)| m).collect()
    }

    /// Ruhesuche; `qdepth` = Höhe innerhalb der Ruhesuche (0 = erste Ebene).
    fn qsearch(&mut self, pos: &Chess, mut w: Window, ply: usize, qdepth: u32) -> i32 {
        self.tick();
        if self.aborted {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        if pos.is_insufficient_material() {
            return 0;
        }
        let in_check = pos.is_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { self.eval(pos, ply) };
        }
        let mut moves = if in_check {
            pos.legal_moves()
        } else {
            pos.capture_moves()
        };
        let stand;
        let mut best;
        if in_check {
            if moves.is_empty() {
                return -(MATE - ply_i32(ply));
            }
            stand = -INF;
            best = -INF;
        } else {
            moves.extend(pos.promotion_moves().into_iter().filter(|m| !m.is_capture()));
            if qdepth == 0 && self.opts.on(Feature::QChecks) {
                moves.extend(pos.legal_moves().into_iter().filter(|&m| {
                    !m.is_capture() && !m.is_promotion() && gives_check(pos, m) && see_quiet(pos, m) >= 0
                }));
            }
            stand = self.eval(pos, ply);
            if stand >= w.beta {
                return stand;
            }
            w.alpha = w.alpha.max(stand);
            best = stand;
        }
        for m in self.sorted(pos, moves, 0, ply) {
            if !in_check && !m.is_promotion() && m.is_capture() {
                if self.opts.on(Feature::Delta) && stand + victim_value(m) + 200 < w.alpha {
                    continue;
                }
                if self.opts.on(Feature::SeePrune) && see(pos, m) < 0 {
                    continue;
                }
            }
            let mut child = pos.clone();
            child.play_unchecked(m);
            let score = -self.qsearch(&child, w.flip(), ply + 1, qdepth + 1);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > w.alpha {
                    w.alpha = score;
                    if w.alpha >= w.beta {
                        break;
                    }
                }
            }
        }
        best
    }

    /// Nullzug: Darf der Gegner zweimal ziehen und reicht es trotzdem für beta, gilt die Stellung als
    /// zu gut. Gibt den Schnittwert zurück oder `None`.
    fn null_move(&mut self, pos: &Chess, depth: i32, beta: i32, ply: usize) -> Option<i32> {
        let child = pos.clone().swap_turn().ok()?;
        let r = 2 + i32::from(depth >= 7);
        self.path.push(0);
        let h = hash(&child);
        let s = -self.negamax(
            &child,
            depth - 1 - r,
            Window::null(beta - 1).flip(),
            ply + 1,
            false,
            h,
        );
        self.path.pop();
        if self.aborted || s < beta {
            return None;
        }
        Some(if s >= MATE_BOUND { beta } else { s })
    }

    fn reward_quiet(&mut self, turn: Color, e: u16, depth: i32, ply: usize) {
        if self.opts.on(Feature::Killers) && self.killers[ply][0] != e {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = e;
        }
        if self.opts.on(Feature::History) {
            let v = &mut self.history[history_idx(turn, e)];
            *v = (*v + depth * depth).min(400_000);
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "eine Suchfunktion, Schritte sind kommentiert"
    )]
    fn negamax(
        &mut self,
        pos: &Chess,
        mut depth: i32,
        mut win: Window,
        ply: usize,
        can_null: bool,
        key: u64,
    ) -> i32 {
        self.tick();
        if self.aborted {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        if ply > 0 && (pos.is_insufficient_material() || self.is_repetition(key, pos.halfmoves())) {
            return 0;
        }
        let in_check = pos.is_check();
        let moves = pos.legal_moves();
        if moves.is_empty() {
            return if in_check { -(MATE - ply_i32(ply)) } else { 0 };
        }
        if ply > 0 && pos.halfmoves() >= 100 {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.eval(pos, ply);
        }
        if in_check && self.opts.on(Feature::CheckExt) && ply_i32(ply) < 2 * self.root_depth {
            depth += 1;
        }
        if depth <= 0 {
            return self.qsearch(pos, win, ply, 0);
        }
        let pv_node = win.beta - win.alpha > 1;
        let alpha0 = win.alpha;

        // Transpositionstabelle
        let mut tt_move = 0u16;
        let mut tt_eval = None;
        if let Some(entry) = self.tt.probe(key) {
            tt_move = entry.mv;
            tt_eval = entry.eval;
            if ply > 0 && entry.depth >= depth && !(pv_node && self.opts.on(Feature::Pvs)) {
                let sc = Self::from_tt(entry.score, ply);
                let cut = match entry.bound {
                    Bound::Exact => true,
                    Bound::Lower => sc >= win.beta,
                    Bound::Upper => sc <= win.alpha,
                    Bound::None => false,
                };
                if cut {
                    return sc;
                }
            }
        }
        let static_eval = if in_check {
            None
        } else {
            Some(tt_eval.unwrap_or_else(|| self.eval(pos, ply)))
        };

        // Nullzug
        if let Some(se) = static_eval {
            if self.opts.on(Feature::NullMove)
                && !pv_node
                && can_null
                && depth >= 3
                && ply > 0
                && se >= win.beta
                && non_pawn_material(pos)
            {
                if let Some(sc) = self.null_move(pos, depth, win.beta, ply) {
                    return sc;
                }
                if self.aborted {
                    return 0;
                }
            }
        }

        // Futility: ruhige Züge können alpha nicht mehr erreichen
        let futile = static_eval.is_some_and(|se| {
            self.opts.on(Feature::Futility)
                && !pv_node
                && depth <= 2
                && win.alpha.abs() < MATE_BOUND
                && se + 150 * depth <= win.alpha
        });

        let turn = pos.turn();
        let mut best = -INF;
        let mut best_move = 0u16;
        for (i, m) in self.sorted(pos, moves, tt_move, ply).into_iter().enumerate() {
            let code = enc(m);
            let quiet = !m.is_capture() && !m.is_promotion();
            let mut child = pos.clone();
            child.play_unchecked(m);
            let gives_check = child.is_check();
            if futile && i > 0 && quiet && !gives_check {
                continue;
            }
            let ch = hash(&child);
            self.path.push(ch);
            let nd = depth - 1;
            let score = if i == 0 {
                -self.negamax(&child, nd, win.flip(), ply + 1, true, ch)
            } else {
                let is_killer = self.opts.on(Feature::Killers) && self.killers[ply].contains(&code);
                let reduce = self.opts.on(Feature::Lmr)
                    && depth >= 3
                    && i >= 3
                    && quiet
                    && !in_check
                    && !gives_check
                    && !is_killer;
                let r = if reduce {
                    1 + i32::from(i >= 6 && depth >= 6)
                } else {
                    0
                };
                let probe = if self.opts.on(Feature::Pvs) {
                    Window::null(win.alpha)
                } else {
                    win
                };
                let mut sc = -self.negamax(&child, nd - r, probe.flip(), ply + 1, true, ch);
                if r > 0 && sc > win.alpha && !self.aborted {
                    sc = -self.negamax(&child, nd, probe.flip(), ply + 1, true, ch);
                }
                if self.opts.on(Feature::Pvs) && sc > win.alpha && sc < win.beta && !self.aborted {
                    sc = -self.negamax(&child, nd, win.flip(), ply + 1, true, ch);
                }
                sc
            };
            self.path.pop();
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                best_move = code;
                if ply == 0 {
                    self.root_best = Some((m, score));
                }
                if score > win.alpha {
                    win.alpha = score;
                    if win.alpha >= win.beta {
                        if quiet {
                            self.reward_quiet(turn, code, depth, ply);
                        }
                        break;
                    }
                }
            }
        }
        if best == -INF {
            // alle Züge durch Futility übersprungen
            return static_eval.unwrap_or(alpha0).max(alpha0);
        }
        let bound = if best >= win.beta {
            Bound::Lower
        } else if best > alpha0 {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt
            .store(key, best_move, Self::to_tt(best, ply), static_eval, depth, bound);
        best
    }

    fn find(pos: &Chess, e: u16) -> Option<Move> {
        if e == 0 {
            return None;
        }
        pos.legal_moves().into_iter().find(|&m| enc(m) == e)
    }

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
            *v /= 8;
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

    /// Eine Iteration mit Aspiration-Fenster um `prev`; `None` bei Abbruch.
    fn iterate(&mut self, pos: &Chess, d: i32, prev: i32, root_hash: u64) -> Option<i32> {
        let mut delta = 40;
        let mut w = if self.opts.on(Feature::Aspiration) && d >= 4 && prev.abs() < MATE_BOUND {
            Window {
                alpha: prev - delta,
                beta: prev + delta,
            }
        } else {
            Window {
                alpha: -INF,
                beta: INF,
            }
        };
        loop {
            let score = self.negamax(pos, d, w, 0, false, root_hash);
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
            if delta > 1000 {
                w = Window {
                    alpha: -INF,
                    beta: INF,
                };
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

        let Some((fallback, fs)) = self.one_ply_best(pos) else {
            return (None, Info::default());
        };
        let mut best = fallback;
        let mut info = Info {
            score: fs,
            pv: vec![fallback],
            ..Info::default()
        };
        let max_depth = lim.depth.unwrap_or(100).min(100);
        let mut prev = 0;
        for d in 1..=max_depth {
            self.root_depth = d;
            self.root_best = None;
            let Some(score) = self.iterate(pos, d, prev, root_hash) else {
                if d == 1 {
                    // unvollständige Tiefe 1: der zuerst durchsuchte Zug ist der Netzbeste
                    if let Some((m, s)) = self.root_best {
                        best = m;
                        info.score = s;
                        info.pv = vec![m];
                    }
                }
                break;
            };
            if let Some((m, _)) = self.root_best {
                best = m;
            }
            prev = score;
            info = Info {
                depth: d,
                seldepth: self.seldepth,
                score,
                nodes: self.nodes,
                evals: self.evals,
                time: self.start.elapsed(),
                pv: self.pv(pos, best, usize::try_from(d).unwrap_or(0) + 4),
                hashfull: self.tt.hashfull(),
            };
            on_depth(&info);
            if score.abs() >= MATE_BOUND && d >= MATE - score.abs() {
                break;
            }
            if lim
                .movetime
                .is_some_and(|t| !lim.infinite && self.start.elapsed() > t / 2)
            {
                break;
            }
        }
        info.nodes = self.nodes;
        info.evals = self.evals;
        info.time = self.start.elapsed();
        (Some(best), info)
    }
}
