//! Zugsortierung: TT-Zug, Schläge nach MVV-LVA, Killerzüge, History.

use super::{enc, is_quiet, Feature, Searcher};
use crate::exchange::{value, victim_value};
use shakmaty::{Chess, Color, Move, Position};

/// Zugsortierung: Rangstufen (TT-Zug vor Schlägen vor Killern vor History).
const ORDER_TT_MOVE: i32 = 10_000_000;
const ORDER_CAPTURE: i32 = 1_000_000;
const ORDER_KILLERS: [i32; 2] = [900_000, 800_000];

/// History: Obergrenze je Eintrag.
const HISTORY_MAX: i32 = 400_000;

fn history_idx(c: Color, e: u16) -> usize {
    usize::from(c == Color::Black) * 4096 + usize::from(e & 0xfff)
}

impl Searcher {
    fn order_score(&self, turn: Color, m: Move, tt_move: u16, ply: usize) -> i32 {
        let e = enc(m);
        if e == tt_move {
            return ORDER_TT_MOVE;
        }
        if !is_quiet(m) {
            // MVV-LVA: wertvolles Opfer zuerst, bei gleichem Opfer der billigere Angreifer
            let promo = m.promotion().map_or(0, value);
            return ORDER_CAPTURE + 10 * (victim_value(m) + promo) - value(m.role()) / 100;
        }
        if self.opts.on(Feature::Killers) {
            if let Some(k) = self.killers[ply].iter().position(|&k| k == e) {
                return ORDER_KILLERS[k];
            }
        }
        if self.opts.on(Feature::History) {
            return self.history[history_idx(turn, e)];
        }
        0
    }

    pub(super) fn sorted(&self, pos: &Chess, moves: &[Move], tt_move: u16, ply: usize) -> Vec<Move> {
        let turn = pos.turn();
        let mut v: Vec<(i32, Move)> = moves
            .iter()
            .map(|&m| (self.order_score(turn, m, tt_move, ply), m))
            .collect();
        v.sort_unstable_by_key(|&(score, _)| std::cmp::Reverse(score));
        v.into_iter().map(|(_, m)| m).collect()
    }

    pub(super) fn is_killer(&self, ply: usize, code: u16) -> bool {
        self.opts.on(Feature::Killers) && self.killers[ply].contains(&code)
    }

    /// Ruhiger Zug hat einen Beta-Schnitt erzeugt: als Killer und in der History merken.
    pub(super) fn reward_quiet(&mut self, turn: Color, code: u16, depth: i32, ply: usize) {
        if self.opts.on(Feature::Killers) && self.killers[ply][0] != code {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = code;
        }
        if self.opts.on(Feature::History) {
            let v = &mut self.history[history_idx(turn, code)];
            *v = (*v + depth * depth).min(HISTORY_MAX);
        }
    }
}
