//! Zugsortierung: TT-Zug, Schläge nach MVV-LVA, Killerzüge, History.

use super::{is_quiet, move_code, Feature, Searcher};
use crate::board::{Color, Move, Position};
use crate::exchange::{value, victim_value};

/// Zugsortierung: Rangstufen (TT-Zug vor Schlägen vor Killern vor History).
const ORDER_TT_MOVE: i32 = 10_000_000;
const ORDER_CAPTURE: i32 = 1_000_000;
const ORDER_KILLERS: [i32; 2] = [900_000, 800_000];

/// History: Obergrenze je Eintrag.
const HISTORY_MAX: i32 = 400_000;

/// Eintrag in der History-Tabelle: je Farbe ein Block, darin Start- und Zielfeld des Zugs.
fn history_idx(turn: Color, code: u16) -> usize {
    usize::from(turn == Color::Black) * 4096 + usize::from(code & 0xfff)
}

impl Searcher {
    /// Rang eines Zugs: je größer, desto früher wird er durchsucht.
    fn order_score(&self, turn: Color, mv: Move, tt_move: u16, ply: usize) -> i32 {
        let code = move_code(mv);
        if code == tt_move {
            return ORDER_TT_MOVE;
        }
        if !is_quiet(mv) {
            // MVV-LVA: wertvolles Opfer zuerst, bei gleichem Opfer der billigere Angreifer
            let promo = mv.promotion().map_or(0, value);
            return ORDER_CAPTURE + 10 * (victim_value(mv) + promo) - value(mv.role()) / 100;
        }
        if self.opts.on(Feature::Killers) {
            if let Some(slot) = self.killers[ply].iter().position(|&killer| killer == code) {
                return ORDER_KILLERS[slot];
            }
        }
        if self.opts.on(Feature::History) {
            return self.history[history_idx(turn, code)];
        }
        0
    }

    pub(super) fn sorted(&self, pos: &Position, moves: &[Move], tt_move: u16, ply: usize) -> Vec<Move> {
        let turn = pos.turn();
        let mut ranked: Vec<(i32, Move)> = moves
            .iter()
            .map(|&mv| (self.order_score(turn, mv, tt_move, ply), mv))
            .collect();
        ranked.sort_unstable_by_key(|&(score, _)| std::cmp::Reverse(score));
        ranked.into_iter().map(|(_, mv)| mv).collect()
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
            let score = &mut self.history[history_idx(turn, code)];
            *score = (*score + depth * depth).min(HISTORY_MAX);
        }
    }
}
