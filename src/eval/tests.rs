//! Bit-identical evaluation: [`Net::score_from`] and [`Net::update`] against a simple reference
//! (dense scan, rows in stored order), compared with `f32::to_bits`.

use super::{feature, Net, N_INPUTS};
use crate::board::{Color, Move, Piece, Position, Square};

fn net() -> Net {
    Net::load("weights/weights.bin").expect("weights file")
}

/// Reference for `add_row`: the same summands in the same order.
fn add_row_reference(net: &Net, src: usize, scale: f32, dest: &mut [f32], offset: usize) {
    for (target, weight) in net.row(src) {
        dest[target - offset] += weight * scale;
    }
}

/// Reference for `score_from`: checks every neuron individually, active means `> 0.0`.
fn score_reference(net: &Net, acc: &[f32]) -> f32 {
    let (n1, n2, n3) = (net.n1, net.n2, net.n3);
    let mut work = acc[n1..].to_vec();
    for (neuron, &act) in acc[..n1].iter().enumerate() {
        if act > 0.0 {
            add_row_reference(net, N_INPUTS + neuron, act, &mut work, n1);
        }
    }
    let (h2, rest) = work.split_at_mut(n2);
    for (neuron, &act) in h2.iter().enumerate() {
        if act > 0.0 {
            add_row_reference(net, N_INPUTS + n1 + neuron, act, rest, n1 + n2);
        }
    }
    let (h3, out) = rest.split_at_mut(n3);
    for (neuron, &act) in h3.iter().enumerate() {
        if act > 0.0 {
            add_row_reference(net, N_INPUTS + n1 + n2 + neuron, act, out, n1 + n2 + n3);
        }
    }
    out[0]
}

/// Simple deterministic random numbers (xorshift64).
struct Rng(u64);

impl Rng {
    fn below(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(limit).expect("small")).expect("small")
    }
}

/// A position with the accumulators of both perspectives (index 0 = White).
struct Node {
    position: Position,
    acc: [Vec<f32>; 2],
}

fn child(net: &Net, node: &Node, mv: Move, buf: &mut Vec<(usize, f32)>, reference: bool) -> Node {
    let mut position = node.position.clone();
    position.play_unchecked(mv);
    let mut acc = node.acc.clone();
    for (persp, slot) in Color::ALL.into_iter().zip(&mut acc) {
        if reference {
            if !net.update(
                node.position.board(),
                position.board(),
                persp,
                &mut slot.clone(),
                buf,
            ) {
                net.refresh(position.board(), persp, slot);
                continue;
            }
            for &(input, sign) in buf.iter() {
                add_row_reference(net, input, sign, slot, 0);
            }
        } else if !net.update(node.position.board(), position.board(), persp, slot, buf) {
            net.refresh(position.board(), persp, slot);
        }
    }
    Node { position, acc }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Long make/unmake sequences: random moves, with take-backs (stack) as in the search.
#[test]
fn score_and_update_bit_identical_to_reference() {
    let net = net();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf = Vec::new();
    let mut work = vec![0.0; net.n_work()];
    let mut checked = 0;
    for _game in 0..30 {
        let start = Position::default();
        let mut acc = [vec![0.0; net.n_targets()], vec![0.0; net.n_targets()]];
        for (persp, slot) in Color::ALL.into_iter().zip(&mut acc) {
            net.refresh(start.board(), persp, slot);
        }
        let mut stack = vec![Node { position: start, acc }];
        for _step in 0..400 {
            let node = stack.last().expect("never empty");
            let moves = node.position.legal_moves();
            if moves.is_empty() || stack.len() > 60 || (stack.len() > 1 && rng.below(4) == 0) {
                stack.pop();
                if stack.is_empty() {
                    break;
                }
                continue;
            }
            let mv = moves[rng.below(moves.len())];
            let fast = child(&net, node, mv, &mut buf, false);
            let slow = child(&net, node, mv, &mut buf, true);
            assert_eq!(bits(&fast.acc[0]), bits(&slow.acc[0]), "accumulator White");
            assert_eq!(bits(&fast.acc[1]), bits(&slow.acc[1]), "accumulator Black");
            let side = usize::from(fast.position.turn() == Color::Black);
            let score = net.score_from(&fast.acc[side], &mut work);
            assert_eq!(score.to_bits(), score_reference(&net, &fast.acc[side]).to_bits());
            checked += 1;
            stack.push(fast);
        }
    }
    assert!(checked > 5_000, "only {checked} positions checked");
}

#[test]
fn feature_mirrors_for_black() {
    let (color, role) = (Color::White, Piece::Rook);
    assert_eq!(feature(Color::White, color, role, Square::D1), 3 * 64 + 3);
    assert_eq!(feature(Color::Black, color, role, Square::D1), (6 + 3) * 64 + 59);
}
