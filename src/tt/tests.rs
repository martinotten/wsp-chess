//! Packung, Ersetzungsregel und gleichzeitige Zugriffe der Transpositionstabelle.

use super::{Bound, Entry, Tt, DEPTH_MAX};
use std::sync::Arc;
use std::thread;

fn entry(key: u64, mv: u16, score: i32, depth: i32) -> Entry {
    Entry {
        key,
        mv,
        score,
        eval: Some(-score / 2),
        depth,
        bound: Bound::Exact,
        age: 0,
    }
}

#[test]
fn pack_roundtrip_at_the_limits() {
    let mate = 30_000 + 128;
    for (mv, score, eval, depth, bound, age) in [
        (0x7FFF, mate, Some(20_000), DEPTH_MAX, Bound::Upper, 255),
        (1, -mate, Some(-20_000), 1, Bound::Lower, 0),
        (0, 0, None, 101, Bound::Exact, 17),
        (0x1234, -1, Some(0), 64, Bound::Exact, 128),
    ] {
        let original = Entry {
            key: 0xDEAD_BEEF_0123_4567,
            mv,
            score,
            eval,
            depth,
            bound,
            age,
        };
        assert_eq!(Entry::unpack(original.key, original.pack()), original);
    }
}

#[test]
fn mate_scores_roundtrip_through_the_table_at_any_height() {
    let table = Tt::new(1);
    for (key, score) in [(11_u64, 29_990 + 100), (12, -(29_990 + 100)), (13, 30_000 + 128)] {
        table.store(entry(key, 7, score, 5), 3);
        assert_eq!(table.probe(key).map(|found| found.score), Some(score));
    }
}

#[test]
fn replacement_rule_and_kept_move() {
    let table = Tt::new(1);
    let key = 42;
    table.store(entry(key, 99, 10, 8), 1);
    // gleiche Stellung ohne Zug: alter Zug bleibt, Rest wird ersetzt
    table.store(
        Entry {
            mv: 0,
            ..entry(key, 0, 20, 3)
        },
        1,
    );
    let found = table.probe(key).expect("Eintrag");
    assert_eq!((found.mv, found.score, found.depth), (99, 20, 3));
    // andere Stellung auf demselben Platz, flacher und gleiche Generation: nicht ersetzt
    let other = key + (table.slots.len() as u64);
    table.store(entry(other, 5, 1, 2), 1);
    assert!(table.probe(other).is_none());
    assert!(table.probe(key).is_some());
    // neue Generation: ersetzt
    table.store(entry(other, 5, 1, 2), 2);
    assert!(table.probe(other).is_some());
    assert!(table.probe(key).is_none());
    assert_eq!(table.hashfull(2), 1);
    table.clear();
    assert!(table.probe(other).is_none());
    assert_eq!(table.hashfull(2), 0);
}

/// Mehrere Threads schreiben und lesen dieselben Plätze: keine Panik, und jeder gefundene Eintrag ist
/// einer, den ein Thread für genau diesen Schlüssel geschrieben hat.
#[test]
fn concurrent_writers_never_yield_foreign_entries() {
    let table = Arc::new(Tt::new(1));
    let n_slots = table.slots.len() as u64;
    let handles: Vec<_> = (0..8_u64)
        .map(|thread_id| {
            let table = Arc::clone(&table);
            thread::spawn(move || {
                let mut found = 0_u32;
                for round in 0..200_000_u64 {
                    // 16 Schlüssel je Platz, alle Threads auf denselben 64 Plätzen
                    let key = (round % 64) + n_slots * ((round + thread_id) % 16) + 1;
                    let value = i32::try_from(key % 1000).expect("klein");
                    table.store(
                        entry(
                            key,
                            u16::try_from(key % 0x7FFF).expect("klein"),
                            value,
                            1 + value % 100,
                        ),
                        1,
                    );
                    if let Some(hit) = table.probe(key) {
                        assert_eq!(hit.score, i32::try_from(hit.key % 1000).expect("klein"));
                        assert_eq!(u64::from(hit.mv), hit.key % 0x7FFF);
                        assert_eq!(hit.eval, Some(-hit.score / 2));
                        found += 1;
                    }
                }
                found
            })
        })
        .collect();
    let total: u32 = handles
        .into_iter()
        .map(|handle| handle.join().expect("kein Absturz"))
        .sum();
    assert!(total > 0);
}
