//! Schalter für die Suchtechniken, Grenzen einer Suche und ihr Ergebnis.

use shakmaty::Move;
use std::time::Duration;

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
            .find(|(_, feature_name)| feature_name.eq_ignore_ascii_case(name))
            .map(|&(feature, _)| feature)
    }
}

/// Ein-/Ausschalter je [`Feature`]; Standard: alle an außer [`Feature::QChecks`].
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
    /// Ist die Technik eingeschaltet?
    #[must_use]
    pub fn on(&self, feature: Feature) -> bool {
        self.0[feature as usize]
    }

    /// Schaltet die Technik ein oder aus.
    pub fn set(&mut self, feature: Feature, value: bool) {
        self.0[feature as usize] = value;
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
