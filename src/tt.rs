//! Transpositionstabelle: ein Eintrag je Platz, Ersetzung nach Alter und Tiefe.

/// Art der gespeicherten Schranke.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Leerer Platz.
    #[default]
    None,
    /// Genauer Wert.
    Exact,
    /// Untere Schranke (Beta-Schnitt).
    Lower,
    /// Obere Schranke (kein Zug über Alpha).
    Upper,
}

/// Ein Eintrag der Tabelle.
#[derive(Clone, Copy, Default, Debug)]
pub struct Entry {
    /// Zobrist-Schlüssel der Stellung.
    pub key: u64,
    /// Bester Zug (siehe `search::enc`), 0 = keiner.
    pub mv: u16,
    /// Suchwert, Mattwerte relativ zur Stellung normalisiert.
    pub score: i32,
    /// Statische Bewertung (cp), falls bekannt.
    pub eval: Option<i32>,
    /// Restsuchtiefe, mit der `score` berechnet wurde.
    pub depth: i32,
    /// Art der Schranke.
    pub bound: Bound,
    /// Suchgeneration; setzt [`Tt::store`].
    pub(crate) age: u8,
}

/// Transpositionstabelle fester Größe (Zweierpotenz).
pub struct Tt {
    slots: Vec<Entry>,
    age: u8,
}

impl Tt {
    /// Tabelle mit höchstens `mb` Megabyte.
    #[must_use]
    pub fn new(mb: usize) -> Tt {
        let n = (mb.max(1) * 1024 * 1024 / size_of::<Entry>()).next_power_of_two() / 2;
        Tt {
            slots: vec![Entry::default(); n.max(1024)],
            age: 0,
        }
    }

    /// Alle Einträge löschen.
    pub fn clear(&mut self) {
        self.slots.fill(Entry::default());
        self.age = 0;
    }

    /// Neue Suche: ältere Einträge werden bevorzugt ersetzt.
    pub fn next_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Bits gezielt abgeschnitten (Hash-Index)"
    )]
    fn idx(&self, key: u64) -> usize {
        (key as usize) & (self.slots.len() - 1)
    }

    /// Eintrag zur Stellung `key`, falls vorhanden.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let e = self.slots[self.idx(key)];
        (e.bound != Bound::None && e.key == key).then_some(e)
    }

    /// Speichert einen Eintrag. Ersetzt wird ein leerer, gleicher, veralteter oder flacherer Eintrag;
    /// ohne neuen Zug bleibt der alte Zug derselben Stellung erhalten.
    pub fn store(&mut self, new: Entry) {
        let age = self.age;
        let i = self.idx(new.key);
        let e = &mut self.slots[i];
        let same = e.key == new.key;
        if e.bound == Bound::None || same || e.age != age || new.depth >= e.depth {
            let mv = if new.mv == 0 && same { e.mv } else { new.mv };
            *e = Entry { mv, age, ..new };
        }
    }

    /// Belegung in Promille (Stichprobe der ersten 1000 Plätze, nur aktuelle Suche).
    #[must_use]
    pub fn hashfull(&self) -> usize {
        self.slots
            .iter()
            .take(1000)
            .filter(|e| e.bound != Bound::None && e.age == self.age)
            .count()
    }
}
