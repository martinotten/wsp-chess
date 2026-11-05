//! Dünne Auswertung des handgeschriebenen Netzes (Export: `chess/export_rs.py`).
//!
//! Ziele `T = [z1 | z2 | z3 | out]`. Jede Quelle (768 Eingaben, dann h1, h2, h3) hat eine Liste
//! (Zielindex, Gewicht) in spätere Schichten. Weil je Stellung nur ~20 h1-, ~24 h2- und ~2 h3-Neuronen
//! aktiv sind, werden nur deren Zeilen aufaddiert. Ergebnis `s` in Bauern, Sicht des Spielers am Zug.
//!
//! Der Eingabeteil (Biases plus Beiträge aller gesetzten Eingabebits zu allen Zielen) heißt
//! Akkumulator. Er lässt sich mit [`Net::update`] von einem Brett auf ein anderes nachführen.

use crate::board::{Color, Piece, Position, Square};
use cozy_chess::Board;
use std::fmt;

/// Zahl der Eingabebits (12 Figurenarten × 64 Felder).
pub const N_INPUTS: usize = 768;
const MAGIC: &[u8; 4] = b"WSPN";
const VERSION: u32 = 1;
/// Höchstzahl geänderter Merkmale, bis zu der [`Net::update`] nachführt statt neu zu berechnen.
pub const MAX_CHANGES: usize = 12;

/// Fehler beim Laden der Gewichtsdatei.
#[derive(Debug)]
pub enum LoadError {
    /// Datei nicht lesbar.
    Io(std::io::Error),
    /// Inhalt passt nicht zum Format.
    Format(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(err) => write!(out, "Gewichtsdatei nicht lesbar: {err}"),
            LoadError::Format(reason) => write!(out, "Gewichtsdatei ungültig: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Das Netz in dünner Darstellung (je Quelle eine Liste von Zielen).
///
/// Quellen sind die Eingabemerkmale und die Neuronen h1, h2, h3; Ziele sind die Vorstufen z1, z2,
/// z3 und der Ausgang, hintereinander in einem Vektor. Die Gewichte der Quelle `src` stehen in
/// `targets[starts[src]..starts[src + 1]]` und `weights[…]` (gleicher Bereich).
pub struct Net {
    /// Größen der Schichten h1, h2, h3.
    n1: usize,
    n2: usize,
    n3: usize,
    bias: Vec<f32>,
    starts: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<f32>,
}

/// Eingabeindex wie `features.xi` in Python: `(Seite*6 + Typ)*64 + Feld`, Sicht `persp`
/// (Seite 0 = Figuren von `persp`; ist `persp` Schwarz, wird das Brett vertikal gespiegelt).
#[must_use]
pub fn feature(persp: Color, color: Color, role: Piece, sq: Square) -> usize {
    let side = usize::from(color != persp);
    let seen_sq = if persp == Color::Black { sq.flip_rank() } else { sq };
    (side * 6 + role as usize) * 64 + seen_sq as usize
}

/// Ruft `visit(index, activation)` für alle positiven Werte auf; Blöcke zu 16 Werten werden zuerst
/// als Ganzes geprüft, denn fast alle Neuronen sind inaktiv.
///
/// Vorprüfung über das Bitmuster als `i32`: Eine positive Zahl hat ein Muster `> 0`, `0.0`, `-0.0` und
/// negative Zahlen nicht. Das Maximum ganzer Zahlen vektorisiert zuverlässig. Die Vorprüfung lässt
/// keinen positiven Wert aus; entschieden wird danach wie bisher mit `value > 0.0`.
#[inline]
fn for_active(values: &[f32], mut visit: impl FnMut(usize, f32)) {
    for (block_index, block) in values.chunks(16).enumerate() {
        let max_bits = block
            .iter()
            .fold(i32::MIN, |max, value| max.max(value.to_bits().cast_signed()));
        if max_bits > 0 {
            for (offset, &value) in block.iter().enumerate() {
                if value > 0.0 {
                    visit(block_index * 16 + offset, value);
                }
            }
        }
    }
}

/// Liest Little-Endian-Werte fester Breite der Reihe nach.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8], LoadError> {
        if self.bytes.len() < len {
            return Err(LoadError::Format("Datei zu kurz"));
        }
        let (head, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        Ok(head)
    }

    fn u32s(&mut self, count: usize) -> Result<Vec<u32>, LoadError> {
        Ok(self
            .take(4 * count)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect())
    }

    fn f32s(&mut self, count: usize) -> Result<Vec<f32>, LoadError> {
        Ok(self
            .take(4 * count)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect())
    }

    fn usizes(&mut self, count: usize) -> Result<Vec<usize>, LoadError> {
        self.u32s(count)?
            .into_iter()
            .map(|value| usize::try_from(value).map_err(|_| LoadError::Format("Index zu groß")))
            .collect()
    }
}

impl Net {
    /// Lädt die Gewichte aus einer Datei im Format von `export_rs.py`.
    ///
    /// # Errors
    /// Wenn die Datei fehlt oder nicht zum Format passt.
    pub fn load(path: &str) -> Result<Net, LoadError> {
        Net::from_bytes(&std::fs::read(path).map_err(LoadError::Io)?)
    }

    /// Liest die Gewichte aus Bytes und prüft die Struktur: Jede Quelle wirkt nur auf spätere
    /// Schichten. Danach können [`Net::score_from`] und [`Net::update`] nur innerhalb der
    /// Vektoren schreiben.
    ///
    /// # Errors
    /// Wenn die Bytes nicht zum Format passen.
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, LoadError> {
        let mut reader = Reader { bytes };
        if reader.take(4)? != MAGIC {
            return Err(LoadError::Format("Kennung WSPN fehlt"));
        }
        if reader.u32s(1)? != [VERSION] {
            return Err(LoadError::Format("unbekannte Version"));
        }
        let sizes = reader.usizes(3)?;
        let (n1, n2, n3) = (sizes[0], sizes[1], sizes[2]);
        let n_targets = n1 + n2 + n3 + 1;
        let n_sources = N_INPUTS + n1 + n2 + n3;
        let bias = reader.f32s(n_targets)?;
        let starts = reader.usizes(n_sources + 1)?;
        let nnz = starts[n_sources];
        let targets = reader.usizes(nnz)?;
        let weights = reader.f32s(nnz)?;
        if !reader.bytes.is_empty() {
            return Err(LoadError::Format("Daten nach dem Ende"));
        }
        if starts[0] != 0 || starts.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(LoadError::Format("Zeilenanfänge nicht aufsteigend"));
        }
        // Erstes erlaubtes Ziel je Quelle: Eingaben -> alles, h1 -> ab z2, h2 -> ab z3, h3 -> nur out
        for src in 0..n_sources {
            let min = if src < N_INPUTS {
                0
            } else if src < N_INPUTS + n1 {
                n1
            } else if src < N_INPUTS + n1 + n2 {
                n1 + n2
            } else {
                n1 + n2 + n3
            };
            if targets[starts[src]..starts[src + 1]]
                .iter()
                .any(|&target| target < min || target >= n_targets)
            {
                return Err(LoadError::Format("Gewicht in eine frühere Schicht"));
            }
        }
        Ok(Net {
            n1,
            n2,
            n3,
            bias,
            starts,
            targets,
            weights,
        })
    }

    /// Länge eines Akkumulators.
    #[must_use]
    pub fn n_targets(&self) -> usize {
        self.n1 + self.n2 + self.n3 + 1
    }

    /// Länge des Arbeitsvektors für [`Net::score_from`] (z2, z3, out).
    #[must_use]
    pub fn n_work(&self) -> usize {
        self.n2 + self.n3 + 1
    }

    /// Ziele und Gewichte der Quelle `src` in gespeicherter Reihenfolge (Referenz für Tests).
    #[cfg(test)]
    fn row(&self, src: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let row = self.starts[src]..self.starts[src + 1];
        self.targets[row.clone()]
            .iter()
            .copied()
            .zip(self.weights[row].iter().copied())
    }

    /// Addiert `scale` × Gewichte der Quelle `src` auf `dest`, der ab Ziel `offset` beginnt.
    #[inline]
    fn add_row(&self, src: usize, scale: f32, dest: &mut [f32], offset: usize) {
        let row = self.starts[src]..self.starts[src + 1];
        for (&target, &weight) in self.targets[row.clone()].iter().zip(&self.weights[row]) {
            dest[target - offset] += weight * scale;
        }
    }

    /// Akkumulator für eine Perspektive von Grund auf.
    pub fn refresh(&self, board: &Board, persp: Color, acc: &mut [f32]) {
        acc.copy_from_slice(&self.bias);
        // aufsteigende Felder: feste Summationsreihenfolge, damit Werte bitgleich reproduzierbar sind
        for sq in board.occupied() {
            if let (Some(color), Some(role)) = (board.color_on(sq), board.piece_on(sq)) {
                self.add_row(feature(persp, color, role, sq), 1.0, acc, 0);
            }
        }
    }

    /// Führt den Akkumulator `acc` (Perspektive `persp`) von Brett `from` auf Brett `to` nach.
    /// Gibt `false` zurück (ohne Änderung), wenn sich mehr als [`MAX_CHANGES`] Merkmale unterscheiden;
    /// dann ist eine Neuberechnung mit [`Net::refresh`] billiger. `buf` ist Arbeitsspeicher.
    pub fn update(
        &self,
        from: &Board,
        to: &Board,
        persp: Color,
        acc: &mut [f32],
        buf: &mut Vec<(usize, f32)>,
    ) -> bool {
        buf.clear();
        for color in Color::ALL {
            for role in Piece::ALL {
                let (before, after) = (from.colored_pieces(color, role), to.colored_pieces(color, role));
                buf.extend(
                    (before & !after)
                        .into_iter()
                        .map(|sq| (feature(persp, color, role, sq), -1.0)),
                );
                buf.extend(
                    (after & !before)
                        .into_iter()
                        .map(|sq| (feature(persp, color, role, sq), 1.0)),
                );
                if buf.len() > MAX_CHANGES {
                    return false;
                }
            }
        }
        for &(input, sign) in buf.iter() {
            self.add_row(input, sign, acc, 0);
        }
        true
    }

    /// Bewertung `s` (Bauern) aus dem Akkumulator der Perspektive des Spielers am Zug.
    /// `work` muss [`Net::n_work`] lang sein.
    pub fn score_from(&self, acc: &[f32], work: &mut [f32]) -> f32 {
        let (n1, n2, n3) = (self.n1, self.n2, self.n3);
        work.copy_from_slice(&acc[n1..]);
        for_active(&acc[..n1], |neuron, act| {
            self.add_row(N_INPUTS + neuron, act, work, n1);
        });
        let (h2, rest) = work.split_at_mut(n2);
        for_active(h2, |neuron, act| {
            self.add_row(N_INPUTS + n1 + neuron, act, rest, n1 + n2);
        });
        let (h3, out) = rest.split_at_mut(n3);
        for_active(h3, |neuron, act| {
            self.add_row(N_INPUTS + n1 + n2 + neuron, act, out, n1 + n2 + n3);
        });
        out[0]
    }

    /// Bewertung ohne Akkumulator (Referenz für Tests und den Befehl `eval`).
    #[must_use]
    pub fn score(&self, pos: &Position) -> f32 {
        let mut acc = vec![0.0; self.n_targets()];
        let mut work = vec![0.0; self.n_work()];
        self.refresh(pos.board(), pos.turn(), &mut acc);
        self.score_from(&acc, &mut work)
    }
}

#[cfg(test)]
mod tests;
