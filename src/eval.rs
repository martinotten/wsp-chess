//! Dünne Auswertung des handgeschriebenen Netzes (Export: `chess/export_rs.py`).
//!
//! Ziele `T = [z1 | z2 | z3 | out]`. Jede Quelle (768 Eingaben, dann h1, h2, h3) hat eine Liste
//! (Zielindex, Gewicht) in spätere Schichten. Weil je Stellung nur ~20 h1-, ~24 h2- und ~2 h3-Neuronen
//! aktiv sind, werden nur deren Zeilen aufaddiert. Ergebnis `s` in Bauern, Sicht des Spielers am Zug.
//!
//! Der Eingabeteil (Biases plus Beiträge aller gesetzten Eingabebits zu allen Zielen) heißt
//! Akkumulator. Er lässt sich mit [`Net::update`] von einem Brett auf ein anderes nachführen.

use shakmaty::{Board, Color, Piece, Position, Role, Square};
use std::fmt;

/// Zahl der Eingabebits (12 Figurenarten × 64 Felder).
pub const N_INPUTS: usize = 768;
const MAGIC: &[u8; 4] = b"WSPN";
const VERSION: u32 = 1;

/// Fehler beim Laden der Gewichtsdatei.
#[derive(Debug)]
pub enum LoadError {
    /// Datei nicht lesbar.
    Io(std::io::Error),
    /// Inhalt passt nicht zum Format.
    Format(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "Gewichtsdatei nicht lesbar: {e}"),
            LoadError::Format(m) => write!(f, "Gewichtsdatei ungültig: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Das Netz in dünner Darstellung (je Quelle eine Liste von Zielen).
pub struct Net {
    n1: usize,
    n2: usize,
    n3: usize,
    bias: Vec<f32>,
    starts: Vec<usize>,
    tgt: Vec<usize>,
    w: Vec<f32>,
}

/// Eingabeindex wie `features.xi` in Python: `(Seite*6 + Typ)*64 + Feld`, Sicht `persp`
/// (Seite 0 = Figuren von `persp`; ist `persp` Schwarz, wird das Brett vertikal gespiegelt).
#[must_use]
pub fn feature(persp: Color, piece: Piece, sq: Square) -> usize {
    let side = usize::from(piece.color != persp);
    let t = usize::from(piece.role) - 1;
    let s = if persp == Color::Black {
        sq.flip_vertical()
    } else {
        sq
    };
    (side * 6 + t) * 64 + s.to_usize()
}

/// Ruft `f(i, v)` für alle `v > 0` auf; Blöcke zu 16 Werten werden zuerst als Ganzes geprüft
/// (vektorisierbar), denn fast alle Neuronen sind inaktiv.
#[inline]
#[expect(
    clippy::needless_bitwise_bool,
    reason = "ohne Verzweigung, damit der Block vektorisiert wird"
)]
fn for_active(v: &[f32], mut f: impl FnMut(usize, f32)) {
    for (ci, ch) in v.chunks(16).enumerate() {
        if ch.iter().fold(false, |a, &x| a | (x > 0.0)) {
            for (j, &x) in ch.iter().enumerate() {
                if x > 0.0 {
                    f(ci * 16 + j, x);
                }
            }
        }
    }
}

/// Liest Little-Endian-Werte fester Breite der Reihe nach.
struct Reader<'a> {
    b: &'a [u8],
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], LoadError> {
        if self.b.len() < n {
            return Err(LoadError::Format("Datei zu kurz"));
        }
        let (head, rest) = self.b.split_at(n);
        self.b = rest;
        Ok(head)
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, LoadError> {
        Ok(self
            .take(4 * n)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect())
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, LoadError> {
        Ok(self
            .take(4 * n)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    }

    fn usizes(&mut self, n: usize) -> Result<Vec<usize>, LoadError> {
        self.u32s(n)?
            .into_iter()
            .map(|x| usize::try_from(x).map_err(|_| LoadError::Format("Index zu groß")))
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
    pub fn from_bytes(b: &[u8]) -> Result<Net, LoadError> {
        let mut r = Reader { b };
        if r.take(4)? != MAGIC {
            return Err(LoadError::Format("Kennung WSPN fehlt"));
        }
        if r.u32s(1)? != [VERSION] {
            return Err(LoadError::Format("unbekannte Version"));
        }
        let sizes = r.usizes(3)?;
        let (n1, n2, n3) = (sizes[0], sizes[1], sizes[2]);
        let nt = n1 + n2 + n3 + 1;
        let ns = N_INPUTS + n1 + n2 + n3;
        let bias = r.f32s(nt)?;
        let starts = r.usizes(ns + 1)?;
        let nnz = starts[ns];
        let tgt = r.usizes(nnz)?;
        let w = r.f32s(nnz)?;
        if !r.b.is_empty() {
            return Err(LoadError::Format("Daten nach dem Ende"));
        }
        if starts[0] != 0 || starts.windows(2).any(|p| p[0] > p[1]) {
            return Err(LoadError::Format("Zeilenanfänge nicht aufsteigend"));
        }
        // Erstes erlaubtes Ziel je Quelle: Eingaben -> alles, h1 -> ab z2, h2 -> ab z3, h3 -> nur out
        for src in 0..ns {
            let min = if src < N_INPUTS {
                0
            } else if src < N_INPUTS + n1 {
                n1
            } else if src < N_INPUTS + n1 + n2 {
                n1 + n2
            } else {
                n1 + n2 + n3
            };
            if tgt[starts[src]..starts[src + 1]]
                .iter()
                .any(|&t| t < min || t >= nt)
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
            tgt,
            w,
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

    /// Addiert `scale` × Zeile `src` auf `z`; Ziele werden um `off` verschoben.
    #[inline]
    fn add_row(&self, src: usize, scale: f32, z: &mut [f32], off: usize) {
        let (a, b) = (self.starts[src], self.starts[src + 1]);
        for (&t, &w) in self.tgt[a..b].iter().zip(&self.w[a..b]) {
            z[t - off] += w * scale;
        }
    }

    /// Akkumulator für eine Perspektive von Grund auf.
    pub fn refresh(&self, board: &Board, persp: Color, acc: &mut [f32]) {
        acc.copy_from_slice(&self.bias);
        for (sq, piece) in board {
            self.add_row(feature(persp, piece, sq), 1.0, acc, 0);
        }
    }

    /// Führt den Akkumulator `acc` (Perspektive `persp`) von Brett `from` auf Brett `to` nach.
    /// Gibt `false` zurück (ohne Änderung), wenn sich mehr als `max` Figuren unterscheiden.
    pub fn update(
        &self,
        from: &Board,
        to: &Board,
        persp: Color,
        acc: &mut [f32],
        max: usize,
        buf: &mut Vec<(usize, f32)>,
    ) -> bool {
        buf.clear();
        for color in Color::ALL {
            for role in Role::ALL {
                let piece = Piece { color, role };
                let (a, b) = (from.by_piece(piece), to.by_piece(piece));
                buf.extend((a & !b).into_iter().map(|sq| (feature(persp, piece, sq), -1.0)));
                buf.extend((b & !a).into_iter().map(|sq| (feature(persp, piece, sq), 1.0)));
                if buf.len() > max {
                    return false;
                }
            }
        }
        for &(f, sign) in buf.iter() {
            self.add_row(f, sign, acc, 0);
        }
        true
    }

    /// Bewertung `s` (Bauern) aus dem Akkumulator der Perspektive des Spielers am Zug.
    /// `work` muss [`Net::n_work`] lang sein.
    pub fn score_from(&self, acc: &[f32], work: &mut [f32]) -> f32 {
        let (n1, n2, n3) = (self.n1, self.n2, self.n3);
        work.copy_from_slice(&acc[n1..]);
        for_active(&acc[..n1], |i, h| self.add_row(N_INPUTS + i, h, work, n1));
        let (h2, rest) = work.split_at_mut(n2);
        for_active(h2, |j, h| self.add_row(N_INPUTS + n1 + j, h, rest, n1 + n2));
        let (h3, out) = rest.split_at_mut(n3);
        for_active(h3, |k, h| {
            self.add_row(N_INPUTS + n1 + n2 + k, h, out, n1 + n2 + n3);
        });
        out[0]
    }

    /// Bewertung ohne Akkumulator (Referenz für Tests und den Befehl `eval`).
    #[must_use]
    pub fn score<P: Position>(&self, pos: &P) -> f32 {
        let mut acc = vec![0.0; self.n_targets()];
        let mut work = vec![0.0; self.n_work()];
        self.refresh(pos.board(), pos.turn(), &mut acc);
        self.score_from(&acc, &mut work)
    }
}
