//! Schach-Engine mit dem handgeschriebenen Netz (Weight-Space Programming) als Bewertung.
//!
//! - [`eval`] – dünne, inkrementelle Auswertung des Netzes
//! - [`exchange`] – Materialwerte und Abtauschbewertung (SEE)
//! - [`search`] – Alpha-Beta-Suche mit abschaltbaren Techniken
//! - [`tt`] – Transpositionstabelle
//! - [`uci`] – UCI-Protokoll

pub mod board;
pub mod eval;
pub mod exchange;
pub mod search;
pub mod tt;
pub mod uci;
