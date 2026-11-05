//! `wsp-chess`: UCI engine using the hand-written chess network as its evaluation.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

fn main() {
    let out: Box<dyn Write + Send> = Box::new(io::stdout());
    wsp_chess::uci::run(io::stdin().lock(), Arc::new(Mutex::new(out)));
}
