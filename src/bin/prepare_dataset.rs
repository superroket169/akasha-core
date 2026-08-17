//! Converts a downloaded FineWeb-Edu parquet shard into data/train.txt +
//! data/eval.txt. The first `eval_docs` documents go to eval, the rest to
//! train — sample-10BT shards are pre-shuffled, so a prefix slice is a valid
//! held-out set with zero overlap by construction.
//!
//! Usage: cargo run --release --bin prepare_dataset -- <parquet_path> [eval_docs]

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;
use std::io::{BufWriter, Write};

const DEFAULT_EVAL_DOCS: usize = 2000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parquet_path = args.get(1).unwrap_or_else(|| {
        eprintln!("usage: prepare_dataset <parquet_path> [eval_docs={DEFAULT_EVAL_DOCS}]");
        std::process::exit(1);
    });
    let eval_docs: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EVAL_DOCS);

    let file = File::open(parquet_path).unwrap_or_else(|e| panic!("cannot open {parquet_path}: {e}"));
    let reader =
        SerializedFileReader::new(file).unwrap_or_else(|e| panic!("not a valid parquet file: {e}"));

    let text_col = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.name() == "text")
        .expect("parquet file has no `text` column");

    std::fs::create_dir_all("data").unwrap();
    let mut eval = BufWriter::new(File::create("data/eval.txt").unwrap());
    let mut train = BufWriter::new(File::create("data/train.txt").unwrap());

    let mut docs = 0usize;
    let mut bytes = 0usize;
    for row in reader.get_row_iter(None).expect("cannot read rows") {
        let row = row.expect("corrupt row");
        let text = row.get_string(text_col).expect("text column not a string");
        let out = if docs < eval_docs { &mut eval } else { &mut train };
        out.write_all(text.as_bytes()).unwrap();
        out.write_all(b"\n").unwrap();
        bytes += text.len() + 1;
        docs += 1;
        if docs % 50_000 == 0 {
            println!("  {docs} documents...");
        }
    }
    eval.flush().unwrap();
    train.flush().unwrap();

    println!(
        "{docs} documents, {:.2} GB text -> data/train.txt ({} docs) + data/eval.txt ({} docs)",
        bytes as f64 / 1e9,
        docs.saturating_sub(eval_docs),
        eval_docs.min(docs),
    );
}
