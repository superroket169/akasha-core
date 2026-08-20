//! Converts a downloaded databricks-dolly-15k parquet file into instruction-
//! formatted data/train.txt + data/eval.txt. Each row becomes a
//! "User: ...\nAssistant: ...\n\n" block

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufWriter, Write};

const DEFAULT_EVAL_DOCS: usize = 500;

fn col_index(reader: &SerializedFileReader<File>, name: &str) -> usize {
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|c| c.name() == name)
        .unwrap_or_else(|| panic!("parquet file has no `{name}` column"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parquet_path = args.get(1).unwrap_or_else(|| {
        eprintln!("usage: prepare_finetune_dataset <parquet_path> [eval_docs={DEFAULT_EVAL_DOCS}]");
        std::process::exit(1);
    });
    let eval_docs: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EVAL_DOCS);

    let file =
        File::open(parquet_path).unwrap_or_else(|e| panic!("cannot open {parquet_path}: {e}"));
    let reader =
        SerializedFileReader::new(file).unwrap_or_else(|e| panic!("not a valid parquet file: {e}"));

    let instruction_col = col_index(&reader, "instruction");
    let context_col = col_index(&reader, "context");
    let response_col = col_index(&reader, "response");

    let mut blocks = Vec::new();
    for row in reader.get_row_iter(None).expect("cannot read rows") {
        let row = row.expect("corrupt row");
        let instruction = row
            .get_string(instruction_col)
            .expect("instruction not a string");
        let context = row.get_string(context_col).expect("context not a string");
        let response = row.get_string(response_col).expect("response not a string");

        let block = if context.trim().is_empty() {
            format!(
                "User: {}\nAssistant: {}\n\n",
                instruction.trim(),
                response.trim()
            )
        } else {
            format!(
                "User: {}\n\n{}\nAssistant: {}\n\n",
                instruction.trim(),
                context.trim(),
                response.trim()
            )
        };
        blocks.push(block);
    }

    // Fixed seed: reproducible eval split across runs.
    let mut rng = rand::rngs::StdRng::seed_from_u64(1337);
    blocks.shuffle(&mut rng);

    let eval_docs = eval_docs.min(blocks.len());
    let (eval_blocks, train_blocks) = blocks.split_at(eval_docs);

    std::fs::create_dir_all("data").unwrap();
    let mut eval = BufWriter::new(File::create("data/eval.txt").unwrap());
    let mut train = BufWriter::new(File::create("data/train.txt").unwrap());

    let mut train_bytes = 0usize;
    for b in train_blocks {
        train_bytes += b.len();
        train.write_all(b.as_bytes()).unwrap();
    }
    let mut eval_bytes = 0usize;
    for b in eval_blocks {
        eval_bytes += b.len();
        eval.write_all(b.as_bytes()).unwrap();
    }
    train.flush().unwrap();
    eval.flush().unwrap();

    println!(
        "{} rows -> data/train.txt ({} rows, {:.2} MB) + data/eval.txt ({} rows, {:.2} MB)",
        blocks.len(),
        train_blocks.len(),
        train_bytes as f64 / 1e6,
        eval_blocks.len(),
        eval_bytes as f64 / 1e6,
    );
}
