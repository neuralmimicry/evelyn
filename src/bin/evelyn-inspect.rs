//! `evelyn-inspect <model.gguf> [layer]`: architecture, FFN layout and
//! quantisation of a checkpoint (stage 2 importer front end).
use evelyn::gguf::{Gguf, Value, type_name};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let g = Gguf::open(&args[1])?;
    let arch = g.architecture().unwrap_or("?").to_string();
    println!(
        "GGUF v{}  arch={arch}  tensors={}",
        g.version,
        g.tensors.len()
    );
    for (k, v) in &g.metadata {
        if k.starts_with(&format!("{arch}.")) || k.starts_with("general.") && !k.contains("license")
        {
            let s = match v {
                Value::Array(a) => format!("[array {}]", a.len()),
                other => format!("{other:?}"),
            };
            println!("  {k} = {}", &s[..s.len().min(100)]);
        }
    }
    let layer = args.get(2).map(String::as_str).unwrap_or("0");
    let prefix = format!("blk.{layer}.");
    for t in g.tensors.values().filter(|t| t.name.starts_with(&prefix)) {
        println!(
            "  {:<40} {:<5} {:?}",
            t.name,
            type_name(t.ggml_type),
            t.dims
        );
    }
    let mut hist = std::collections::BTreeMap::new();
    for t in g.tensors.values() {
        *hist.entry(type_name(t.ggml_type)).or_insert(0usize) += 1;
    }
    println!("tensor types: {hist:?}");
    Ok(())
}
