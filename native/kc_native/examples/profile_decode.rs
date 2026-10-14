//! Run with `cargo run --release --locked --example profile_decode`.
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use serde::de::{SeqAccess, Visitor};
use serde::Deserializer as _;
use std::hint::black_box;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

struct Strings;
impl<'de> Visitor<'de> for Strings {
    type Value = usize;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an array of strings")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<usize, A::Error> {
        let mut bytes = 0;
        while let Some(s) = seq.next_element::<String>()? {
            bytes += black_box(s).len();
        }
        Ok(bytes)
    }
}

fn inflate(blob: &[u8]) -> Vec<u8> {
    let mut raw = Vec::new();
    ZlibDecoder::new(blob).read_to_end(&mut raw).unwrap();
    raw
}

fn main() {
    let texts: Vec<_> = (0..200)
        .map(|i| format!("alpha {i} {}", "gateway memory \\ quoted \" text \u{5185}\u{5b58} ".repeat(80)))
        .collect();
    let raw = serde_json::to_vec(&texts).unwrap();
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(&raw).unwrap();
    let blob = enc.finish().unwrap();
    let mut rows = Vec::new();
    for round in 0..6 {
        let mut stages = vec!["inflate", "reader_json", "slice_json", "reader_total", "slice_total"];
        if round % 2 == 1 {
            stages.reverse();
        }
        for stage in stages {
            let start = Instant::now();
            for _ in 0..200 {
                match stage {
                    "inflate" => { black_box(inflate(&blob)); }
                    "reader_json" | "reader_total" => {
                        let source: Box<dyn Read> = if stage == "reader_total" {
                            Box::new(ZlibDecoder::new(blob.as_slice()))
                        } else {
                            Box::new(raw.as_slice())
                        };
                        let mut de = serde_json::Deserializer::from_reader(BufReader::with_capacity(1 << 16, source));
                        black_box((&mut de).deserialize_seq(Strings).unwrap());
                        de.end().unwrap();
                    }
                    _ => {
                        let inflated;
                        let bytes = if stage == "slice_total" {
                            inflated = inflate(&blob);
                            &inflated
                        } else { &raw };
                        let mut de = serde_json::Deserializer::from_slice(bytes);
                        black_box((&mut de).deserialize_seq(Strings).unwrap());
                        de.end().unwrap();
                    }
                }
            }
            rows.push(serde_json::json!({"round": round, "stage": stage,
                "us_per_call": start.elapsed().as_secs_f64() * 1e6 / 200.0,
                "loadavg": std::fs::read_to_string("/proc/loadavg").ok()}));
        }
    }
    println!("{}", serde_json::json!({"raw_bytes":raw.len(), "compressed_bytes":blob.len(), "calls_per_stage":200, "rows":rows}));
}
