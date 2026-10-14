//! Native primitives for `kiro_crew.history_search`: needle counting and snippet
//! extraction. Python keeps query parsing and every scoring decision; these two
//! functions only do the scans, and release the GIL while scanning a large input.
//!
//! EXACTNESS: each function must return what the Python code it replaces returns
//! for every input.
//!   * `count_needles` is `count_needle` (history_search.py) applied to several
//!     needles. It reads the str's own storage (latin-1, UCS-2 or UCS-4), so it
//!     counts code points exactly as `str.count` / `str.find` do, lone surrogates
//!     included, and never asks CPython for a UTF-8 copy (which CPython would keep
//!     inside the str for the rest of its life).
//!   * `snippet_from_blob` is the loop body of `_content_snippet` over the texts
//!     `SessionSearchIndex.raw_texts` would decode from the same blob.
//!   * `str.isdigit`, `str.isspace` and `str.casefold` come from tables generated
//!     from the interpreter (`search_tables.rs`); `UNIDATA_VERSION` records which.
//!
//! The Python side uses this module only when `SEARCH_FINGERPRINT` equals the hash
//! of its own definitions of those functions (`_native_search_fingerprint`), so an
//! edit to the Python semantics turns the native path off until this file is
//! updated, instead of letting the two drift apart.

use crate::search_tables::{DIGIT_RANGES, FOLDS, SPACES};
use flate2::read::ZlibDecoder;
use memchr::memmem;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString, PyTuple};
use std::io::Read;

/// `history_search._native_search_fingerprint()` for the Python this mirrors.
pub const SEARCH_FINGERPRINT: &str = "66f0dcc621a31526";

/// Inputs at least this large (in code units or bytes) are scanned with the GIL
/// released. Below it the release and re-acquire cost more than the scan, and a
/// re-acquire can wait out another thread's whole switch interval.
const DETACH_MIN: usize = 1 << 17;

/// Largest inflated blob accepted; a larger one is left to the Python path.
const INFLATE_MAX: u64 = 1 << 31;

// ------------------------------------------------------------------ unicode

#[inline]
pub fn is_digit(cp: u32) -> bool {
    if cp < 0x80 {
        return (0x30..=0x39).contains(&cp);
    }
    DIGIT_RANGES
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

#[inline]
pub fn is_space(cp: u32) -> bool {
    if cp < 0x80 {
        return matches!(cp, 0x09..=0x0D | 0x1C..=0x20);
    }
    SPACES.binary_search(&cp).is_ok()
}

/// Append `chr(cp).casefold()` to `out`.
#[inline]
pub fn fold_into(cp: u32, out: &mut Vec<u32>) {
    if cp < 0x80 {
        out.push(if (0x41..=0x5A).contains(&cp) {
            cp + 32
        } else {
            cp
        });
        return;
    }
    match FOLDS.binary_search_by_key(&cp, |&(c, _, _)| c) {
        Ok(i) => {
            let (_, folded, n) = FOLDS[i];
            out.extend_from_slice(&folded[..n as usize]);
        }
        Err(_) => out.push(cp),
    }
}

// ------------------------------------------------------------------ counting

/// A str's storage, borrowed. Element width is the str's PEP 393 kind.
pub enum Units<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
}

impl Units<'_> {
    fn len(&self) -> usize {
        match self {
            Units::U8(u) => u.len(),
            Units::U16(u) => u.len(),
            Units::U32(u) => u.len(),
        }
    }
}

/// Borrow `s`'s storage.
///
/// SAFETY: the caller keeps `s` alive for as long as the result is used. A str is
/// immutable and on 3.12+ always "ready", so the storage is valid and unchanged
/// for that whole time, also while the GIL is released.
unsafe fn units_of<'a>(s: &'a Bound<'_, PyString>) -> Units<'a> {
    let op = s.as_ptr();
    let len = ffi::PyUnicode_GET_LENGTH(op) as usize;
    let data = ffi::PyUnicode_DATA(op);
    match ffi::PyUnicode_KIND(op) {
        ffi::PyUnicode_1BYTE_KIND => Units::U8(std::slice::from_raw_parts(data as *const u8, len)),
        ffi::PyUnicode_2BYTE_KIND => {
            Units::U16(std::slice::from_raw_parts(data as *const u16, len))
        }
        _ => Units::U32(std::slice::from_raw_parts(data as *const u32, len)),
    }
}

pub trait Unit: Copy + Into<u32> + Send + Sync {
    const MAX: u32;
    fn from_cp(cp: u32) -> Self;
}

impl Unit for u8 {
    const MAX: u32 = 0xFF;
    fn from_cp(cp: u32) -> Self {
        cp as u8
    }
}

impl Unit for u16 {
    const MAX: u32 = 0xFFFF;
    fn from_cp(cp: u32) -> Self {
        cp as u16
    }
}

impl Unit for u32 {
    const MAX: u32 = u32::MAX;
    fn from_cp(cp: u32) -> Self {
        cp
    }
}

fn as_bytes<T: Unit>(s: &[T]) -> &[u8] {
    // SAFETY: T is u8/u16/u32, plain integers with no padding.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// `needle` in the haystack's width, or None when one of its code points is
/// wider than that width can hold (then it cannot occur at all).
fn narrow<T: Unit>(needle: &[u32]) -> Option<Vec<T>> {
    needle
        .iter()
        .map(|&cp| (cp <= T::MAX).then(|| T::from_cp(cp)))
        .collect()
}

/// Leftmost non-overlapping occurrences of `needle` in `hay`, as `str.count`
/// counts them; with `digit_bounded`, skipping a hit that sits inside a longer
/// number exactly as `count_needle`'s find loop does. `needle` is non-empty.
pub fn count_in<T: Unit>(hay: &[T], needle: &[T], digit_bounded: bool) -> usize {
    if needle.len() > hay.len() {
        return 0;
    }
    let width = std::mem::size_of::<T>();
    let hb = as_bytes(hay);
    let nb = as_bytes(needle);
    let finder = memmem::Finder::new(nb);
    let left_bounded = digit_bounded && is_digit(needle[0].into());
    let mut pos = 0usize;
    let mut total = 0usize;
    while pos + nb.len() <= hb.len() {
        let Some(off) = finder.find(&hb[pos..]) else {
            break;
        };
        let b = pos + off;
        if !b.is_multiple_of(width) {
            // A byte match straddling two code units is not a code-point match.
            pos = b + 1;
            continue;
        }
        let i = b / width;
        let end = i + needle.len();
        if !digit_bounded {
            total += 1;
        } else {
            let before_ok = !left_bounded || i == 0 || !is_digit(hay[i - 1].into());
            let after_ok = end >= hay.len() || !is_digit(hay[end].into());
            if before_ok && after_ok {
                total += 1;
            }
        }
        pos = end * width;
    }
    total
}

pub struct Spec {
    pub spellings: Vec<Vec<u32>>,
    pub digit_bounded: bool,
    pub must_hit: bool,
}

fn count_spec<T: Unit>(hay: &[T], spec: &Spec) -> usize {
    if hay.is_empty() {
        return 0;
    }
    spec.spellings
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| narrow::<T>(s).map_or(0, |n| count_in(hay, &n, spec.digit_bounded)))
        .sum()
}

/// Counts per spec in order, or None as soon as a `must_hit` spec counts zero.
/// The `must_hit` specs run first, so a disqualified text costs only them.
pub fn count_all<T: Unit>(hay: &[T], specs: &[Spec]) -> Option<Vec<usize>> {
    let mut out = vec![0usize; specs.len()];
    for (i, spec) in specs.iter().enumerate().filter(|(_, s)| s.must_hit) {
        let n = count_spec(hay, spec);
        if n == 0 {
            return None;
        }
        out[i] = n;
    }
    for (i, spec) in specs.iter().enumerate().filter(|(_, s)| !s.must_hit) {
        out[i] = count_spec(hay, spec);
    }
    Some(out)
}

fn code_points(s: &Bound<'_, PyString>) -> Vec<u32> {
    // SAFETY: `s` outlives this call and the copy below.
    match unsafe { units_of(s) } {
        Units::U8(u) => u.iter().map(|&c| c as u32).collect(),
        Units::U16(u) => u.iter().map(|&c| c as u32).collect(),
        Units::U32(u) => u.to_vec(),
    }
}

/// For each spec `(spellings, digit_bounded, must_hit)`, the summed count of its
/// spellings in `text` (`count_needle`), or None when a `must_hit` spec is absent.
#[pyfunction]
fn count_needles(
    py: Python<'_>,
    text: &Bound<'_, PyString>,
    specs: &Bound<'_, PyList>,
) -> PyResult<Option<Vec<usize>>> {
    let mut parsed = Vec::with_capacity(specs.len());
    for item in specs.iter() {
        let tup = item.cast::<PyTuple>()?;
        let spellings = tup.get_item(0)?;
        let spellings = spellings.cast::<PyTuple>()?;
        let mut cps = Vec::with_capacity(spellings.len());
        for s in spellings.iter() {
            cps.push(code_points(s.cast::<PyString>()?));
        }
        parsed.push(Spec {
            spellings: cps,
            digit_bounded: tup.get_item(1)?.extract()?,
            must_hit: tup.get_item(2)?.extract()?,
        });
    }
    // SAFETY: `text` is an argument, alive until this function returns.
    let units = unsafe { units_of(text) };
    let run = || match &units {
        Units::U8(h) => count_all(h, &parsed),
        Units::U16(h) => count_all(h, &parsed),
        Units::U32(h) => count_all(h, &parsed),
    };
    Ok(if units.len() < DETACH_MIN {
        run()
    } else {
        py.detach(run)
    })
}

// ------------------------------------------------------------------ snippets

fn find_cp(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    let hb = as_bytes(hay);
    let nb = as_bytes(needle);
    let finder = memmem::Finder::new(nb);
    let mut pos = 0usize;
    while pos + nb.len() <= hb.len() {
        let b = pos + finder.find(&hb[pos..])?;
        if b.is_multiple_of(4) {
            return Some(b / 4);
        }
        pos = b + 1;
    }
    None
}

/// `" ".join(text.split())` over `text`, as code points.
fn collapse_ws(text: &[u32], out: &mut Vec<u32>) {
    let mut first = true;
    for word in text.split(|&c| is_space(c)).filter(|w| !w.is_empty()) {
        if !first {
            out.push(0x20);
        }
        out.extend_from_slice(word);
        first = false;
    }
}

/// Reused per-call buffers for `snippet_in`.
#[derive(Default)]
struct SnippetBufs {
    original: Vec<u32>,
    folded: Vec<u32>,
}

/// The `_content_snippet` body for ONE text: the excerpt around the first
/// needle found in it, or None when it holds none of them.
fn snippet_in(
    text: &str,
    needles: &[Vec<u32>],
    lead: usize,
    trail: usize,
    max: usize,
    bufs: &mut SnippetBufs,
) -> Option<String> {
    let SnippetBufs { original, folded } = bufs;
    original.clear();
    original.extend(text.chars().map(|c| c as u32));
    folded.clear();
    for &cp in original.iter() {
        fold_into(cp, folded);
    }
    for needle in needles {
        let pos = if needle.is_empty() {
            0
        } else {
            match find_cp(folded, needle) {
                Some(pos) => pos,
                None => continue,
            }
        };
        let start = pos.saturating_sub(lead);
        let end = original.len().min(pos + needle.len() + trail);
        let mut out: Vec<u32> = Vec::new();
        if start > 0 {
            out.push(0x2026);
        }
        if start < end {
            collapse_ws(&original[start..end], &mut out);
        }
        if end < original.len() {
            out.push(0x2026);
        }
        out.truncate(max);
        return Some(
            out.into_iter()
                .map(|cp| char::from_u32(cp).expect("from a valid str"))
                .collect(),
        );
    }
    None
}

/// The `_content_snippet` loop over `texts`. `needles` are folded.
#[cfg(test)]
pub fn snippet_of(
    texts: &[String],
    needles: &[Vec<u32>],
    lead: usize,
    trail: usize,
    max: usize,
) -> String {
    let mut bufs = SnippetBufs::default();
    texts
        .iter()
        .find_map(|t| snippet_in(t, needles, lead, trail, max, &mut bufs))
        .unwrap_or_default()
}

/// Walks the entire JSON array one string at a time, retaining only the first
/// snippet. Later strings are validated but need no folding or snippet scan.
struct Scan<'a> {
    needles: &'a [Vec<u32>],
    lead: usize,
    trail: usize,
    max: usize,
}

impl<'de> serde::de::Visitor<'de> for Scan<'_> {
    type Value = Option<String>;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON array of strings")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut bufs = SnippetBufs::default();
        let mut found = None;
        while let Some(text) = seq.next_element::<String>()? {
            if found.is_none() {
                found = snippet_in(
                    &text,
                    self.needles,
                    self.lead,
                    self.trail,
                    self.max,
                    &mut bufs,
                );
            }
        }
        Ok(found)
    }
}

/// The first snippet in a fully validated zlib-compressed JSON string array,
/// or None when decoding fails or the inflated input exceeds the native limit.
pub fn snippet_from_compressed(
    blob: &[u8],
    needles: &[Vec<u32>],
    lead: usize,
    trail: usize,
    max: usize,
) -> Option<String> {
    scan_compressed(
        blob,
        Scan { needles, lead, trail, max },
        INFLATE_MAX,
        1 << 20,
    )
}

fn scan_compressed(blob: &[u8], scan: Scan<'_>, inflate_max: u64, buffer_max: u64) -> Option<String> {
    use serde::Deserializer as _;
    let mut inflate = ZlibDecoder::new(blob).take(inflate_max + 1);
    let mut prefix = Vec::new();
    // Small inputs use serde's slice scanner instead of its bytewise reader.
    // Larger inputs keep this bounded prefix and continue the same zlib stream.
    inflate.by_ref().take(buffer_max + 1).read_to_end(&mut prefix).ok()?;
    if inflate.limit() == 0 {
        return None;
    }
    let found = if prefix.len() as u64 <= buffer_max {
        // read_to_end reached real EOF, including the zlib checksum.
        let mut de = serde_json::Deserializer::from_slice(&prefix);
        let found = (&mut de).deserialize_seq(scan).ok()?;
        de.end().ok()?;
        found
    } else {
        let mut reader = std::io::BufReader::with_capacity(
            1 << 16,
            prefix.as_slice().chain(&mut inflate),
        );
        let mut de = serde_json::Deserializer::from_reader(&mut reader);
        let found = (&mut de).deserialize_seq(scan).ok()?;
        // End-of-JSON must reach the zlib terminator and verify its checksum.
        // flate2 reports an incomplete stream as UnexpectedEof, not clean EOF.
        de.end().ok()?;
        found
    };
    // Take's synthetic EOF cannot vouch for the stream; the extra byte lets an
    // input exactly at the limit reach its real EOF and checksum validation.
    if inflate.limit() == 0 {
        return None;
    }
    Some(found.unwrap_or_default())
}

/// The snippet `_content_snippet` builds from the texts stored in `blob` (a
/// `session_entry.raw_texts` value), or None when the complete zlib stream and
/// JSON string array cannot be validated, so the caller takes the Python path.
#[pyfunction]
fn snippet_from_blob(
    py: Python<'_>,
    blob: &Bound<'_, PyBytes>,
    needles: &Bound<'_, PyList>,
    lead: usize,
    trail: usize,
    max: usize,
) -> PyResult<Option<String>> {
    let mut parsed = Vec::with_capacity(needles.len());
    for n in needles.iter() {
        parsed.push(code_points(n.cast::<PyString>()?));
    }
    let bytes = blob.as_bytes();
    let run = || snippet_from_compressed(bytes, &parsed, lead, trail, max);
    Ok(if bytes.len() < DETACH_MIN / 4 {
        run()
    } else {
        py.detach(run)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SEARCH_FINGERPRINT", SEARCH_FINGERPRINT)?;
    m.add(
        "SEARCH_UNIDATA_VERSION",
        crate::search_tables::UNIDATA_VERSION,
    )?;
    m.add_function(wrap_pyfunction!(count_needles, m)?)?;
    m.add_function(wrap_pyfunction!(snippet_from_blob, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cps(s: &str) -> Vec<u32> {
        s.chars().map(|c| c as u32).collect()
    }

    #[test]
    fn counts_like_str_count() {
        let hay = cps("aaaa");
        assert_eq!(count_in(&hay, &cps("aa"), false), 2);
        let hay: Vec<u8> = b"pr 4411. #44110 1544110293".to_vec();
        let n: Vec<u8> = b"4411".to_vec();
        assert_eq!(count_in(&hay, &n, false), 3);
        assert_eq!(count_in(&hay, &n, true), 1);
    }

    #[test]
    fn straddling_bytes_are_not_matches() {
        // U+0100 U+0001 is bytes 00 01 01 00 (little endian): "01 01" spans units.
        let hay: Vec<u16> = vec![0x0100, 0x0001];
        assert_eq!(count_in(&hay, &[0x0101u16], false), 0);
    }

    #[test]
    fn must_hit_short_circuits() {
        let specs = vec![
            Spec {
                spellings: vec![cps("zz")],
                digit_bounded: false,
                must_hit: true,
            },
            Spec {
                spellings: vec![cps("a")],
                digit_bounded: false,
                must_hit: false,
            },
        ];
        assert_eq!(count_all(&cps("aaa"), &specs), None);
    }

    #[test]
    fn unicode_tables() {
        assert!(is_digit(0x663) && is_digit(0xB2) && !is_digit('a' as u32));
        assert!(is_space(0x1C) && is_space(0x3000) && !is_space('x' as u32));
        let mut out = Vec::new();
        fold_into(0xDF, &mut out);
        assert_eq!(out, cps("ss"));
    }

    #[test]
    fn snippet_window() {
        let texts = vec!["no".to_string(), "Hello   big\tWorld".to_string()];
        assert_eq!(
            snippet_of(&texts, &[cps("world")], 3, 1, 200),
            "\u{2026}ig World"
        );
    }

    fn compress(json: &str) -> Vec<u8> {
        use std::io::Write;
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(json.as_bytes()).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn bounded_decode_keeps_integrity() {
        let needles = [cps("alpha")];
        let decode = |blob: &[u8], limit, buffer| {
            scan_compressed(blob, Scan { needles: &needles, lead: 20, trail: 80, max: 200 }, limit, buffer)
        };
        for size in [31, 32, 33, 1024] {
            let raw = format!("[\"alpha\"]{}", " ".repeat(size - 9));
            let blob = compress(&raw);
            for buffer in [8, 32, 2048] {
                assert_eq!(decode(&blob, size as u64, buffer).as_deref(), Some("alpha"));
                assert_eq!(decode(&blob, size as u64 - 1, buffer), None);
                for end in 0..blob.len() {
                    assert_eq!(decode(&blob[..end], INFLATE_MAX, buffer), None);
                }
                let mut corrupt = blob.clone();
                *corrupt.last_mut().unwrap() ^= 1;
                assert_eq!(decode(&corrupt, INFLATE_MAX, buffer), None);
                assert_eq!(decode(&compress(&(raw.clone() + "false")), INFLATE_MAX, buffer), None);
            }
        }
        for raw in [r#"["alpha",1]"#, r#"["alpha","\ud800"]"#, r#"["alpha",]"#] {
            for buffer in [8, 2048] {
                assert_eq!(decode(&compress(raw), INFLATE_MAX, buffer), None);
            }
        }
    }

    #[test]
    fn compressed_blob() {
        let blob = compress(r#"["alpha", "Beta \u00df", "gamma"]"#);
        assert_eq!(
            snippet_from_compressed(&blob, &[cps("ss")], 40, 100, 200).as_deref(),
            Some("Beta \u{df}")
        );
        assert_eq!(
            snippet_from_compressed(&blob, &[cps("zz")], 40, 100, 200).as_deref(),
            Some("")
        );
        assert_eq!(
            snippet_from_compressed(b"junk", &[cps("a")], 40, 100, 200),
            None
        );
        let not_strings = compress("[1, 2]");
        assert_eq!(
            snippet_from_compressed(&not_strings, &[cps("a")], 40, 100, 200),
            None
        );
        let mut truncated = compress(r#"["alpha", "beta"]"#);
        truncated.truncate(truncated.len() - 2);
        assert_eq!(
            snippet_from_compressed(&truncated, &[cps("zz")], 40, 100, 200),
            None
        );
    }
}
