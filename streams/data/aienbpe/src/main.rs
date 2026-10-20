//! aienbpe: byte-level BPE for the aien-train corpus. Standard library only.
//!
//! Document stream (stdin of train/encode, stdout of decode): repeated [u32 LE byte length][bytes].
//! Tokenizer file `AIENBPE 1`: see docs/data/CORPUS.md. Token file `AIENTOK1`: same document.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

const MAX_CHUNK: usize = 128;
const PRETOK_ID: &str = "aien-pretok-1";

// ---------- small fast hasher (FxHash-style) ----------
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut v = 0u64;
            for (i, b) in c.iter().enumerate() {
                v |= (*b as u64) << (8 * i);
            }
            self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
        self.0 = (self.0.rotate_left(5) ^ bytes.len() as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64)
    }
    fn finish(&self) -> u64 {
        self.0
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

// ---------- pre-tokenizer aien-pretok-1 (byte classes; documented in CORPUS.md) ----------
#[inline]
fn class(b: u8) -> u8 {
    match b {
        b'A'..=b'Z' | b'a'..=b'z' | 0x80..=0xFF => 0, // letter (any non-ASCII byte counts as letter)
        b'0'..=b'9' => 1,
        0x09..=0x0D | 0x20 => 2, // space class
        _ => 3,                  // punctuation and controls
    }
}

fn emit_capped(s: &[u8], f: &mut impl FnMut(&[u8])) {
    for p in s.chunks(MAX_CHUNK) {
        f(p);
    }
}

/// Run of `c` starting at `i`; digit runs are at most 3 long.
fn run_end(d: &[u8], i: usize, c: u8) -> usize {
    let mut k = i;
    while k < d.len() && class(d[k]) == c {
        k += 1;
        if c == 1 && k - i == 3 {
            break;
        }
    }
    k
}

fn pretok(d: &[u8], f: &mut impl FnMut(&[u8])) {
    let n = d.len();
    let mut i = 0;
    while i < n {
        let c = class(d[i]);
        if c != 2 {
            let k = run_end(d, i, c);
            emit_capped(&d[i..k], f);
            i = k;
            continue;
        }
        let mut j = i;
        while j < n && class(d[j]) == 2 {
            j += 1;
        }
        if j < n && d[j - 1] == b' ' {
            if j - 1 > i {
                emit_capped(&d[i..j - 1], f);
                i = j - 1;
            } else {
                let k = run_end(d, j, class(d[j]));
                emit_capped(&d[i..k], f);
                i = k;
            }
        } else {
            emit_capped(&d[i..j], f);
            i = j;
        }
    }
}

// ---------- tokenizer model ----------
struct Model {
    merges: Vec<(u32, u32)>,
    vocab: u32, // includes 256 bytes, merges, and the one special token
    rank: FxMap<u64, u32>, // pair -> new token id
}
fn pk(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}
impl Model {
    fn new(merges: Vec<(u32, u32)>) -> Model {
        let mut rank = FxMap::default();
        for (i, &(a, b)) in merges.iter().enumerate() {
            rank.insert(pk(a, b), 256 + i as u32);
        }
        let vocab = 256 + merges.len() as u32 + 1;
        Model { merges, vocab, rank }
    }
    fn eos(&self) -> u32 {
        self.vocab - 1
    }
    fn load(path: &str) -> Model {
        let f = BufReader::new(std::fs::File::open(path).expect("open tokenizer"));
        let mut lines = f.lines().map(|l| l.unwrap());
        assert_eq!(lines.next().unwrap(), "AIENBPE 1");
        let merges_n;
        let mut vocab = 0u32;
        loop {
            let l = lines.next().expect("header");
            if let Some(v) = l.strip_prefix("pretok ") {
                assert_eq!(v, PRETOK_ID);
            } else if let Some(v) = l.strip_prefix("vocab_size ") {
                vocab = v.parse().unwrap();
            } else if l.starts_with("eos_id ") {
            } else if let Some(v) = l.strip_prefix("merges ") {
                merges_n = v.parse().unwrap();
                break;
            }
        }
        let mut merges = Vec::with_capacity(merges_n);
        for _ in 0..merges_n {
            let l = lines.next().expect("merge line");
            let mut it = l.split(' ');
            merges.push((it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap()));
        }
        let m = Model::new(merges);
        assert_eq!(m.vocab, vocab, "vocab_size does not match merge count");
        m
    }
    fn save(&self, path: &str) {
        let mut w = BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(w, "AIENBPE 1").unwrap();
        writeln!(w, "pretok {}", PRETOK_ID).unwrap();
        writeln!(w, "vocab_size {}", self.vocab).unwrap();
        writeln!(w, "eos_id {}", self.eos()).unwrap();
        writeln!(w, "merges {}", self.merges.len()).unwrap();
        for &(a, b) in &self.merges {
            writeln!(w, "{} {}", a, b).unwrap();
        }
    }
    fn encode_chunk(&self, chunk: &[u8], ids: &mut Vec<u32>) {
        ids.clear();
        ids.extend(chunk.iter().map(|&b| b as u32));
        loop {
            let mut best = u32::MAX;
            let mut bp = (0, 0);
            for w in ids.windows(2) {
                if let Some(&r) = self.rank.get(&pk(w[0], w[1])) {
                    if r < best {
                        best = r;
                        bp = (w[0], w[1]);
                    }
                }
            }
            if best == u32::MAX {
                return;
            }
            let mut o = 0;
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == bp.0 && ids[i + 1] == bp.1 {
                    ids[o] = best;
                    i += 2;
                } else {
                    ids[o] = ids[i];
                    i += 1;
                }
                o += 1;
            }
            ids.truncate(o);
        }
    }
    fn token_bytes(&self) -> Vec<Vec<u8>> {
        let mut t: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        for &(a, b) in &self.merges {
            let mut v = t[a as usize].clone();
            v.extend_from_slice(&t[b as usize]);
            t.push(v);
        }
        t
    }
}

// ---------- streams ----------
fn read_doc(r: &mut impl Read, buf: &mut Vec<u8>) -> bool {
    let mut l = [0u8; 4];
    match r.read_exact(&mut l) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return false,
        Err(e) => panic!("read: {e}"),
    }
    buf.resize(u32::from_le_bytes(l) as usize, 0);
    r.read_exact(buf).expect("short document");
    true
}

// ---------- train ----------
fn train(vocab: u32, min_count: u64) {
    let mut counts: FxMap<Box<[u8]>, u64> = FxMap::default();
    let mut stdin = BufReader::with_capacity(1 << 20, std::io::stdin().lock());
    let (mut docs, mut bytes) = (0u64, 0u64);
    let mut doc = Vec::new();
    while read_doc(&mut stdin, &mut doc) {
        docs += 1;
        bytes += doc.len() as u64;
        pretok(&doc, &mut |c| {
            if let Some(v) = counts.get_mut(c) {
                *v += 1;
            } else {
                counts.insert(c.into(), 1);
            }
        });
        if docs % 200_000 == 0 {
            eprintln!("counted docs={docs} bytes={bytes} unique_chunks={}", counts.len());
        }
    }
    let unique = counts.len();
    let mut words: Vec<(Vec<u32>, u64)> = Vec::new();
    let mut kept_occ = 0u64;
    let mut all_occ = 0u64;
    for (k, v) in counts.drain() {
        all_occ += v;
        if v >= min_count {
            kept_occ += v;
            words.push((k.iter().map(|&b| b as u32).collect(), v));
        }
    }
    drop(counts);
    // deterministic order independent of hash iteration
    words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    eprintln!(
        "docs={docs} bytes={bytes} unique_chunks={unique} kept_chunks={} kept_occurrences={kept_occ}/{all_occ}",
        words.len()
    );

    let mut pc: FxMap<u64, i64> = FxMap::default();
    let mut pw: FxMap<u64, Vec<u32>> = FxMap::default();
    for (wi, (ids, c)) in words.iter().enumerate() {
        for w in ids.windows(2) {
            let k = pk(w[0], w[1]);
            *pc.entry(k).or_insert(0) += *c as i64;
            let l = pw.entry(k).or_default();
            if l.last() != Some(&(wi as u32)) {
                l.push(wi as u32);
            }
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<u64>)> = pc.iter().map(|(&k, &c)| (c, Reverse(k))).collect();
    let nmerge = (vocab - 257) as usize;
    let mut merges: Vec<(u32, u32)> = Vec::with_capacity(nmerge);
    let mut delta: FxMap<u64, i64> = FxMap::default();
    while merges.len() < nmerge {
        let best = loop {
            let Some((c, Reverse(k))) = heap.pop() else { break None };
            let cur = pc.get(&k).copied().unwrap_or(0);
            if cur <= 0 {
                continue;
            }
            if cur != c {
                heap.push((cur, Reverse(k)));
                continue;
            }
            break Some((k, cur));
        };
        let Some((k, cnt)) = best else {
            eprintln!("no pairs left after {} merges", merges.len());
            break;
        };
        let (a, b) = ((k >> 32) as u32, (k & 0xFFFF_FFFF) as u32);
        let new_id = 256 + merges.len() as u32;
        merges.push((a, b));
        let mut list = pw.remove(&k).unwrap_or_default();
        list.sort_unstable();
        list.dedup();
        delta.clear();
        for &wi in &list {
            let (ids, c) = &mut words[wi as usize];
            if !ids.windows(2).any(|w| w[0] == a && w[1] == b) {
                continue;
            }
            for w in ids.windows(2) {
                *delta.entry(pk(w[0], w[1])).or_insert(0) -= *c as i64;
            }
            let mut o = 0;
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    ids[o] = new_id;
                    i += 2;
                } else {
                    ids[o] = ids[i];
                    i += 1;
                }
                o += 1;
            }
            ids.truncate(o);
            for w in ids.windows(2) {
                let kk = pk(w[0], w[1]);
                *delta.entry(kk).or_insert(0) += *c as i64;
                if w[0] == new_id || w[1] == new_id {
                    let l = pw.entry(kk).or_default();
                    if l.last() != Some(&wi) {
                        l.push(wi);
                    }
                }
            }
        }
        for (&kk, &d) in delta.iter() {
            if d == 0 {
                continue;
            }
            let e = pc.entry(kk).or_insert(0);
            *e += d;
            if d > 0 && *e > 0 {
                heap.push((*e, Reverse(kk)));
            }
        }
        pc.remove(&k);
        if merges.len() % 2000 == 0 || merges.len() == nmerge {
            eprintln!("merges={} last_pair_count={cnt}", merges.len());
        }
    }
    let model = Model::new(merges);
    // self-check: the encoder must reproduce the trainer's final segmentation
    let mut tmp = Vec::new();
    let mut bad = 0;
    let bytes_of = model.token_bytes();
    for (ids, _) in words.iter().take(300_000) {
        let mut raw = Vec::new();
        for &t in ids {
            raw.extend_from_slice(&bytes_of[t as usize]);
        }
        model.encode_chunk(&raw, &mut tmp);
        if &tmp != ids {
            bad += 1;
        }
    }
    eprintln!("selfcheck encoder==trainer on top 300000 chunks: mismatches={bad}");
    assert_eq!(bad, 0);
    model.save(&std::env::args().nth(4).expect("out path"));
}

// ---------- encode ----------
const HDR: usize = 64;
fn header(vocab: u32, dtype: u32, eos: u32, tokens: u64, docs: u64, tok_hash16: [u8; 16]) -> [u8; HDR] {
    let mut h = [0u8; HDR];
    h[0..8].copy_from_slice(b"AIENTOK1");
    h[8..12].copy_from_slice(&1u32.to_le_bytes());
    h[12..16].copy_from_slice(&vocab.to_le_bytes());
    h[16..20].copy_from_slice(&dtype.to_le_bytes());
    h[20..24].copy_from_slice(&eos.to_le_bytes());
    h[24..32].copy_from_slice(&tokens.to_le_bytes());
    h[32..40].copy_from_slice(&docs.to_le_bytes());
    h[40..56].copy_from_slice(&tok_hash16);
    h
}

fn encode(args: &[String]) {
    // encode <tok.bpe> <out.tok|-> <threads> <tokfile-sha256-hex16>
    let model = Model::load(&args[0]);
    let out_path = &args[1];
    let threads: usize = args[2].parse().unwrap();
    let mut hash16 = [0u8; 16];
    for i in 0..16 {
        hash16[i] = u8::from_str_radix(&args[3][2 * i..2 * i + 2], 16).unwrap();
    }
    assert!(model.vocab <= 65536, "vocab does not fit uint16");
    let mut out = if out_path == "-" { None } else { Some(BufWriter::with_capacity(1 << 22, std::fs::File::create(out_path).unwrap())) };
    if let Some(o) = out.as_mut() {
        o.write_all(&[0u8; HDR]).unwrap();
    }
    let mut stdin = BufReader::with_capacity(1 << 22, std::io::stdin().lock());
    let mut caches: Vec<FxMap<Vec<u8>, Box<[u16]>>> = (0..threads).map(|_| FxMap::default()).collect();
    let (mut ndocs, mut nbytes, mut ntok) = (0u64, 0u64, 0u64);
    let mut eof = false;
    while !eof {
        let mut batch: Vec<Vec<u8>> = Vec::new();
        let mut bb = 0usize;
        while bb < (48 << 20) {
            let mut d = Vec::new();
            if !read_doc(&mut stdin, &mut d) {
                eof = true;
                break;
            }
            bb += d.len();
            batch.push(d);
        }
        if batch.is_empty() {
            break;
        }
        nbytes += bb as u64;
        ndocs += batch.len() as u64;
        // split into contiguous groups of roughly equal bytes
        let per = bb / threads + 1;
        let mut groups: Vec<&[Vec<u8>]> = Vec::new();
        let (mut s, mut acc) = (0usize, 0usize);
        for (i, d) in batch.iter().enumerate() {
            acc += d.len();
            if acc >= per {
                groups.push(&batch[s..=i]);
                s = i + 1;
                acc = 0;
            }
        }
        if s < batch.len() {
            groups.push(&batch[s..]);
        }
        let eos = model.eos() as u16;
        let results: Vec<Vec<u16>> = std::thread::scope(|sc| {
            let hs: Vec<_> = groups
                .iter()
                .zip(caches.iter_mut())
                .map(|(g, cache)| {
                    let model = &model;
                    sc.spawn(move || {
                        let mut toks: Vec<u16> = Vec::new();
                        let mut tmp = Vec::new();
                        for d in g.iter() {
                            pretok(d, &mut |c| {
                                if let Some(t) = cache.get(c) {
                                    toks.extend_from_slice(t);
                                } else {
                                    model.encode_chunk(c, &mut tmp);
                                    let t: Box<[u16]> = tmp.iter().map(|&x| x as u16).collect();
                                    toks.extend_from_slice(&t);
                                    if cache.len() >= 1_000_000 {
                                        cache.clear();
                                    }
                                    cache.insert(c.to_vec(), t);
                                }
                            });
                            toks.push(eos);
                        }
                        toks
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in results {
            ntok += r.len() as u64;
            if let Some(o) = out.as_mut() {
                let mut b = Vec::with_capacity(r.len() * 2);
                for t in r {
                    b.extend_from_slice(&t.to_le_bytes());
                }
                o.write_all(&b).unwrap();
            }
        }
    }
    if let Some(mut o) = out {
        o.flush().unwrap();
        let mut f = o.into_inner().unwrap();
        f.seek(SeekFrom::Start(0)).unwrap();
        f.write_all(&header(model.vocab, 2, model.eos(), ntok, ndocs, hash16)).unwrap();
    }
    println!("{{\"docs\":{ndocs},\"bytes\":{nbytes},\"tokens\":{ntok},\"bytes_per_token\":{:.5}}}", nbytes as f64 / ntok as f64);
}

fn decode(args: &[String]) {
    // decode <tok.bpe> <in.tok>  -> document stream on stdout
    let model = Model::load(&args[0]);
    let tb = model.token_bytes();
    let mut f = BufReader::with_capacity(1 << 22, std::fs::File::open(&args[1]).unwrap());
    let mut h = [0u8; HDR];
    f.read_exact(&mut h).unwrap();
    assert_eq!(&h[0..8], b"AIENTOK1");
    assert_eq!(u32::from_le_bytes(h[12..16].try_into().unwrap()), model.vocab);
    let eos = model.eos();
    let mut out = BufWriter::with_capacity(1 << 22, std::io::stdout().lock());
    let mut cur: Vec<u8> = Vec::new();
    let mut b = [0u8; 2];
    while f.read_exact(&mut b).is_ok() {
        let t = u16::from_le_bytes(b) as u32;
        if t == eos {
            out.write_all(&(cur.len() as u32).to_le_bytes()).unwrap();
            out.write_all(&cur).unwrap();
            cur.clear();
        } else {
            cur.extend_from_slice(&tb[t as usize]);
        }
    }
    assert!(cur.is_empty(), "token file does not end with EOS");
}

fn truncate(args: &[String]) {
    // truncate <in.bpe> <vocab> <out.bpe>
    let m = Model::load(&args[0]);
    let v: u32 = args[1].parse().unwrap();
    let n = (v - 257) as usize;
    assert!(n <= m.merges.len());
    Model::new(m.merges[..n].to_vec()).save(&args[2]);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        // train <vocab> <min_count> <out.bpe>
        Some("train") => train(a[2].parse().unwrap(), a[3].parse().unwrap()),
        Some("encode") => encode(&a[2..]),
        Some("decode") => decode(&a[2..]),
        Some("truncate") => truncate(&a[2..]),
        _ => eprintln!("usage: aienbpe train <vocab> <min_count> <out.bpe> | encode <tok.bpe> <out.tok|-> <threads> <hash16hex> | decode <tok.bpe> <in.tok> | truncate <in.bpe> <vocab> <out.bpe>"),
    }
}
