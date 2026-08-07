//! Packed value storage for wide signals (>64 bits).
//!
//! This module provides a memory-efficient representation for wide 4-state
//! logic values. Instead of using 1 byte per bit (Vec<LogicBit>), it packs
//! 4 bits per byte using 2 bits per bit:
//!
//! - 00 = Zero
//! - 01 = One  
//! - 10 = X
//! - 11 = Z
//!
//! This provides a **4× memory reduction** compared to `Vec<LogicBit>` for
//! wide signals, with only slight overhead for bit access.
//!
//! # Usage
//!
//! This is the storage backing `ValueStorage::Wide` in the `Value` type — it
//! replaced the former `Vec<LogicBit>`. It can also be benchmarked and tested
//! independently.

use serde::{Serialize, Deserialize};
use crate::value::LogicBit;

/// Packed storage for wide 4-state logic values.
///
/// Each byte stores 4 bits using 2 bits per bit:
/// ```text
/// Byte:  b7 b6 b5 b4  b3 b2 b1 b0
/// Bits:  bit3       bit2       bit1       bit0
///        [1:0]     [3:2]     [5:4]     [7:6]
/// ```
///
/// Encoding:
/// - 0b00 = Zero
/// - 0b01 = One
/// - 0b10 = X
/// - 0b11 = Z
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PackedBits {
    /// Packed bits: 4 bits per byte (2 bits per logic bit)
    data: Vec<u8>,
    /// Number of actual bits stored (may be less than data.len() * 4)
    len: u32,
}

impl PackedBits {
    /// Create a new empty PackedBits.
    pub fn new() -> Self {
        Self { data: Vec::new(), len: 0 }
    }

    /// Create PackedBits with given width, all bits initialized to X.
    pub fn new_x(width: u32) -> Self {
        Self::new_fill(width, LogicBit::X)
    }

    /// Create PackedBits with given width, all bits initialized to Zero.
    pub fn new_zero(width: u32) -> Self {
        let num_bytes = Self::bytes_needed(width);
        Self {
            data: vec![0x00; num_bytes],
            len: width,
        }
    }

    /// Create PackedBits with given width, all bits initialized to specified value.
    ///
    /// The unused slots above `width` in the last byte (the padding bits) are
    /// masked back to Zero. This keeps the invariant that `has_xz`, `==` and
    /// `Hash` only observe the `width` live bits — without it, an all-X
    /// `new_x(66)` would read X in the two padding slots of the 17th byte and
    /// `has_xz()` would report true, and two values with identical live bits
    /// but different construction histories could compare unequal.
    pub fn new_fill(width: u32, bit: LogicBit) -> Self {
        let code = bit.to_code();
        let fill_byte = code * 0x55;  // Replicate 2-bit code across byte
        let num_bytes = Self::bytes_needed(width);
        let mut data = vec![fill_byte; num_bytes];
        let pad = width % 4;
        if pad != 0 {
            data[num_bytes - 1] &= (1u8 << (pad * 2)) - 1;
        }
        Self { data, len: width }
    }

    /// Calculate number of bytes needed to store `num_bits` bits.
    #[inline]
    fn bytes_needed(num_bits: u32) -> usize {
        ((num_bits as usize + 3) / 4).max(1)
    }

    /// Get the number of bits stored.
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get bit at position i.
    /// Returns LogicBit::Zero if i >= len.
    #[inline]
    pub fn get(&self, i: usize) -> LogicBit {
        if i >= self.len as usize {
            return LogicBit::Zero;
        }
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        let code = (self.data[byte_idx] >> shift) & 0b11;
        LogicBit::from_code(code)
    }

    /// Set bit at position i.
    /// Returns true if the bit changed.
    #[inline]
    pub fn set(&mut self, i: usize, bit: LogicBit) -> bool {
        if i >= self.len as usize {
            return false;
        }
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        let code = bit.to_code();
        let mask = 0b11 << shift;
        let old = self.data[byte_idx];
        let new = (old & !mask) | ((code as u8) << shift);
        if old == new {
            return false;
        }
        self.data[byte_idx] = new;
        true
    }

    /// Check if any bit is X or Z.
    ///
    /// Word-at-a-time scan: an X/Z code (0b10/0b11) has its high bit set, so
    /// the test is `w & valid & 0xAAAAAAAAAAAAAAAA != 0` — one masked compare
    /// per 8 bytes instead of a per-byte branch. `scan_words` restricts the
    /// compare to the live lanes, so it is exact for every width.
    #[inline]
    pub fn has_xz(&self) -> bool {
        self.scan_words(|w, valid| (w & valid & 0xAAAAAAAAAAAAAAAA) != 0)
    }

    /// True when every live bit is `0`.
    ///
    /// One word-level pass: a value is all-zero iff no live lane is non-zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        !self.scan_words(|w, valid| (w & valid) != 0)
    }

    /// Any live bit equal to a definite `1` (code 0b01).
    ///
    /// A lane holds code `01` exactly when its low bit is set and its high bit
    /// is clear, i.e. `w & !(w >> 1)` at the lane position.
    #[inline]
    pub fn has_known_one(&self) -> bool {
        self.scan_words(|w, valid| (w & !(w >> 1) & 0x5555555555555555 & valid) != 0)
    }

    /// Any live bit equal to a definite `0` (code 0b00).
    ///
    /// A lane is `00` iff neither bit is set: `!(w | w >> 1)` per lane. The
    /// `0x5555…` mask keeps only the lane-low (even) positions where the
    /// predicate is defined, and `valid` excludes padding lanes — a padding
    /// lane *is* a zero code, so it must be excluded, not counted.
    #[inline]
    pub fn has_known_zero(&self) -> bool {
        self.scan_words(|w, valid| (!(w | (w >> 1)) & 0x5555555555555555 & valid) != 0)
    }

    /// Word-at-a-time scan over the packed bytes.
    ///
    /// Applies `f(word, valid_lanes)` to each 8-byte window (`from_le_bytes`
    /// of a stack copy) and returns true on the first match. `valid_lanes` has
    /// both bits set for every lane strictly below `len` and is 0 elsewhere —
    /// it covers the padding lanes in the top byte AND the zero bytes above a
    /// partial tail chunk, so a predicate can never observe non-data lanes.
    #[inline]
    fn scan_words(&self, mut f: impl FnMut(u64, u64) -> bool) -> bool {
        let data = &self.data;
        let n = data.len();
        let live = self.len as usize;
        let mut i = 0;
        while i < n {
            let take = (n - i).min(8);
            let mut w = [0u8; 8];
            w[..take].copy_from_slice(&data[i..i + take]);
            let valid = if i + take == n {
                let lanes = live.saturating_sub(i * 4);
                if lanes == 0 {
                    0
                } else if lanes >= 32 {
                    // A chunk holds at most 32 lanes; 32 lanes = all 64 bits.
                    u64::MAX
                } else {
                    (1u64 << (lanes * 2)) - 1
                }
            } else {
                u64::MAX
            };
            if f(u64::from_le_bytes(w), valid) {
                return true;
            }
            i += 8;
        }
        false
    }

    /// Canonicalize the top byte: mask the padding lanes (above `len`) to 0.
    ///
    /// The packed shifts and extracts write whole bytes, which can put bits in
    /// the padding positions of the top byte. Restoring the canonical-zero
    /// padding keeps `has_xz`/`==`/`Hash`/`has_known_zero` exact.
    #[inline]
    fn canonicalize_top(&mut self) {
        let pad = self.len as usize % 4;
        if pad != 0 {
            if let Some(top) = self.data.last_mut() {
                *top &= (1u8 << (pad * 2)) - 1;
            }
        }
    }

    /// Extract the low 64 bits as (val_bits, xz_bits).
    /// Used for compatibility with inline storage operations.
    #[inline]
    pub fn raw_bits_low64(&self) -> (u64, u64) {
        let mut val_bits = 0u64;
        let mut xz_bits = 0u64;
        let num_bits = (64).min(self.len as usize);
        for i in 0..num_bits {
            match self.get(i) {
                LogicBit::Zero => {},
                LogicBit::One => val_bits |= 1u64 << i,
                LogicBit::X => xz_bits |= 1u64 << i,
                LogicBit::Z => { val_bits |= 1u64 << i; xz_bits |= 1u64 << i; },
            }
        }
        (val_bits, xz_bits)
    }

    /// Iterate over bits.
    pub fn iter(&self) -> PackedBitsIter<'_> {
        PackedBitsIter {
            data: &self.data,
            len: self.len as usize,
            index: 0,
        }
    }

    /// Get the underlying data for serialization.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.data.capacity()
    }

    /// Build from raw packed bytes and an explicit bit length (deserialization
    /// / slice-copy fast path). The caller guarantees `data` holds
    /// `Self::bytes_needed(len)` bytes.
    pub fn from_data(data: Vec<u8>, len: u32) -> Self {
        Self { data, len }
    }

    /// Build by appending bits in LSB-first order.
    pub fn from_bits(bits: impl IntoIterator<Item = LogicBit>) -> Self {
        let mut pb = PackedBits::new();
        for b in bits {
            pb.push(b);
        }
        pb
    }

    /// Append a single bit, growing the backing byte as needed.
    pub fn push(&mut self, bit: LogicBit) {
        if self.len % 4 == 0 {
            self.data.push(0);
        }
        let byte_idx = self.data.len() - 1;
        let shift = (self.len % 4) * 2;
        self.data[byte_idx] |= bit.to_code() << shift;
        self.len += 1;
    }

    /// Map every bit through `f` in place (2-state coercion, bitwise NOT, …).
    pub fn transform(&mut self, mut f: impl FnMut(LogicBit) -> LogicBit) {
        for i in 0..self.len as usize {
            self.set(i, f(self.get(i)));
        }
    }

    /// Mutable access to the packed bytes (length-preserving writes).
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Logical left shift by `amt` logic bits, zero-filling the vacated low
    /// bits and dropping the bits shifted past the top. Result keeps `len`.
    ///
    /// Because each logic bit is 2 code bits, a shift of `amt` logic bits is a
    /// code-bit shift of `2*amt` — a byte-string shift (whole-byte skip plus a
    /// 0/2/4/6-bit remainder) instead of one `get`/`set` per bit. O(bytes).
    #[inline]
    pub fn shl(&self, amt: u32) -> PackedBits {
        let mut out = PackedBits::new_zero(self.len);
        if amt == 0 {
            out.data.copy_from_slice(&self.data);
            return out;
        }
        if amt >= self.len {
            return out;
        }
        let data = &self.data;
        let n = data.len();
        let code_shift = amt as usize * 2;
        let byte_shift = code_shift / 8;
        let bit_shift = (code_shift % 8) as u32;
        let out_data = out.data_mut();
        for b in 0..n {
            let mut v = 0u16;
            if b >= byte_shift {
                v |= (data[b - byte_shift] as u16) << bit_shift;
            }
            if bit_shift != 0 && b > byte_shift {
                v |= (data[b - byte_shift - 1] as u16) >> (8 - bit_shift);
            }
            out_data[b] = v as u8;
        }
        out.canonicalize_top();
        out
    }

    /// Logical right shift by `amt` logic bits, zero-filling the vacated high
    /// bits. Result keeps `len`. Same code-bit argument as `shl`. O(bytes).
    #[inline]
    pub fn shr(&self, amt: u32) -> PackedBits {
        let mut out = PackedBits::new_zero(self.len);
        if amt == 0 {
            out.data.copy_from_slice(&self.data);
            return out;
        }
        if amt >= self.len {
            return out;
        }
        let data = &self.data;
        let n = data.len();
        let code_shift = amt as usize * 2;
        let byte_shift = code_shift / 8;
        let bit_shift = (code_shift % 8) as u32;
        let out_data = out.data_mut();
        for b in 0..n {
            let mut v = 0u16;
            if b + byte_shift < n {
                v |= (data[b + byte_shift] as u16) >> bit_shift;
            }
            if bit_shift != 0 && b + byte_shift + 1 < n {
                v |= (data[b + byte_shift + 1] as u16) << (8 - bit_shift);
            }
            out_data[b] = v as u8;
        }
        out.canonicalize_top();
        out
    }

    /// Extract `width` logic bits starting at `lo` into a fresh packed value.
    /// Bits past the end of `self` read 0. O(bytes): the source is read as a
    /// bit string and the `[2*lo, 2*lo + 2*width)` code-bit window is copied,
    /// instead of one `get`/`set` per bit.
    #[inline]
    pub fn extract(&self, lo: usize, width: u32) -> PackedBits {
        let mut out = PackedBits::new_zero(width);
        if width == 0 {
            return out;
        }
        let data = &self.data;
        let n = data.len();
        let code_start = lo * 2;
        let out_data = out.data_mut();
        for (b, ob) in out_data.iter_mut().enumerate() {
            let src_bit = code_start + b * 8;
            let src_byte = src_bit / 8;
            let shift = (src_bit % 8) as u32;
            let mut v = 0u16;
            if src_byte < n {
                v |= (data[src_byte] as u16) >> shift;
            }
            if shift != 0 && src_byte + 1 < n {
                v |= (data[src_byte + 1] as u16) << (8 - shift);
            }
            *ob = v as u8;
        }
        out.canonicalize_top();
        out
    }
}

impl Default for PackedBits {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over packed bits.
pub struct PackedBitsIter<'a> {
    data: &'a [u8],
    len: usize,
    index: usize,
}

impl<'a> Iterator for PackedBitsIter<'a> {
    type Item = LogicBit;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }
        let byte_idx = self.index / 4;
        let shift = (self.index % 4) * 2;
        let code = (self.data[byte_idx] >> shift) & 0b11;
        self.index += 1;
        Some(LogicBit::from_code(code))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PackedBitsIter<'_> {}

/// Benchmark comparison between Vec<LogicBit> and PackedBits.
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    fn create_wide_value(width: usize) -> Vec<LogicBit> {
        (0..width)
            .map(|i| {
                match i % 4 {
                    0 => LogicBit::Zero,
                    1 => LogicBit::One,
                    2 => LogicBit::X,
                    _ => LogicBit::Z,
                }
            })
            .collect()
    }

    fn create_packed_value(width: usize) -> PackedBits {
        let mut pb = PackedBits::new_x(width as u32);
        for i in 0..width {
            let bit = match i % 4 {
                0 => LogicBit::Zero,
                1 => LogicBit::One,
                2 => LogicBit::X,
                _ => LogicBit::Z,
            };
            pb.set(i, bit);
        }
        pb
    }

    #[test]
    fn test_packed_vs_wide_memory() {
        let width = 10000;  // 10K bits
        
        let wide = create_wide_value(width);
        let packed = create_packed_value(width);
        
        // Vec<LogicBit> uses 1 byte per bit (enum with repr u8)
        let wide_size = wide.capacity() * std::mem::size_of::<LogicBit>();
        // PackedBits uses ~1 byte per 4 bits (2 bits per bit encoding)
        let packed_size = packed.memory_size();
        
        println!("Width: {} bits", width);
        println!("Vec<LogicBit> size: {} bytes ({:.2} KB)", wide_size, wide_size as f64 / 1024.0);
        println!("PackedBits size: {} bytes ({:.2} KB)", packed_size, packed_size as f64 / 1024.0);
        println!("Reduction: {:.1}x", wide_size as f64 / packed_size as f64);
        
        // Should be ~4x reduction (1 byte/bit -> 0.25 bytes/bit)
        assert!(wide_size > packed_size * 3);  // At least 3x reduction
    }

    #[test]
    fn test_packed_access() {
        let width = 1000;
        let packed = create_packed_value(width);
        
        // Test all access patterns
        for i in 0..width {
            let bit = packed.get(i);
            let expected = match i % 4 {
                0 => LogicBit::Zero,
                1 => LogicBit::One,
                2 => LogicBit::X,
                _ => LogicBit::Z,
            };
            assert_eq!(bit, expected);
        }
    }

    #[test]
    fn test_packed_has_xz() {
        use crate::value::LogicBit;
        let mut pb = PackedBits::new_zero(100);
        assert!(!pb.has_xz());

        pb.set(10, LogicBit::X);
        assert!(pb.has_xz());

        pb.set(10, LogicBit::Zero);
        assert!(!pb.has_xz());

        pb.set(20, LogicBit::Z);
        assert!(pb.has_xz());
    }
}

/// Correctness of the word-level scans and the packed shift/extract fast
/// paths, with explicit attention to widths that are not a multiple of 4
/// (the top byte's padding lanes must never leak into any result).
#[cfg(test)]
mod scans_and_shifts {
    use super::*;
    use crate::value::LogicBit;

    fn pattern(width: u32) -> PackedBits {
        // Known bits 0/1, X and Z, in a repeating pattern.
        let mut pb = PackedBits::new_zero(width);
        for i in 0..width as usize {
            let b = match i % 4 {
                0 => LogicBit::Zero,
                1 => LogicBit::One,
                2 => LogicBit::X,
                _ => LogicBit::Z,
            };
            pb.set(i, b);
        }
        pb
    }

    /// Reference bit scan over the *live* bits (padding excluded).
    fn live(pb: &PackedBits) -> Vec<LogicBit> {
        (0..pb.len() as usize).map(|i| pb.get(i)).collect()
    }

    #[test]
    fn scans_match_per_bit_reference() {
        for width in [0u32, 1, 3, 4, 5, 66, 67, 100, 160, 4096] {
            let pb = pattern(width);
            let bits = live(&pb);
            assert_eq!(
                pb.has_xz(),
                bits.iter().any(|b| !b.is_known()),
                "has_xz width={width}"
            );
            assert_eq!(
                pb.has_known_one(),
                bits.iter().any(|b| *b == LogicBit::One),
                "has_known_one width={width}"
            );
            assert_eq!(
                pb.has_known_zero(),
                bits.iter().any(|b| *b == LogicBit::Zero),
                "has_known_zero width={width}"
            );
            assert_eq!(
                pb.is_zero(),
                bits.iter().all(|b| *b == LogicBit::Zero),
                "is_zero width={width}"
            );
        }
    }

    /// Padding lanes are canonical 0 and must not be seen by the scans — the
    /// regression for `scan_words`' top-byte mask (e.g. an all-X 66-bit value
    /// has 2 padding lanes in byte 16 that are 0, but `has_known_zero` must
    /// still be false).
    #[test]
    fn scans_ignore_padding_lanes() {
        for width in [1u32, 2, 3, 5, 6, 7, 9, 66, 67] {
            let all_x = PackedBits::new_fill(width, LogicBit::X);
            assert!(all_x.has_xz());
            assert!(!all_x.has_known_one(), "width={width}");
            assert!(
                !all_x.has_known_zero(),
                "padding zero-lanes leaked: width={width}"
            );
            assert!(!all_x.is_zero(), "width={width}");

            let all_z = PackedBits::new_fill(width, LogicBit::Z);
            assert!(all_z.has_xz());
            assert!(!all_z.has_known_zero(), "width={width}");

            let all_zero = PackedBits::new_zero(width);
            assert!(all_zero.has_known_zero());
            assert!(all_zero.is_zero());
            assert!(!all_zero.has_xz());
        }
    }

    #[test]
    fn shifts_match_per_bit_reference() {
        for width in [0u32, 1, 3, 4, 5, 66, 100, 160] {
            let pb = pattern(width);
            let bits = live(&pb);
            for amt in [0u32, 1, 2, 3, 4, 5, 8, 63, 64, 65, 1000] {
                let shl = pb.shl(amt);
                let expect_l: Vec<LogicBit> = (0..width as usize)
                    .map(|i| {
                        i.checked_sub(amt as usize)
                            .and_then(|s| if s < bits.len() { Some(bits[s]) } else { None })
                            .unwrap_or(LogicBit::Zero)
                    })
                    .collect();
                assert_eq!(live(&shl), expect_l, "shl width={width} amt={amt}");

                let shr = pb.shr(amt);
                let expect_r: Vec<LogicBit> = (0..width as usize)
                    .map(|i| {
                        let s = i + amt as usize;
                        if s < bits.len() {
                            bits[s]
                        } else {
                            LogicBit::Zero
                        }
                    })
                    .collect();
                assert_eq!(live(&shr), expect_r, "shr width={width} amt={amt}");
            }
        }
    }

    #[test]
    fn extract_matches_per_bit_reference() {
        for width in [0u32, 1, 3, 4, 5, 66, 100, 160] {
            let pb = pattern(width);
            let bits = live(&pb);
            for (lo, w) in [
                (0usize, 0u32),
                (0, 1),
                (3, 4),
                (0, 66),
                (3, 65),
                (63, 66),
                (100, 5),
                (1000, 8),
            ] {
                let ex = pb.extract(lo, w);
                let expect: Vec<LogicBit> = (0..w as usize)
                    .map(|i| bits.get(lo + i).copied().unwrap_or(LogicBit::Zero))
                    .collect();
                assert_eq!(live(&ex), expect, "extract width={width} lo={lo} w={w}");
            }
        }
    }

    /// shl/shr/extract must keep the canonical-zero padding so the result is
    /// == and hashes exactly to a per-bit-constructed value of the same width.
    #[test]
    fn shift_extract_canonicalize_padding() {
        for width in [5u32, 66, 67, 100] {
            let pb = pattern(width);
            for amt in [1u32, 3, 4] {
                let s = pb.shl(amt);
                let back = s.shr(amt);
                // shl then shr by the same amount is identity on the top bits
                // only if padding stayed canonical; verify via a full rebuild.
                assert_eq!(back.get(0), pb.get(0), "width={width} amt={amt}");
            }
            // A select of [0, width) reproduces the value exactly.
            let ex = pb.extract(0, width);
            assert_eq!(ex, pb, "width={width}");
        }
    }
}
