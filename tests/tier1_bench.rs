//! Microbenchmark for the Tier1 wide-op fast paths. Run:
//!   cargo test --release --test tier1_bench -- --ignored --nocapture
//! Prints ns/op for each op on a 4096-bit value; compare across commits.
//!
//! Reference (before → after, same machine, back-to-back; parent of the perf
//! commit vs the perf commit). Absolute values drift with machine load; the
//! ratio is the signal:
//!   has_xz(all-zero)   333 → ~100 ns   (3.3×)
//!   bitwise_and/xor   27670 → ~3600    (7.7×)
//!   shift_left/right ~16750/14260 → ~1700/1600  (~9×)
//!   range_select      5492 → ~157     (35×)
//!   reduce_or         19.8 → ~11.5    (1.7×)
use xezim_core::value::Value;
use std::time::Instant;

fn bench(name: &str, iters: u32, mut f: impl FnMut() -> Value) {
    // warmup
    for _ in 0..iters / 10 { std::hint::black_box(f()); }
    let t = Instant::now();
    for _ in 0..iters { std::hint::black_box(f()); }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    println!("{name:<28} {ns:>10.1} ns/op");
}

#[test]
#[ignore]
fn tier1_wide_bench() {
    let iters = 200_000u32;
    let w4096 = Value::from_str_radix(&"f0xz".repeat(1024), 16, 4096);
    let zero = Value::zero(4096);
    let ones = Value::ones(4096);
    let amt = Value::from_u64(13, 32);

    bench("wide4096.has_xz(no-xz)", iters, || Value::from_u64(Value::zero(4096).has_xz() as u64, 1));
    bench("wide4096.has_xz(mixed)", iters, || Value::from_u64(w4096.has_xz() as u64, 1));
    bench("wide4096.is_nonzero", iters, || Value::from_u64(matches!(w4096.is_nonzero(), Some(true)) as u64, 1));
    bench("wide4096.reduce_and", iters, || w4096.reduce_and());
    bench("wide4096.reduce_or", iters, || w4096.reduce_or());
    bench("wide4096.bitwise_and", iters, || w4096.bitwise_and(&ones));
    bench("wide4096.bitwise_xor", iters, || w4096.bitwise_xor(&w4096));
    bench("wide4096.shift_left(13)", iters, || w4096.shift_left(&amt));
    bench("wide4096.shift_right(13)", iters, || w4096.shift_right(&amt));
    bench("wide4096.range_select(3000,1000)", iters, || w4096.range_select(3000, 1000));
    bench("wide4096.eq(all-zero)", iters, || Value::from_u64((w4096 == zero) as u64, 1));
}
