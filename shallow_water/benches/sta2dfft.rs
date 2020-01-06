use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    shallow_water::sta2dfft::{ptospc, spctop},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "sta2dfft",
        Benchmark::new("ptospc", |b| {
            let mut rvar = include_bytes!("../src/testdata/ptospc/ptospc_ng32_rvar.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut svar = include_bytes!("../src/testdata/ptospc/ptospc_ng32_svar.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig = include_bytes!("../src/testdata/ptospc/ptospc_ng32_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                ptospc(
                    32,
                    32,
                    &mut rvar,
                    &mut svar,
                    &[0, 2, 1, 0, 0],
                    &[0, 2, 1, 0, 0],
                    &trig,
                    &trig,
                );
            })
        }),
    );

    c.bench(
        "sta2dfft",
        Benchmark::new("spctop", |b| {
            let mut rvar = include_bytes!("../src/testdata/spctop/spctop_ng32_rvar.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut svar = include_bytes!("../src/testdata/spctop/spctop_ng32_svar.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig = include_bytes!("../src/testdata/spctop/spctop_ng32_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                spctop(
                    32,
                    32,
                    &mut rvar,
                    &mut svar,
                    &[0, 2, 1, 0, 0],
                    &[0, 2, 1, 0, 0],
                    &trig,
                    &trig,
                );
            })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
