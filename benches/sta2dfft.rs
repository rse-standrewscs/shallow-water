use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    shallow_water::sta2dfft::D2FFT,
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

            let d2fft = D2FFT {
                nx: 32,
                ny: 32,
                xfactors: [0, 2, 1, 0, 0],
                yfactors: [0, 2, 1, 0, 0],
                xtrig: trig.clone(),
                ytrig: trig,
            };

            b.iter(|| {
                d2fft.ptospc(&mut rvar, &mut svar);
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

            let d2fft = D2FFT {
                nx: 32,
                ny: 32,
                xfactors: [0, 2, 1, 0, 0],
                yfactors: [0, 2, 1, 0, 0],
                xtrig: trig.clone(),
                ytrig: trig,
            };

            b.iter(|| {
                d2fft.spctop(&mut rvar, &mut svar);
            })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
