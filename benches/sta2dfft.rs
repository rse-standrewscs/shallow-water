use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Array2, ShapeBuilder},
    shallow_water::{array2_from_file, sta2dfft::D2FFT},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "sta2dfft",
        Benchmark::new("ptospc", |b| {
            let mut rvar = array2_from_file!(32, 32, "../src/testdata/ptospc/ptospc_ng32_rvar.bin");
            let mut svar = array2_from_file!(32, 32, "../src/testdata/ptospc/ptospc_ng32_svar.bin");
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
                d2fft.ptospc(rvar.view_mut(), svar.view_mut());
            })
        }),
    );

    c.bench(
        "sta2dfft",
        Benchmark::new("spctop", |b| {
            let mut rvar = array2_from_file!(32, 32, "../src/testdata/spctop/spctop_ng32_rvar.bin");
            let mut svar = array2_from_file!(32, 32, "../src/testdata/spctop/spctop_ng32_svar.bin");
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
                d2fft.spctop(svar.view_mut(), rvar.view_mut());
            })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
