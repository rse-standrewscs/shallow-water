use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    shallow_water::stafft::{forfft, revfft},
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "stafft",
        Benchmark::new("forfft", |b| {
            let mut x12 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng12_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig12 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng12_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x15 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng15_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig15 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng15_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x24 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng24_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig24 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng24_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                forfft(12, 12, &mut x12, &trig12, &[1, 0, 1, 0, 0]);
                forfft(15, 15, &mut x15, &trig15, &[0, 0, 0, 1, 1]);
                forfft(24, 24, &mut x24, &trig24, &[1, 1, 0, 0, 0]);
            })
        }),
    );

    c.bench(
        "stafft",
        Benchmark::new("revfft", |b| {
            let mut x12 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng12_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig12 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng12_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x15 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng15_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig15 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng15_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x24 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng24_1_x.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let trig24 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng24_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                revfft(12, 12, &mut x12, &trig12, &[1, 0, 1, 0, 0]);
                revfft(15, 15, &mut x15, &trig15, &[0, 0, 0, 1, 1]);
                revfft(24, 24, &mut x24, &trig24, &[1, 1, 0, 0, 0]);
            })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
