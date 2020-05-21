use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Array2, ShapeBuilder},
    shallow_water::{
        array2_from_file,
        stafft::{forfft, revfft},
    },
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "stafft",
        Benchmark::new("forfft", |b| {
            let mut x12 =
                array2_from_file!(12, 12, "../src/stafft/testdata/forfft/forfft_ng12_1_x.bin");
            let trig12 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng12_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x15 =
                array2_from_file!(15, 15, "../src/stafft/testdata/forfft/forfft_ng15_1_x.bin");
            let trig15 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng15_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x24 =
                array2_from_file!(24, 24, "../src/stafft/testdata/forfft/forfft_ng24_1_x.bin");
            let trig24 = include_bytes!("../src/stafft/testdata/forfft/forfft_ng24_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                forfft(12, 12, x12.view_mut(), &trig12, &[1, 0, 1, 0, 0]);
                forfft(15, 15, x15.view_mut(), &trig15, &[0, 0, 0, 1, 1]);
                forfft(24, 24, x24.view_mut(), &trig24, &[1, 1, 0, 0, 0]);
            })
        }),
    );

    c.bench(
        "stafft",
        Benchmark::new("revfft", |b| {
            let mut x12 =
                array2_from_file!(12, 12, "../src/stafft/testdata/revfft/revfft_ng12_1_x.bin");
            let trig12 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng12_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x15 =
                array2_from_file!(15, 15, "../src/stafft/testdata/revfft/revfft_ng15_1_x.bin");
            let trig15 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng15_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut x24 =
                array2_from_file!(24, 24, "../src/stafft/testdata/revfft/revfft_ng24_1_x.bin");
            let trig24 = include_bytes!("../src/stafft/testdata/revfft/revfft_ng24_1_trig.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                revfft(12, 12, x12.view_mut(), &trig12, &[1, 0, 1, 0, 0]);
                revfft(15, 15, x15.view_mut(), &trig15, &[0, 0, 0, 1, 1]);
                revfft(24, 24, x24.view_mut(), &trig24, &[1, 1, 0, 0, 0]);
            })
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
