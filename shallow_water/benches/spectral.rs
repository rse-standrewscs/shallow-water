use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    shallow_water::spectral::Spectral,
};

macro_rules! _1d_from_file {
    ($name:expr) => {
        include_bytes!($name)
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>()
    };
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "spectral",
        Benchmark::new("main_invert", |b| {
            let spectral = Spectral::new(48, 6);
            let qs = _1d_from_file!("../src/spectral/testdata/main_invert/48_6_qs.bin");
            let ds = _1d_from_file!("../src/spectral/testdata/main_invert/48_6_ds.bin");
            let gs = _1d_from_file!("../src/spectral/testdata/main_invert/48_6_gs.bin");
            let mut r = vec![0.0; 48 * 48 * 7];
            let mut u = vec![0.0; 48 * 48 * 7];
            let mut v = vec![0.0; 48 * 48 * 7];
            let mut zeta = vec![0.0; 48 * 48 * 7];

            b.iter(|| spectral.main_invert(&qs, &ds, &gs, &mut r, &mut u, &mut v, &mut zeta))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("jacob", |b| {
            let spectral = Spectral::new(48, 6);
            let aa = _1d_from_file!("../src/spectral/testdata/jacob/48_6_aa.bin");
            let bb = _1d_from_file!("../src/spectral/testdata/jacob/48_6_bb.bin");
            let mut cs = vec![0.0; 48 * 48];

            b.iter(|| spectral.jacob(&aa, &bb, &mut cs))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("divs", |b| {
            let spectral = Spectral::new(48, 6);
            let aa = _1d_from_file!("../src/spectral/testdata/divs/48_6_aa.bin");
            let bb = _1d_from_file!("../src/spectral/testdata/divs/48_6_bb.bin");
            let mut cs = vec![0.0; 48 * 48];

            b.iter(|| spectral.divs(&aa, &bb, &mut cs))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("ptospc3d", |b| {
            let spectral = Spectral::new(30, 4);
            let fp = _1d_from_file!("../src/spectral/testdata/ptospc3d/30_4_fp.bin");
            let mut fs = _1d_from_file!("../src/spectral/testdata/ptospc3d/30_4_fs.bin");

            b.iter(|| spectral.ptospc3d(&fp, &mut fs, 0, 3))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("spctop3d", |b| {
            let spectral = Spectral::new(30, 4);
            let fs = _1d_from_file!("../src/spectral/testdata/spctop3d/30_4_fs.bin");
            let mut fp = _1d_from_file!("../src/spectral/testdata/spctop3d/30_4_fp.bin");

            b.iter(|| spectral.spctop3d(&fs, &mut fp, 0, 3))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("deal3d", |b| {
            let spectral = Spectral::new(30, 4);
            let mut fp = _1d_from_file!("../src/spectral/testdata/deal3d/30_4_fp.bin");

            b.iter(|| spectral.deal3d(&mut fp))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("deal2d", |b| {
            let spectral = Spectral::new(32, 4);
            let mut fp = _1d_from_file!("../src/spectral/testdata/deal2d/32_4_fp.bin");

            b.iter(|| spectral.deal2d(&mut fp))
        })
        .sample_size(10),
    );
    c.bench(
        "spectral",
        Benchmark::new("spec1d", |b| {
            let spectral = Spectral::new(48, 4);
            let ss = _1d_from_file!("../src/spectral/testdata/spec1d/48_4_ss.bin");
            let mut spec = _1d_from_file!("../src/spectral/testdata/spec1d/48_4_spec.bin");

            b.iter(|| spectral.spec1d(&ss, &mut spec))
        }),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
