use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Array2, Array3, ShapeBuilder},
    shallow_water::{
        array2_from_file, array3_from_file,
        spectral::Spectral,
        utils::{arr2zero, arr3zero},
    },
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
            let qs = array3_from_file!(
                48,
                48,
                7,
                "../src/spectral/testdata/main_invert/48_6_qs.bin"
            );
            let ds = array3_from_file!(
                48,
                48,
                7,
                "../src/spectral/testdata/main_invert/48_6_ds.bin"
            );
            let gs = array3_from_file!(
                48,
                48,
                7,
                "../src/spectral/testdata/main_invert/48_6_gs.bin"
            );
            let mut r = arr3zero(48, 6);
            let mut u = r.clone();
            let mut v = r.clone();
            let mut zeta = r.clone();

            b.iter(|| {
                spectral.main_invert(
                    qs.view(),
                    ds.view(),
                    gs.view(),
                    r.view_mut(),
                    u.view_mut(),
                    v.view_mut(),
                    zeta.view_mut(),
                )
            })
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("jacob", |b| {
            let spectral = Spectral::new(48, 6);
            let aa = array2_from_file!(48, 48, "../src/spectral/testdata/jacob/48_6_aa.bin");
            let bb = array2_from_file!(48, 48, "../src/spectral/testdata/jacob/48_6_bb.bin");
            let mut cs = arr2zero(48);

            b.iter(|| spectral.jacob(aa.view(), bb.view(), cs.view_mut()))
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("divs", |b| {
            let spectral = Spectral::new(48, 6);
            let aa = array2_from_file!(48, 48, "../src/spectral/testdata/divs/48_6_aa.bin");
            let bb = array2_from_file!(48, 48, "../src/spectral/testdata/divs/48_6_bb.bin");
            let mut cs = arr2zero(48);

            b.iter(|| spectral.divs(aa.view(), bb.view(), cs.view_mut()))
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("ptospc3d", |b| {
            let spectral = Spectral::new(30, 4);
            let fp = _1d_from_file!("../src/spectral/testdata/ptospc3d/30_4_fp.bin");
            let mut fs = _1d_from_file!("../src/spectral/testdata/ptospc3d/30_4_fs.bin");

            b.iter(|| spectral.ptospc3d(&fp, &mut fs, 0, 3))
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("spctop3d", |b| {
            let spectral = Spectral::new(30, 4);
            let fs = _1d_from_file!("../src/spectral/testdata/spctop3d/30_4_fs.bin");
            let mut fp = _1d_from_file!("../src/spectral/testdata/spctop3d/30_4_fp.bin");

            b.iter(|| spectral.spctop3d(&fs, &mut fp, 0, 3))
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("deal3d", |b| {
            let spectral = Spectral::new(30, 4);
            let mut fp =
                array3_from_file!(30, 30, 5, "../src/spectral/testdata/deal3d/30_4_fp.bin");

            b.iter(|| spectral.deal3d(fp.view_mut()))
        }),
    );
    c.bench(
        "spectral",
        Benchmark::new("deal2d", |b| {
            let spectral = Spectral::new(32, 4);
            let mut fp = array2_from_file!(32, 32, "../src/spectral/testdata/deal2d/32_4_fp.bin");

            b.iter(|| spectral.deal2d(fp.view_mut()))
        }),
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
