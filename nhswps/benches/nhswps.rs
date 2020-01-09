use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    libnhswps::{advance, coeffs, cpsource, psolve, source, vertical, Output, State},
    ndarray::{Array3, ShapeBuilder},
    shallow_water::spectral::Spectral,
};

macro_rules! array3_from_file {
    ($x:expr, $y:expr, $z:expr, $name:expr) => {
        Array3::from_shape_vec(
            ($x, $y, $z).strides((1, $x, $x * $y)),
            include_bytes!($name)
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
    };
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "nhswps",
        Benchmark::new("source", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let aa = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_aa.bin");
                let qs = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_qs.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_ds.bin");
                let ps = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_ps.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_v.bin");
                let ri = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_ri.bin");
                let dpn = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_dpn.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_z.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/source/32_4_zy.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    z,
                    zx,
                    zy,
                    r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ri,
                    aa,
                    zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    dpn,
                    ps,
                    qs,
                    ds,
                    gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    t: 0.0,
                    ngsave: 0,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };
            let mut sqs = include_bytes!("../src/testdata/source/32_4_sqs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut sds = include_bytes!("../src/testdata/source/32_4_sds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut sgs = include_bytes!("../src/testdata/source/32_4_sgs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| source(&state, &mut sqs, &mut sds, &mut sgs))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("vertical", |b| {
            let mut state = {
                let ng = 32;
                let nz = 4;

                let z = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_z.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_zy.bin");
                let r = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_r.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_w.bin");
                let aa = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_aa.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_v.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "../src/testdata/vertical/32_4_ds.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r,
                    ri: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    aa,
                    zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    dpn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ps: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    qs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ds,
                    gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    t: 0.0,
                    ngsave: 0,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };
            b.iter(|| vertical(&mut state))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("coeffs", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let ri = array3_from_file!(ng, ng, nz + 1, "../src/testdata/coeffs/32_4_ri.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/coeffs/32_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/coeffs/32_4_zy.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    v: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    z: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    zx,
                    zy,
                    r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ri,
                    aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    dpn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ps: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    qs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ds: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    t: 0.0,
                    ngsave: 0,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };
            let mut sigx = include_bytes!("../src/testdata/coeffs/32_4_sigx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut sigy = include_bytes!("../src/testdata/coeffs/32_4_sigy.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut cpt1 = include_bytes!("../src/testdata/coeffs/32_4_cpt1.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut cpt2 = include_bytes!("../src/testdata/coeffs/32_4_cpt2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| coeffs(&state, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("cpsource", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let ri = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_ri.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_v.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_w.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_z.bin");
                let zeta =
                    array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_zeta.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/cpsource/32_4_zy.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ri,
                    aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    zeta,
                    pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    dpn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ps: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    qs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    ds: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                    t: 0.0,
                    ngsave: 0,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };
            let mut sp0 = include_bytes!("../src/testdata/cpsource/32_4_sp0.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| cpsource(&state, &mut sp0))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("psolve", |b| {
            let mut state = {
                let ng = 32;
                let nz = 4;

                let ri = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_ri.bin");
                let r = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_r.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_v.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_w.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_z.bin");
                let zeta =
                    array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_zeta.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_zy.bin");
                let ps = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_ps.bin");
                let pn = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_pn.bin");
                let dpn = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_dpn.bin");
                let aa = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_aa.bin");
                let qs = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_qs.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_ds.bin");
                let gs = array3_from_file!(ng, ng, nz + 1, "../src/testdata/psolve/32_4_gs.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r,
                    ri,
                    aa,
                    zeta,
                    pn,
                    dpn,
                    ps,
                    qs,
                    ds,
                    gs,
                    t: 0.0,
                    ngsave: 0,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };

            b.iter(|| psolve(&mut state))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("advance", |b| {
            let mut state = {
                let ng = 24;
                let nz = 4;

                let ri = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_ri.bin");
                let r = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_r.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_v.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_w.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_z.bin");
                let zeta =
                    array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_zeta.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_zy.bin");
                let ps = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_ps.bin");
                let pn = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_pn.bin");
                let dpn = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_dpn.bin");
                let aa = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_aa.bin");
                let qs = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_qs.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_ds.bin");
                let gs = array3_from_file!(ng, ng, nz + 1, "../src/testdata/advance/24_4_gs.bin");

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r,
                    ri,
                    aa,
                    zeta,
                    pn,
                    dpn,
                    ps,
                    qs,
                    ds,
                    gs,
                    t: 0.624_999_999_999_999_9,
                    ngsave: 6,
                    itime: 15,
                    jtime: 2,
                    ggen: true,
                    output: Output::default(),
                }
            };
            b.iter(|| advance(&mut state))
        })
        .sample_size(10),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
