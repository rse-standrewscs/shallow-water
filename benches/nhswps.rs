use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    ndarray::{Array3, ShapeBuilder},
    shallow_water::{
        array3_from_file,
        nhswps::{
            advance::advance, coeffs::coeffs, cpsource::cpsource, psolve::psolve, savegrid,
            source::source, vertical::vertical, Output, State,
        },
        spectral::Spectral,
        utils::*,
    },
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "nhswps",
        Benchmark::new("source", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let aa =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_aa.bin");
                let qs =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_qs.bin");
                let ds =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_ds.bin");
                let ps =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_ps.bin");
                let u =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_u.bin");
                let v =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_v.bin");
                let ri =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_ri.bin");
                let dpn =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_dpn.bin");
                let z =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_z.bin");
                let zx =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_zx.bin");
                let zy =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/source/32_4_zy.bin");

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
            let mut sqs = include_bytes!("../src/nhswps/testdata/source/32_4_sqs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut sds = include_bytes!("../src/nhswps/testdata/source/32_4_sds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let mut sgs = include_bytes!("../src/nhswps/testdata/source/32_4_sgs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| {
                source(
                    &state,
                    viewmut3d(&mut sqs, 32, 32, 5),
                    viewmut3d(&mut sds, 32, 32, 5),
                    viewmut3d(&mut sgs, 32, 32, 5),
                )
            })
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("vertical", |b| {
            let mut state = {
                let ng = 32;
                let nz = 4;

                let z =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/vertical/32_4_z.bin");
                let zx = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/vertical/32_4_zx.bin"
                );
                let zy = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/vertical/32_4_zy.bin"
                );
                let r =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/vertical/32_4_r.bin");
                let w =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/vertical/32_4_w.bin");
                let aa = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/vertical/32_4_aa.bin"
                );
                let u =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/vertical/32_4_u.bin");
                let v =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/vertical/32_4_v.bin");
                let ds = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/vertical/32_4_ds.bin"
                );

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
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("coeffs", |b| {
            let ng = 32;
            let nz = 4;

            let state = {
                let ri =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/coeffs/32_4_ri.bin");
                let zx =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/coeffs/32_4_zx.bin");
                let zy =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/coeffs/32_4_zy.bin");

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

            let mut sigx = array3_from_file!(
                ng,
                ng,
                nz + 1,
                "../src/nhswps/testdata/coeffs/32_4_sigx.bin"
            );
            let mut sigy = array3_from_file!(
                ng,
                ng,
                nz + 1,
                "../src/nhswps/testdata/coeffs/32_4_sigy.bin"
            );
            let mut cpt1 = array3_from_file!(
                ng,
                ng,
                nz + 1,
                "../src/nhswps/testdata/coeffs/32_4_cpt1.bin"
            );
            let mut cpt2 = array3_from_file!(
                ng,
                ng,
                nz + 1,
                "../src/nhswps/testdata/coeffs/32_4_cpt2.bin"
            );

            b.iter(|| {
                coeffs(
                    &state,
                    sigx.view_mut(),
                    sigy.view_mut(),
                    cpt1.view_mut(),
                    cpt2.view_mut(),
                )
            })
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("cpsource", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let ri = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/cpsource/32_4_ri.bin"
                );
                let u =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/cpsource/32_4_u.bin");
                let v =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/cpsource/32_4_v.bin");
                let w =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/cpsource/32_4_w.bin");
                let z =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/cpsource/32_4_z.bin");
                let zeta = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/cpsource/32_4_zeta.bin"
                );
                let zx = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/cpsource/32_4_zx.bin"
                );
                let zy = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/cpsource/32_4_zy.bin"
                );

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
            let mut sp0 = include_bytes!("../src/nhswps/testdata/cpsource/32_4_sp0.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            b.iter(|| cpsource(&state, viewmut3d(&mut sp0, 32, 32, 5)))
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("psolve", |b| {
            let mut state = {
                let ng = 32;
                let nz = 4;

                let ri =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_ri.bin");
                let r =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_r.bin");
                let u =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_u.bin");
                let v =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_v.bin");
                let w =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_w.bin");
                let z =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_z.bin");
                let zeta = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/psolve/32_4_zeta.bin"
                );
                let zx =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_zx.bin");
                let zy =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_zy.bin");
                let ps =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_ps.bin");
                let pn =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_pn.bin");
                let dpn =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_dpn.bin");
                let aa =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_aa.bin");
                let qs =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_qs.bin");
                let ds =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_ds.bin");
                let gs =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/psolve/32_4_gs.bin");

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
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("advance", |b| {
            let mut state = {
                let ng = 24;
                let nz = 4;

                let ri =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/ri.bin");
                let r =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/r.bin");
                let u =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/u.bin");
                let v =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/v.bin");
                let w =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/w.bin");
                let z =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/z.bin");
                let zeta = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/24_4/zeta.bin"
                );
                let zx =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/zx.bin");
                let zy =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/zy.bin");
                let ps =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/ps.bin");
                let pn =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/pn.bin");
                let dpn = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/24_4/dpn.bin"
                );
                let aa =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/aa.bin");
                let qs =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/qs.bin");
                let ds =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/ds.bin");
                let gs =
                    array3_from_file!(ng, ng, nz + 1, "../src/nhswps/testdata/advance/24_4/gs.bin");

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
        }),
    );

    c.bench(
        "nhswps",
        Benchmark::new("advance_128", |b| {
            let mut state = {
                let ng = 128;
                let nz = 16;

                let ri = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/ri.bin"
                );
                let r = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/r.bin"
                );
                let u = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/u.bin"
                );
                let v = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/v.bin"
                );
                let w = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/w.bin"
                );
                let z = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/z.bin"
                );
                let zeta = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/zeta.bin"
                );
                let zx = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/zx.bin"
                );
                let zy = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/zy.bin"
                );
                let ps = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/ps.bin"
                );
                let pn = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/pn.bin"
                );
                let dpn = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/dpn.bin"
                );
                let aa = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/aa.bin"
                );
                let qs = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/qs.bin"
                );
                let ds = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/ds.bin"
                );
                let gs = array3_from_file!(
                    ng,
                    ng,
                    nz + 1,
                    "../src/nhswps/testdata/advance/128_16/gs.bin"
                );

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
                    ngsave: 32,
                    itime: 0,
                    jtime: 0,
                    ggen: false,
                    output: Output::default(),
                }
            };
            b.iter(|| advance(&mut state))
        })
        .sample_size(10),
    );

    c.bench(
        "nhswps",
        Benchmark::new("savegrid", |b| {
            let mut state: State =
                bincode::deserialize(include_bytes!("../src/nhswps/testdata/savegrid_state.bin"))
                    .unwrap();

            state.defragment();

            b.iter(|| savegrid(&mut state))
        })
        .sample_size(10),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
