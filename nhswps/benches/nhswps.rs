use {
    byteorder::{ByteOrder, NetworkEndian},
    criterion::{criterion_group, criterion_main, Benchmark, Criterion},
    libnhswps::{advance, coeffs, cpsource, psolve, source, vertical, Output, State},
    shallow_water::spectral::Spectral,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "nhswps",
        Benchmark::new("source", |b| {
            let state = {
                let ng = 32;
                let nz = 4;

                let aa = include_bytes!("../src/testdata/source/32_4_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let qs = include_bytes!("../src/testdata/source/32_4_qs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("../src/testdata/source/32_4_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ps = include_bytes!("../src/testdata/source/32_4_ps.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("../src/testdata/source/32_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("../src/testdata/source/32_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ri = include_bytes!("../src/testdata/source/32_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let dpn = include_bytes!("../src/testdata/source/32_4_dpn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("../src/testdata/source/32_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/source/32_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/source/32_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w: vec![0.0; ng * ng * (nz + 1)],
                    z,
                    zx,
                    zy,
                    r: vec![0.0; ng * ng * (nz + 1)],
                    ri,
                    aa,
                    zeta: vec![0.0; ng * ng * (nz + 1)],
                    pn: vec![0.0; ng * ng * (nz + 1)],
                    dpn,
                    ps,
                    qs,
                    ds,
                    gs: vec![0.0; ng * ng * (nz + 1)],
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

                let z = include_bytes!("../src/testdata/vertical/32_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/vertical/32_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/vertical/32_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let r = include_bytes!("../src/testdata/vertical/32_4_r.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("../src/testdata/vertical/32_4_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let aa = include_bytes!("../src/testdata/vertical/32_4_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("../src/testdata/vertical/32_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("../src/testdata/vertical/32_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("../src/testdata/vertical/32_4_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r,
                    ri: vec![0.0; ng * ng * (nz + 1)],
                    aa,
                    zeta: vec![0.0; ng * ng * (nz + 1)],
                    pn: vec![0.0; ng * ng * (nz + 1)],
                    dpn: vec![0.0; ng * ng * (nz + 1)],
                    ps: vec![0.0; ng * ng * (nz + 1)],
                    qs: vec![0.0; ng * ng * (nz + 1)],
                    ds,
                    gs: vec![0.0; ng * ng * (nz + 1)],
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

                let ri = include_bytes!("../src/testdata/coeffs/32_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/coeffs/32_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/coeffs/32_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                State {
                    spectral: Spectral::new(ng, nz),
                    u: vec![0.0; ng * ng * (nz + 1)],
                    v: vec![0.0; ng * ng * (nz + 1)],
                    w: vec![0.0; ng * ng * (nz + 1)],
                    z: vec![0.0; ng * ng * (nz + 1)],
                    zx,
                    zy,
                    r: vec![0.0; ng * ng * (nz + 1)],
                    ri,
                    aa: vec![0.0; ng * ng * (nz + 1)],
                    zeta: vec![0.0; ng * ng * (nz + 1)],
                    pn: vec![0.0; ng * ng * (nz + 1)],
                    dpn: vec![0.0; ng * ng * (nz + 1)],
                    ps: vec![0.0; ng * ng * (nz + 1)],
                    qs: vec![0.0; ng * ng * (nz + 1)],
                    ds: vec![0.0; ng * ng * (nz + 1)],
                    gs: vec![0.0; ng * ng * (nz + 1)],
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

                let ri = include_bytes!("../src/testdata/cpsource/32_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("../src/testdata/cpsource/32_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("../src/testdata/cpsource/32_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("../src/testdata/cpsource/32_4_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("../src/testdata/cpsource/32_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zeta = include_bytes!("../src/testdata/cpsource/32_4_zeta.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/cpsource/32_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/cpsource/32_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                State {
                    spectral: Spectral::new(ng, nz),
                    u,
                    v,
                    w,
                    z,
                    zx,
                    zy,
                    r: vec![0.0; ng * ng * (nz + 1)],
                    ri,
                    aa: vec![0.0; ng * ng * (nz + 1)],
                    zeta,
                    pn: vec![0.0; ng * ng * (nz + 1)],
                    dpn: vec![0.0; ng * ng * (nz + 1)],
                    ps: vec![0.0; ng * ng * (nz + 1)],
                    qs: vec![0.0; ng * ng * (nz + 1)],
                    ds: vec![0.0; ng * ng * (nz + 1)],
                    gs: vec![0.0; ng * ng * (nz + 1)],
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

                let ri = include_bytes!("../src/testdata/psolve/32_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let r = include_bytes!("../src/testdata/psolve/32_4_r.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("../src/testdata/psolve/32_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("../src/testdata/psolve/32_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("../src/testdata/psolve/32_4_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("../src/testdata/psolve/32_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zeta = include_bytes!("../src/testdata/psolve/32_4_zeta.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/psolve/32_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/psolve/32_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ps = include_bytes!("../src/testdata/psolve/32_4_ps.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let pn = include_bytes!("../src/testdata/psolve/32_4_pn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let dpn = include_bytes!("../src/testdata/psolve/32_4_dpn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let aa = include_bytes!("../src/testdata/psolve/32_4_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let qs = include_bytes!("../src/testdata/psolve/32_4_qs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("../src/testdata/psolve/32_4_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let gs = include_bytes!("../src/testdata/psolve/32_4_gs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

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

                let ri = include_bytes!("../src/testdata/advance/24_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let r = include_bytes!("../src/testdata/advance/24_4_r.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("../src/testdata/advance/24_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("../src/testdata/advance/24_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("../src/testdata/advance/24_4_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("../src/testdata/advance/24_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zeta = include_bytes!("../src/testdata/advance/24_4_zeta.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("../src/testdata/advance/24_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("../src/testdata/advance/24_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ps = include_bytes!("../src/testdata/advance/24_4_ps.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let pn = include_bytes!("../src/testdata/advance/24_4_pn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let dpn = include_bytes!("../src/testdata/advance/24_4_dpn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let aa = include_bytes!("../src/testdata/advance/24_4_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let qs = include_bytes!("../src/testdata/advance/24_4_qs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("../src/testdata/advance/24_4_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let gs = include_bytes!("../src/testdata/advance/24_4_gs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

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
