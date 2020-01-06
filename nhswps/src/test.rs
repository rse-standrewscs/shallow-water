use {
    crate::*,
    approx::assert_abs_diff_eq,
    byteorder::{ByteOrder, NetworkEndian},
    lazy_static::lazy_static,
};

fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
    for (i, e) in a.iter().enumerate() {
        assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-10);
    }
}

mod source {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let aa = include_bytes!("testdata/source/18_2_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let qs = include_bytes!("testdata/source/18_2_qs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/source/18_2_ds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ps = include_bytes!("testdata/source/18_2_ps.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/source/18_2_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/source/18_2_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ri = include_bytes!("testdata/source/18_2_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let dpn = include_bytes!("testdata/source/18_2_dpn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/source/18_2_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/source/18_2_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/source/18_2_zy.bin")
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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let aa = include_bytes!("testdata/source/32_4_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let qs = include_bytes!("testdata/source/32_4_qs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/source/32_4_ds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ps = include_bytes!("testdata/source/32_4_ps.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/source/32_4_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/source/32_4_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ri = include_bytes!("testdata/source/32_4_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let dpn = include_bytes!("testdata/source/32_4_dpn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/source/32_4_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/source/32_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/source/32_4_zy.bin")
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
    }

    #[test]
    fn _18_2_sqs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/18_2_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _18_2_sds() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/18_2_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _18_2_sgs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/18_2_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sgs2, &sgs);
    }

    #[test]
    fn _32_4_sqs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/32_4_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _32_4_sds() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/32_4_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _32_4_sgs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/32_4_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sgs2, &sgs);
    }
}

mod vertical {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let z = include_bytes!("testdata/vertical/18_2_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/vertical/18_2_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/vertical/18_2_zy.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let r = include_bytes!("testdata/vertical/18_2_r.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/vertical/18_2_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let aa = include_bytes!("testdata/vertical/18_2_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/vertical/18_2_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/vertical/18_2_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/vertical/18_2_ds.bin")
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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let z = include_bytes!("testdata/vertical/32_4_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/vertical/32_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/vertical/32_4_zy.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let r = include_bytes!("testdata/vertical/32_4_r.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/vertical/32_4_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let aa = include_bytes!("testdata/vertical/32_4_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/vertical/32_4_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/vertical/32_4_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/vertical/32_4_ds.bin")
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
    }

    #[test]
    fn _18_2_z() {
        let z2 = include_bytes!("testdata/vertical/18_2_z2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&z2, &state.z);
    }

    #[test]
    fn _18_2_zx() {
        let zx2 = include_bytes!("testdata/vertical/18_2_zx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&zx2, &state.zx);
    }

    #[test]
    fn _18_2_zy() {
        let zy2 = include_bytes!("testdata/vertical/18_2_zy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&zy2, &state.zy);
    }

    #[test]
    fn _18_2_w() {
        let w2 = include_bytes!("testdata/vertical/18_2_w2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&w2, &state.w);
    }

    #[test]
    fn _18_2_aa() {
        let aa2 = include_bytes!("testdata/vertical/18_2_aa2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&aa2, &state.aa);
    }

    #[test]
    fn _32_4() {
        let z2 = include_bytes!("testdata/vertical/32_4_z2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let zx2 = include_bytes!("testdata/vertical/32_4_zx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let zy2 = include_bytes!("testdata/vertical/32_4_zy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let w2 = include_bytes!("testdata/vertical/32_4_w2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let aa2 = include_bytes!("testdata/vertical/32_4_aa2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut state = STATE_32_4.clone();

        vertical(&mut state);

        assert_approx_eq_slice(&z2, &state.z);
        assert_approx_eq_slice(&zx2, &state.zx);
        assert_approx_eq_slice(&zy2, &state.zy);
        assert_approx_eq_slice(&w2, &state.w);
        assert_approx_eq_slice(&aa2, &state.aa);
    }
}

mod coeffs {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let ri = include_bytes!("testdata/coeffs/18_2_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/coeffs/18_2_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/coeffs/18_2_zy.bin")
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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let ri = include_bytes!("testdata/coeffs/32_4_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/coeffs/32_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/coeffs/32_4_zy.bin")
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
    }

    #[test]
    fn _18_2_sigx() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigx2 = include_bytes!("testdata/coeffs/18_2_sigx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigx2, &sigx);
    }

    #[test]
    fn _18_2_sigy() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigy2 = include_bytes!("testdata/coeffs/18_2_sigy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigy2, &sigy);
    }

    #[test]
    fn _18_2_cpt1() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt12 = include_bytes!("testdata/coeffs/18_2_cpt12.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt12, &cpt1);
    }

    #[test]
    fn _18_2_cpt2() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt22 = include_bytes!("testdata/coeffs/18_2_cpt22.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt22, &cpt2);
    }

    #[test]
    fn _32_4_sigx() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigx2 = include_bytes!("testdata/coeffs/32_4_sigx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigx2, &sigx);
    }

    #[test]
    fn _32_4_sigy() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigy2 = include_bytes!("testdata/coeffs/32_4_sigy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigy2, &sigy);
    }

    #[test]
    fn _32_4_cpt1() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt12 = include_bytes!("testdata/coeffs/32_4_cpt12.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt12, &cpt1);
    }

    #[test]
    fn _32_4_cpt2() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt22 = include_bytes!("testdata/coeffs/32_4_cpt22.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt22, &cpt2);
    }
}

mod cpsource {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let ri = include_bytes!("testdata/cpsource/18_2_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/cpsource/18_2_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/cpsource/18_2_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/cpsource/18_2_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/cpsource/18_2_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zeta = include_bytes!("testdata/cpsource/18_2_zeta.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/cpsource/18_2_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/cpsource/18_2_zy.bin")
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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let ri = include_bytes!("testdata/cpsource/32_4_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/cpsource/32_4_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/cpsource/32_4_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/cpsource/32_4_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/cpsource/32_4_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zeta = include_bytes!("testdata/cpsource/32_4_zeta.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/cpsource/32_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/cpsource/32_4_zy.bin")
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
    }

    #[test]
    fn _18_2() {
        let mut sp0 = include_bytes!("testdata/cpsource/18_2_sp0.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sp02 = include_bytes!("testdata/cpsource/18_2_sp02.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        cpsource(&STATE_18_2, &mut sp0);

        assert_approx_eq_slice(&sp02, &sp0);
    }

    #[test]
    fn _32_4() {
        let mut sp0 = include_bytes!("testdata/cpsource/32_4_sp0.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sp02 = include_bytes!("testdata/cpsource/32_4_sp02.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        cpsource(&STATE_32_4, &mut sp0);

        assert_approx_eq_slice(&sp02, &sp0);
    }
}

mod psolve {
    use super::*;

    lazy_static! {
        static ref STATE_24_4: State = {
            let ng = 24;
            let nz = 4;

            let ri = include_bytes!("testdata/psolve/24_4_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let r = include_bytes!("testdata/psolve/24_4_r.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/psolve/24_4_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/psolve/24_4_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/psolve/24_4_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/psolve/24_4_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zeta = include_bytes!("testdata/psolve/24_4_zeta.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/psolve/24_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/psolve/24_4_zy.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ps = include_bytes!("testdata/psolve/24_4_ps.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let pn = include_bytes!("testdata/psolve/24_4_pn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let dpn = include_bytes!("testdata/psolve/24_4_dpn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let aa = include_bytes!("testdata/psolve/24_4_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let qs = include_bytes!("testdata/psolve/24_4_qs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/psolve/24_4_ds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let gs = include_bytes!("testdata/psolve/24_4_gs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut state = State {
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
            };
            psolve(&mut state);
            state
        };
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let ri = include_bytes!("testdata/psolve/32_4_ri.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let r = include_bytes!("testdata/psolve/32_4_r.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let u = include_bytes!("testdata/psolve/32_4_u.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let v = include_bytes!("testdata/psolve/32_4_v.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let w = include_bytes!("testdata/psolve/32_4_w.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let z = include_bytes!("testdata/psolve/32_4_z.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zeta = include_bytes!("testdata/psolve/32_4_zeta.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zx = include_bytes!("testdata/psolve/32_4_zx.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let zy = include_bytes!("testdata/psolve/32_4_zy.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ps = include_bytes!("testdata/psolve/32_4_ps.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let pn = include_bytes!("testdata/psolve/32_4_pn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let dpn = include_bytes!("testdata/psolve/32_4_dpn.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let aa = include_bytes!("testdata/psolve/32_4_aa.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let qs = include_bytes!("testdata/psolve/32_4_qs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let ds = include_bytes!("testdata/psolve/32_4_ds.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            let gs = include_bytes!("testdata/psolve/32_4_gs.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();

            let mut state = State {
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
            };
            psolve(&mut state);
            state
        };
    }

    #[test]
    fn _32_4_z() {
        let z2 = include_bytes!("testdata/psolve/32_4_z2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&z2, &STATE_32_4.z);
    }

    #[test]
    fn _32_4_zx() {
        let zx2 = include_bytes!("testdata/psolve/32_4_zx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&zx2, &STATE_32_4.zx);
    }
    #[test]
    fn _32_4_zy() {
        let zy2 = include_bytes!("testdata/psolve/32_4_zy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&zy2, &STATE_32_4.zy);
    }
    #[test]
    fn _32_4_w() {
        let w2 = include_bytes!("testdata/psolve/32_4_w2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&w2, &STATE_32_4.w);
    }
    #[test]
    fn _32_4_aa() {
        let aa2 = include_bytes!("testdata/psolve/32_4_aa2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&aa2, &STATE_32_4.aa);
    }

    #[test]
    fn _32_4_ri() {
        let ri2 = include_bytes!("testdata/psolve/32_4_ri2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&ri2, &STATE_32_4.ri);
    }
    #[test]
    fn _32_4_pn() {
        let pn2 = include_bytes!("testdata/psolve/32_4_pn2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&pn2, &STATE_32_4.pn);
    }

    #[test]
    fn _32_4_ps() {
        let ps2 = include_bytes!("testdata/psolve/32_4_ps2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&ps2, &STATE_32_4.ps);
    }

    #[test]
    fn _32_4_dpn() {
        let dpn2 = include_bytes!("testdata/psolve/32_4_dpn2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&dpn2, &STATE_32_4.dpn);
    }

    #[test]
    fn _24_4_z() {
        let z2 = include_bytes!("testdata/psolve/24_4_z2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&z2, &STATE_24_4.z);
    }

    #[test]
    fn _24_4_zx() {
        let zx2 = include_bytes!("testdata/psolve/24_4_zx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&zx2, &STATE_24_4.zx);
    }
    #[test]
    fn _24_4_zy() {
        let zy2 = include_bytes!("testdata/psolve/24_4_zy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&zy2, &STATE_24_4.zy);
    }
    #[test]
    fn _24_4_w() {
        let w2 = include_bytes!("testdata/psolve/24_4_w2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&w2, &STATE_24_4.w);
    }
    #[test]
    fn _24_4_aa() {
        let aa2 = include_bytes!("testdata/psolve/24_4_aa2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&aa2, &STATE_24_4.aa);
    }

    #[test]
    fn _24_4_ri() {
        let ri2 = include_bytes!("testdata/psolve/24_4_ri2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&ri2, &STATE_24_4.ri);
    }
    #[test]
    fn _24_4_pn() {
        let pn2 = include_bytes!("testdata/psolve/24_4_pn2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&pn2, &STATE_24_4.pn);
    }

    #[test]
    fn _24_4_ps() {
        let ps2 = include_bytes!("testdata/psolve/24_4_ps2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&ps2, &STATE_24_4.ps);
    }

    #[test]
    fn _24_4_dpn() {
        let dpn2 = include_bytes!("testdata/psolve/24_4_dpn2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        assert_approx_eq_slice(&dpn2, &STATE_24_4.dpn);
    }
}

mod advance {
    use super::*;

    mod _18_6 {
        use super::*;

        lazy_static! {
            static ref STATE: State = {
                let ng = 18;
                let nz = 6;

                let ri = include_bytes!("testdata/advance/18_6_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let r = include_bytes!("testdata/advance/18_6_r.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("testdata/advance/18_6_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("testdata/advance/18_6_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("testdata/advance/18_6_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("testdata/advance/18_6_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zeta = include_bytes!("testdata/advance/18_6_zeta.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("testdata/advance/18_6_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("testdata/advance/18_6_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ps = include_bytes!("testdata/advance/18_6_ps.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let pn = include_bytes!("testdata/advance/18_6_pn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let dpn = include_bytes!("testdata/advance/18_6_dpn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let aa = include_bytes!("testdata/advance/18_6_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let qs = include_bytes!("testdata/advance/18_6_qs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("testdata/advance/18_6_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let gs = include_bytes!("testdata/advance/18_6_gs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                let mut state = State {
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
                    t: 5.555_555_555_555_555E-2,
                    ngsave: 5,
                    itime: 1,
                    jtime: 0,
                    ggen: true,
                    output: Output::default(),
                };
                advance(&mut state);
                state
            };
        }
        #[test]
        fn r() {
            let r2 = include_bytes!("testdata/advance/18_6_r2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&r2, &STATE.r);
        }

        #[test]
        fn ri() {
            let ri2 = include_bytes!("testdata/advance/18_6_ri2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ri2, &STATE.ri);
        }

        #[test]
        fn u() {
            let u2 = include_bytes!("testdata/advance/18_6_u2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&u2, &STATE.u);
        }

        #[test]
        fn v() {
            let v2 = include_bytes!("testdata/advance/18_6_v2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&v2, &STATE.v);
        }

        #[test]
        fn w() {
            let w2 = include_bytes!("testdata/advance/18_6_w2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&w2, &STATE.w);
        }
        #[test]
        fn z() {
            let z2 = include_bytes!("testdata/advance/18_6_z2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&z2, &STATE.z);
        }

        #[test]
        fn zeta() {
            let zeta2 = include_bytes!("testdata/advance/18_6_zeta2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zeta2, &STATE.zeta);
        }

        #[test]
        fn zx() {
            let zx2 = include_bytes!("testdata/advance/18_6_zx2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zx2, &STATE.zx);
        }

        #[test]
        fn zy() {
            let zy2 = include_bytes!("testdata/advance/18_6_zy2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zy2, &STATE.zy);
        }

        #[test]
        fn ps() {
            let ps2 = include_bytes!("testdata/advance/18_6_ps2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ps2, &STATE.ps);
        }

        #[test]
        fn pn() {
            let pn2 = include_bytes!("testdata/advance/18_6_pn2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&pn2, &STATE.pn);
        }

        #[test]
        fn dpn() {
            let dpn2 = include_bytes!("testdata/advance/18_6_dpn2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&dpn2, &STATE.dpn);
        }

        #[test]
        fn aa() {
            let aa2 = include_bytes!("testdata/advance/18_6_aa2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&aa2, &STATE.aa);
        }

        #[test]
        fn qs() {
            let qs2 = include_bytes!("testdata/advance/18_6_qs2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&qs2, &STATE.qs);
        }

        #[test]
        fn ds() {
            let ds2 = include_bytes!("testdata/advance/18_6_ds2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ds2, &STATE.ds);
        }

        #[test]
        fn gs() {
            let gs2 = include_bytes!("testdata/advance/18_6_gs2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&gs2, &STATE.gs);
        }

        #[test]
        fn t() {
            assert_abs_diff_eq!(0.111_111_111_111_111_1, STATE.t);
        }
    }

    mod _24_4 {
        use super::*;

        lazy_static! {
            static ref STATE: State = {
                let ng = 24;
                let nz = 4;

                let ri = include_bytes!("testdata/advance/24_4_ri.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let r = include_bytes!("testdata/advance/24_4_r.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let u = include_bytes!("testdata/advance/24_4_u.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let v = include_bytes!("testdata/advance/24_4_v.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let w = include_bytes!("testdata/advance/24_4_w.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let z = include_bytes!("testdata/advance/24_4_z.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zeta = include_bytes!("testdata/advance/24_4_zeta.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zx = include_bytes!("testdata/advance/24_4_zx.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let zy = include_bytes!("testdata/advance/24_4_zy.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ps = include_bytes!("testdata/advance/24_4_ps.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let pn = include_bytes!("testdata/advance/24_4_pn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let dpn = include_bytes!("testdata/advance/24_4_dpn.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let aa = include_bytes!("testdata/advance/24_4_aa.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let qs = include_bytes!("testdata/advance/24_4_qs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let ds = include_bytes!("testdata/advance/24_4_ds.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();
                let gs = include_bytes!("testdata/advance/24_4_gs.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_f64)
                    .collect::<Vec<f64>>();

                let mut state = State {
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
                };
                advance(&mut state);
                state
            };
        }
        #[test]
        fn r() {
            let r2 = include_bytes!("testdata/advance/24_4_r2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&r2, &STATE.r);
        }

        #[test]
        fn ri() {
            let ri2 = include_bytes!("testdata/advance/24_4_ri2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ri2, &STATE.ri);
        }

        #[test]
        fn u() {
            let u2 = include_bytes!("testdata/advance/24_4_u2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&u2, &STATE.u);
        }

        #[test]
        fn v() {
            let v2 = include_bytes!("testdata/advance/24_4_v2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&v2, &STATE.v);
        }

        #[test]
        fn w() {
            let w2 = include_bytes!("testdata/advance/24_4_w2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&w2, &STATE.w);
        }
        #[test]
        fn z() {
            let z2 = include_bytes!("testdata/advance/24_4_z2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&z2, &STATE.z);
        }

        #[test]
        fn zeta() {
            let zeta2 = include_bytes!("testdata/advance/24_4_zeta2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zeta2, &STATE.zeta);
        }

        #[test]
        fn zx() {
            let zx2 = include_bytes!("testdata/advance/24_4_zx2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zx2, &STATE.zx);
        }

        #[test]
        fn zy() {
            let zy2 = include_bytes!("testdata/advance/24_4_zy2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&zy2, &STATE.zy);
        }

        #[test]
        fn ps() {
            let ps2 = include_bytes!("testdata/advance/24_4_ps2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ps2, &STATE.ps);
        }

        #[test]
        fn pn() {
            let pn2 = include_bytes!("testdata/advance/24_4_pn2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&pn2, &STATE.pn);
        }

        #[test]
        fn dpn() {
            let dpn2 = include_bytes!("testdata/advance/24_4_dpn2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&dpn2, &STATE.dpn);
        }

        #[test]
        fn aa() {
            let aa2 = include_bytes!("testdata/advance/24_4_aa2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&aa2, &STATE.aa);
        }

        #[test]
        fn qs() {
            let qs2 = include_bytes!("testdata/advance/24_4_qs2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&qs2, &STATE.qs);
        }

        #[test]
        fn ds() {
            let ds2 = include_bytes!("testdata/advance/24_4_ds2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&ds2, &STATE.ds);
        }

        #[test]
        fn gs() {
            let gs2 = include_bytes!("testdata/advance/24_4_gs2.bin")
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>();
            assert_approx_eq_slice(&gs2, &STATE.gs);
        }
    }
}
