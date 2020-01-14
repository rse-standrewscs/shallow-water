use {
    crate::*,
    approx::assert_abs_diff_eq,
    byteorder::{ByteOrder, NetworkEndian},
    lazy_static::lazy_static,
    ndarray::ShapeBuilder,
};

fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
    for (i, e) in a.iter().enumerate() {
        assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-10);
    }
}

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

mod source {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zy.bin");

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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zy.bin");

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

            let z = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_zy.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_r.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_w.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_aa.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_v.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_ds.bin");

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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let z = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_zy.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_r.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_w.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_aa.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_v.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_ds.bin");
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
    }

    #[test]
    fn _18_2_z() {
        let z2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_z2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(z2, state.z, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_zx() {
        let zx2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_zx2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(zx2, state.zx, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_zy() {
        let zy2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_zy2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(zy2, state.zy, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_w() {
        let w2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_w2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(w2, state.w, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_aa() {
        let aa2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_aa2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(&aa2, &state.aa, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4() {
        let z2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_z2.bin");
        let zx2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_zx2.bin");
        let zy2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_zy2.bin");
        let w2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_w2.bin");
        let aa2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_aa2.bin");

        let mut state = STATE_32_4.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(z2, state.z, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zx2, state.zx, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zy2, state.zy, epsilon = 1.0E-10);
        assert_abs_diff_eq!(w2, state.w, epsilon = 1.0E-10);
        assert_abs_diff_eq!(aa2, state.aa, epsilon = 1.0E-10);
    }
}

mod coeffs {
    use super::*;

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_ri.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_zy.bin");

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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_ri.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_zy.bin");

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
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_ri.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zy.bin");

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
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_ri.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zy.bin");

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

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_ri.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_r.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_zy.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_ps.bin");
            let pn = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_pn.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_dpn.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_ds.bin");
            let gs = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/24_4_gs.bin");

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

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_ri.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_r.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_zy.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_ps.bin");
            let pn = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_pn.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_dpn.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_ds.bin");
            let gs = array3_from_file!(ng, ng, nz + 1, "testdata/psolve/32_4_gs.bin");

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
        let z2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_z2.bin");
        assert_abs_diff_eq!(z2, STATE_32_4.z, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_zx() {
        let zx2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_zx2.bin");
        assert_abs_diff_eq!(zx2, STATE_32_4.zx, epsilon = 1.0E-10);
    }
    #[test]
    fn _32_4_zy() {
        let zy2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_zy2.bin");
        assert_abs_diff_eq!(zy2, STATE_32_4.zy, epsilon = 1.0E-10);
    }
    #[test]
    fn _32_4_w() {
        let w2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_w2.bin");
        assert_abs_diff_eq!(&w2, &STATE_32_4.w, epsilon = 1.0E-10);
    }
    #[test]
    fn _32_4_aa() {
        let aa2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_aa2.bin");
        assert_abs_diff_eq!(&aa2, &STATE_32_4.aa, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_ri() {
        let ri2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_ri2.bin");
        assert_abs_diff_eq!(&ri2, &STATE_32_4.ri, epsilon = 1.0E-10, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_pn() {
        let pn2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_pn2.bin");
        assert_abs_diff_eq!(&pn2, &STATE_32_4.pn, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_ps() {
        let ps2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_ps2.bin");
        assert_abs_diff_eq!(&ps2, &STATE_32_4.ps, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_dpn() {
        let dpn2 = array3_from_file!(32, 32, 5, "testdata/psolve/32_4_dpn2.bin");
        assert_abs_diff_eq!(&dpn2, &STATE_32_4.dpn, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_z() {
        let z2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_z2.bin");
        assert_abs_diff_eq!(z2, STATE_24_4.z, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_zx() {
        let zx2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_zx2.bin");
        assert_abs_diff_eq!(zx2, STATE_24_4.zx, epsilon = 1.0E-10);
    }
    #[test]
    fn _24_4_zy() {
        let zy2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_zy2.bin");
        assert_abs_diff_eq!(zy2, STATE_24_4.zy, epsilon = 1.0E-10);
    }
    #[test]
    fn _24_4_w() {
        let w2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_w2.bin");
        assert_abs_diff_eq!(&w2, &STATE_24_4.w, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_aa() {
        let aa2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_aa2.bin");
        assert_abs_diff_eq!(&aa2, &STATE_24_4.aa, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_ri() {
        let ri2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_ri2.bin");
        assert_abs_diff_eq!(&ri2, &STATE_24_4.ri, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_pn() {
        let pn2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_pn2.bin");
        assert_abs_diff_eq!(&pn2, &STATE_24_4.pn, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_ps() {
        let ps2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_ps2.bin");
        assert_abs_diff_eq!(&ps2, &STATE_24_4.ps, epsilon = 1.0E-10);
    }

    #[test]
    fn _24_4_dpn() {
        let dpn2 = array3_from_file!(24, 24, 5, "testdata/psolve/24_4_dpn2.bin");
        assert_abs_diff_eq!(&dpn2, &STATE_24_4.dpn, epsilon = 1.0E-10);
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

                let ri = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/ri.bin");
                let r = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/r.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/v.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/w.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/z.bin");
                let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/zeta.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/zy.bin");
                let ps = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/ps.bin");
                let pn = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/pn.bin");
                let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/dpn.bin");
                let aa = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/aa.bin");
                let qs = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/qs.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/ds.bin");
                let gs = array3_from_file!(ng, ng, nz + 1, "testdata/advance/18_6/gs.bin");

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
            let r2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/r2.bin");
            assert_abs_diff_eq!(&r2, &STATE.r, epsilon = 1.0E-10);
        }

        #[test]
        fn ri() {
            let ri2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/ri2.bin");
            assert_abs_diff_eq!(&ri2, &STATE.ri, epsilon = 1.0E-10);
        }

        #[test]
        fn u() {
            let u2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/u2.bin");
            assert_abs_diff_eq!(&u2, &STATE.u, epsilon = 1.0E-10);
        }

        #[test]
        fn v() {
            let v2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/v2.bin");
            assert_abs_diff_eq!(&v2, &STATE.v, epsilon = 1.0E-10);
        }

        #[test]
        fn w() {
            let w2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/w2.bin");
            assert_abs_diff_eq!(&w2, &STATE.w, epsilon = 1.0E-10);
        }
        #[test]
        fn z() {
            let z2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/z2.bin");
            assert_abs_diff_eq!(&z2, &STATE.z, epsilon = 1.0E-10);
        }

        #[test]
        fn zeta() {
            let zeta2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/zeta2.bin");
            assert_abs_diff_eq!(&zeta2, &STATE.zeta, epsilon = 1.0E-10);
        }

        #[test]
        fn zx() {
            let zx2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/zx2.bin");
            assert_abs_diff_eq!(&zx2, &STATE.zx, epsilon = 1.0E-10);
        }

        #[test]
        fn zy() {
            let zy2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/zy2.bin");
            assert_abs_diff_eq!(&zy2, &STATE.zy, epsilon = 1.0E-10);
        }

        #[test]
        fn ps() {
            let ps2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/ps2.bin");
            assert_abs_diff_eq!(&ps2, &STATE.ps, epsilon = 1.0E-10);
        }

        #[test]
        fn pn() {
            let pn2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/pn2.bin");
            assert_abs_diff_eq!(&pn2, &STATE.pn, epsilon = 1.0E-10);
        }

        #[test]
        fn dpn() {
            let dpn2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/dpn2.bin");
            assert_abs_diff_eq!(&dpn2, &STATE.dpn, epsilon = 1.0E-10);
        }

        #[test]
        fn aa() {
            let aa2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/aa2.bin");
            assert_abs_diff_eq!(&aa2, &STATE.aa, epsilon = 1.0E-10);
        }

        #[test]
        fn qs() {
            let qs2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/qs2.bin");
            assert_abs_diff_eq!(&qs2, &STATE.qs, epsilon = 1.0E-10);
        }

        #[test]
        fn ds() {
            let ds2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/ds2.bin");
            assert_abs_diff_eq!(&ds2, &STATE.ds, epsilon = 1.0E-10);
        }

        #[test]
        fn gs() {
            let gs2 = array3_from_file!(18, 18, 7, "testdata/advance/18_6/gs2.bin");
            assert_abs_diff_eq!(&gs2, &STATE.gs, epsilon = 1.0E-10);
        }

        #[test]
        fn t() {
            assert_abs_diff_eq!(0.111_111_111_111_111_1, STATE.t, epsilon = 1.0E-10);
        }
    }

    mod _24_4 {
        use super::*;

        lazy_static! {
            static ref STATE: State = {
                let ng = 24;
                let nz = 4;

                let ri = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/ri.bin");
                let r = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/r.bin");
                let u = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/u.bin");
                let v = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/v.bin");
                let w = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/w.bin");
                let z = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/z.bin");
                let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/zeta.bin");
                let zx = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/zx.bin");
                let zy = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/zy.bin");
                let ps = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/ps.bin");
                let pn = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/pn.bin");
                let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/dpn.bin");
                let aa = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/aa.bin");
                let qs = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/qs.bin");
                let ds = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/ds.bin");
                let gs = array3_from_file!(ng, ng, nz + 1, "testdata/advance/24_4/gs.bin");

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
            let r2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/r2.bin");
            assert_abs_diff_eq!(&r2, &STATE.r, epsilon = 1.0E-10);
        }

        #[test]
        fn ri() {
            let ri2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/ri2.bin");
            assert_abs_diff_eq!(&ri2, &STATE.ri, epsilon = 1.0E-10);
        }

        #[test]
        fn u() {
            let u2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/u2.bin");
            assert_abs_diff_eq!(&u2, &STATE.u, epsilon = 1.0E-10);
        }

        #[test]
        fn v() {
            let v2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/v2.bin");
            assert_abs_diff_eq!(&v2, &STATE.v, epsilon = 1.0E-10);
        }

        #[test]
        fn w() {
            let w2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/w2.bin");
            assert_abs_diff_eq!(&w2, &STATE.w, epsilon = 1.0E-10);
        }
        #[test]
        fn z() {
            let z2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/z2.bin");
            assert_abs_diff_eq!(&z2, &STATE.z, epsilon = 1.0E-10);
        }

        #[test]
        fn zeta() {
            let zeta2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/zeta2.bin");
            assert_abs_diff_eq!(&zeta2, &STATE.zeta, epsilon = 1.0E-10);
        }

        #[test]
        fn zx() {
            let zx2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/zx2.bin");
            assert_abs_diff_eq!(&zx2, &STATE.zx, epsilon = 1.0E-10);
        }

        #[test]
        fn zy() {
            let zy2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/zy2.bin");
            assert_abs_diff_eq!(&zy2, &STATE.zy, epsilon = 1.0E-10);
        }

        #[test]
        fn ps() {
            let ps2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/ps2.bin");
            assert_abs_diff_eq!(&ps2, &STATE.ps, epsilon = 1.0E-10);
        }

        #[test]
        fn pn() {
            let pn2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/pn2.bin");
            assert_abs_diff_eq!(&pn2, &STATE.pn, epsilon = 1.0E-10);
        }

        #[test]
        fn dpn() {
            let dpn2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/dpn2.bin");
            assert_abs_diff_eq!(&dpn2, &STATE.dpn, epsilon = 1.0E-10);
        }

        #[test]
        fn aa() {
            let aa2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/aa2.bin");
            assert_abs_diff_eq!(&aa2, &STATE.aa, epsilon = 1.0E-10);
        }

        #[test]
        fn qs() {
            let qs2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/qs2.bin");
            assert_abs_diff_eq!(&qs2, &STATE.qs, epsilon = 1.0E-10);
        }

        #[test]
        fn ds() {
            let ds2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/ds2.bin");
            assert_abs_diff_eq!(&ds2, &STATE.ds, epsilon = 1.0E-10);
        }

        #[test]
        fn gs() {
            let gs2 = array3_from_file!(24, 24, 5, "testdata/advance/24_4/gs2.bin");
            assert_abs_diff_eq!(&gs2, &STATE.gs, epsilon = 1.0E-10);
        }
    }
}
