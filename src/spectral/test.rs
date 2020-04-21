use {
    super::*,
    crate::{array3_from_file, utils::*},
    approx::assert_abs_diff_eq,
    byteorder::{ByteOrder, NetworkEndian},
    ndarray::ShapeBuilder,
};

macro_rules! _1d_from_file {
    ($name:expr) => {
        include_bytes!($name)
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>()
    };
}

mod main_invert {
    use super::*;

    #[test]
    fn _18_2() {
        let spectral = Spectral::new(18, 2);

        let qs = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_qs.bin");
        let ds = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_ds.bin");
        let gs = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_gs.bin");

        let mut r = arr3zero(18, 2);
        let mut u = r.clone();
        let mut v = r.clone();
        let mut zeta = r.clone();

        let r2 = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_r.bin");
        let u2 = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_u.bin");
        let v2 = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_v.bin");
        let zeta2 = array3_from_file!(18, 18, 3, "testdata/main_invert/18_2_zeta.bin");

        spectral.main_invert(
            qs.view(),
            ds.view(),
            gs.view(),
            r.view_mut(),
            u.view_mut(),
            v.view_mut(),
            zeta.view_mut(),
        );

        assert_abs_diff_eq!(r2, r, epsilon = 1.0E-10);
        assert_abs_diff_eq!(u2, u, epsilon = 1.0E-10);
        assert_abs_diff_eq!(v2, v, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zeta2, zeta, epsilon = 1.0E-10);
    }

    #[test]
    fn _30_4() {
        let spectral = Spectral::new(30, 4);

        let qs = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_qs.bin");
        let ds = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_ds.bin");
        let gs = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_gs.bin");

        let mut r = arr3zero(30, 4);
        let mut u = r.clone();
        let mut v = r.clone();
        let mut zeta = r.clone();

        let r2 = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_r.bin");
        let u2 = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_u.bin");
        let v2 = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_v.bin");
        let zeta2 = array3_from_file!(30, 30, 5, "testdata/main_invert/30_4_zeta.bin");

        spectral.main_invert(
            qs.view(),
            ds.view(),
            gs.view(),
            r.view_mut(),
            u.view_mut(),
            v.view_mut(),
            zeta.view_mut(),
        );
        assert_abs_diff_eq!(r2, r, epsilon = 1.0E-10);
        assert_abs_diff_eq!(u2, u, epsilon = 1.0E-10);
        assert_abs_diff_eq!(v2, v, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zeta2, zeta, epsilon = 1.0E-10);
    }

    #[test]
    fn _48_6() {
        let spectral = Spectral::new(48, 6);

        let qs = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_qs.bin");
        let ds = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_ds.bin");
        let gs = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_gs.bin");

        let mut r = arr3zero(48, 6);
        let mut u = r.clone();
        let mut v = r.clone();
        let mut zeta = r.clone();

        let r2 = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_r.bin");
        let u2 = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_u.bin");
        let v2 = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_v.bin");
        let zeta2 = array3_from_file!(48, 48, 7, "testdata/main_invert/48_6_zeta.bin");

        spectral.main_invert(
            qs.view(),
            ds.view(),
            gs.view(),
            r.view_mut(),
            u.view_mut(),
            v.view_mut(),
            zeta.view_mut(),
        );
        assert_abs_diff_eq!(r2, r, epsilon = 1.0E-10);
        assert_abs_diff_eq!(u2, u, epsilon = 1.0E-10);
        assert_abs_diff_eq!(v2, v, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zeta2, zeta, epsilon = 1.0E-10);
    }
}

#[test]
fn jacob_30_4() {
    let spectral = Spectral::new(30, 4);

    let aa = _1d_from_file!("testdata/jacob/30_4_aa.bin");
    let bb = _1d_from_file!("testdata/jacob/30_4_bb.bin");
    let cs2 = _1d_from_file!("testdata/jacob/30_4_cs.bin");

    let mut cs = vec![0.0; 30 * 30];

    spectral.jacob(&aa, &bb, &mut cs);

    assert_approx_eq_slice(&cs2, &cs);
}

#[test]
fn jacob_48_6() {
    let spectral = Spectral::new(48, 6);

    let aa = _1d_from_file!("testdata/jacob/48_6_aa.bin");
    let bb = _1d_from_file!("testdata/jacob/48_6_bb.bin");
    let cs2 = _1d_from_file!("testdata/jacob/48_6_cs.bin");

    let mut cs = vec![0.0; 48 * 48];

    spectral.jacob(&aa, &bb, &mut cs);

    assert_approx_eq_slice(&cs2, &cs);
}

#[test]
fn divs_30_4() {
    let spectral = Spectral::new(30, 4);

    let aa = _1d_from_file!("testdata/divs/30_4_aa.bin");
    let bb = _1d_from_file!("testdata/divs/30_4_bb.bin");
    let cs2 = _1d_from_file!("testdata/divs/30_4_cs.bin");

    let mut cs = vec![0.0; 30 * 30];

    spectral.divs(&aa, &bb, &mut cs);

    assert_approx_eq_slice(&cs2, &cs);
}

#[test]
fn divs_48_6() {
    let spectral = Spectral::new(48, 6);

    let aa = _1d_from_file!("testdata/divs/48_6_aa.bin");
    let bb = _1d_from_file!("testdata/divs/48_6_bb.bin");
    let cs2 = _1d_from_file!("testdata/divs/48_6_cs.bin");

    let mut cs = vec![0.0; 48 * 48];

    spectral.divs(&aa, &bb, &mut cs);

    assert_approx_eq_slice(&cs2, &cs);
}

#[test]
fn ptospc3d_18_2() {
    let spectral = Spectral::new(18, 2);

    let fp = _1d_from_file!("testdata/ptospc3d/18_2_fp.bin");
    let mut fs = _1d_from_file!("testdata/ptospc3d/18_2_fs.bin");
    let fp2 = _1d_from_file!("testdata/ptospc3d/18_2_fp2.bin");
    let fs2 = _1d_from_file!("testdata/ptospc3d/18_2_fs2.bin");

    spectral.ptospc3d(&fp, &mut fs, 0, 1);

    assert_approx_eq_slice(&fs2, &fs);
    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn ptospc3d_30_4() {
    let spectral = Spectral::new(30, 4);

    let fp = _1d_from_file!("testdata/ptospc3d/30_4_fp.bin");
    let mut fs = _1d_from_file!("testdata/ptospc3d/30_4_fs.bin");
    let fp2 = _1d_from_file!("testdata/ptospc3d/30_4_fp2.bin");
    let fs2 = _1d_from_file!("testdata/ptospc3d/30_4_fs2.bin");

    spectral.ptospc3d(&fp, &mut fs, 0, 3);

    assert_approx_eq_slice(&fs2, &fs);
    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn spctop3d_18_2() {
    let spectral = Spectral::new(18, 2);

    let fs = _1d_from_file!("testdata/spctop3d/18_2_fs.bin");
    let mut fp = _1d_from_file!("testdata/spctop3d/18_2_fp.bin");
    let fs2 = _1d_from_file!("testdata/spctop3d/18_2_fs2.bin");
    let fp2 = _1d_from_file!("testdata/spctop3d/18_2_fp2.bin");

    spectral.spctop3d(&fs, &mut fp, 0, 1);

    assert_approx_eq_slice(&fs2, &fs);
    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn spctop3d_30_4() {
    let spectral = Spectral::new(30, 4);

    let fs = _1d_from_file!("testdata/spctop3d/30_4_fs.bin");
    let mut fp = _1d_from_file!("testdata/spctop3d/30_4_fp.bin");
    let fs2 = _1d_from_file!("testdata/spctop3d/30_4_fs2.bin");
    let fp2 = _1d_from_file!("testdata/spctop3d/30_4_fp2.bin");

    spectral.spctop3d(&fs, &mut fp, 0, 3);

    assert_approx_eq_slice(&fs2, &fs);
    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn deal3d_18_2() {
    let spectral = Spectral::new(18, 2);

    let mut fp = _1d_from_file!("testdata/deal3d/18_2_fp.bin");
    let fp2 = _1d_from_file!("testdata/deal3d/18_2_fp2.bin");

    spectral.deal3d(&mut fp);

    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn deal3d_30_4() {
    let spectral = Spectral::new(30, 4);

    let mut fp = _1d_from_file!("testdata/deal3d/30_4_fp.bin");
    let fp2 = _1d_from_file!("testdata/deal3d/30_4_fp2.bin");

    spectral.deal3d(&mut fp);

    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn deal2d_18_2() {
    let spectral = Spectral::new(18, 2);

    let mut fp = _1d_from_file!("testdata/deal2d/18_2_fp.bin");
    let fp2 = _1d_from_file!("testdata/deal2d/18_2_fp2.bin");

    spectral.deal2d(&mut fp);

    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn deal2d_32_4() {
    let spectral = Spectral::new(32, 4);

    let mut fp = _1d_from_file!("testdata/deal2d/32_4_fp.bin");
    let fp2 = _1d_from_file!("testdata/deal2d/32_4_fp2.bin");

    spectral.deal2d(&mut fp);

    assert_approx_eq_slice(&fp2, &fp);
}

#[test]
fn spec1d_48_4() {
    let spectral = Spectral::new(48, 4);

    let ss = _1d_from_file!("testdata/spec1d/48_4_ss.bin");
    let mut spec = _1d_from_file!("testdata/spec1d/48_4_spec.bin");
    let spec2 = _1d_from_file!("testdata/spec1d/48_4_spec2.bin");

    spectral.spec1d(&ss, &mut spec);

    assert_approx_eq_slice(&spec2, &spec);
}

#[test]
fn spec1d_18_2() {
    let spectral = Spectral::new(18, 2);

    let ss = _1d_from_file!("testdata/spec1d/18_2_ss.bin");
    let mut spec = _1d_from_file!("testdata/spec1d/18_2_spec.bin");
    let spec2 = _1d_from_file!("testdata/spec1d/18_2_spec2.bin");

    spectral.spec1d(ss.as_slice(), &mut spec);

    assert_approx_eq_slice(&spec2, &spec);
}

mod init_spectral {
    use {super::*, crate::utils::assert_approx_eq_slice, lazy_static::lazy_static};

    lazy_static! {
        static ref SPECTRAL_32_4: Spectral = Spectral::new(32, 4);
        static ref SPECTRAL_120_16: Spectral = Spectral::new(120, 16);
    }

    #[test]
    fn spmf() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_spmf.bin"),
            SPECTRAL_32_4.spmf
        );
    }

    #[test]
    fn hlap() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_hlap.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.hlap
        );
    }

    #[test]
    fn glap() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_glap.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.glap
        );
    }

    #[test]
    fn rlap() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_rlap.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.rlap
        );
    }

    #[test]
    fn helm() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_helm.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.helm
        );
    }

    #[test]
    fn c2g2() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_c2g2.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.c2g2
        );
    }

    #[test]
    fn simp() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_simp.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.simp
        );
    }

    #[test]
    fn rope() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_rope.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.rope
        );
    }

    #[test]
    fn fope() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_fope.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.fope
        );
    }

    #[test]
    fn filt() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_filt.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.filt
        );
    }

    #[test]
    fn diss() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_diss.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.diss
        );
    }

    #[test]
    fn opak() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_opak.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.opak
        );
    }

    #[test]
    fn rdis() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_rdis.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.rdis
        );
    }

    #[test]
    fn etdv() {
        assert_eq!(
            view3d(
                &_1d_from_file!("testdata/init_spectral/32_4_etdv.bin"),
                32,
                32,
                4
            ),
            SPECTRAL_32_4.etdv
        );
    }

    #[test]
    fn htdv() {
        assert_eq!(
            view3d(
                &_1d_from_file!("testdata/init_spectral/32_4_htdv.bin"),
                32,
                32,
                4
            ),
            SPECTRAL_32_4.htdv
        );
    }

    #[test]
    fn ap() {
        assert_eq!(
            view2d(
                &_1d_from_file!("testdata/init_spectral/32_4_ap.bin"),
                32,
                32
            ),
            SPECTRAL_32_4.ap
        );
    }

    #[test]
    fn etd1() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_etd1.bin"),
            SPECTRAL_32_4.etd1
        );
    }

    #[test]
    fn htd1() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_htd1.bin"),
            SPECTRAL_32_4.htd1
        );
    }

    #[test]
    fn theta() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_theta.bin"),
            SPECTRAL_32_4.theta
        );
    }

    #[test]
    fn weight() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_weight.bin"),
            SPECTRAL_32_4.weight
        );
    }

    #[test]
    fn hrkx() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_hrkx.bin"),
            SPECTRAL_32_4.hrkx
        );
    }

    #[test]
    fn hrky() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_hrky.bin"),
            SPECTRAL_32_4.hrky
        );
    }

    #[test]
    fn rk() {
        assert_eq!(
            _1d_from_file!("testdata/init_spectral/32_4_rk.bin"),
            SPECTRAL_32_4.rk
        );
    }

    #[test]
    fn xtrig() {
        assert_approx_eq_slice(
            &_1d_from_file!("testdata/init_spectral/32_4_xtrig.bin"),
            &SPECTRAL_32_4.d2fft.xtrig,
        );
    }

    #[test]
    fn ytrig() {
        assert_approx_eq_slice(
            &_1d_from_file!("testdata/init_spectral/32_4_ytrig.bin"),
            &SPECTRAL_32_4.d2fft.ytrig,
        );
    }

    #[test]
    fn alk() {
        assert_approx_eq_slice(
            &_1d_from_file!("testdata/init_spectral/32_4_alk.bin"),
            &SPECTRAL_32_4.alk,
        );
    }

    #[test]
    fn factors() {
        assert_eq!([0, 2, 1, 0, 0], SPECTRAL_32_4.d2fft.xfactors);
        assert_eq!([0, 2, 1, 0, 0], SPECTRAL_32_4.d2fft.yfactors);
    }

    #[test]
    fn kmag() {
        assert_eq!(
            view2d(
                &include_bytes!("testdata/init_spectral/32_4_kmag.bin")
                    .chunks(8)
                    .map(NetworkEndian::read_u64)
                    .map(|x| x as usize)
                    .collect::<Vec<usize>>(),
                32,
                32,
            ),
            SPECTRAL_32_4.kmag
        );
    }

    #[test]
    fn kmax() {
        assert_eq!(23, SPECTRAL_32_4.kmax);
        assert_eq!(16, SPECTRAL_32_4.kmaxred);
    }
}
