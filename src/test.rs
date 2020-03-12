use {
    crate::{
        balinit::balinit,
        nhswps::{nhswps, Output},
        parameters::Parameters,
        swto3d::swto3d,
        vstrip::init_pv_strip,
    },
    approx::assert_abs_diff_eq,
    byteorder::{ByteOrder, LittleEndian},
    lazy_static::lazy_static,
};

lazy_static! {
    static ref OUTPUT_18_4: Output = {
        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 4;
        params.numerical.time_step = 1.0 / (18 as f64);

        let qq = init_pv_strip(&params);
        let (qq, dd, gg) = balinit(qq.as_slice_memory_order().unwrap(), &params);
        let (qq, dd, gg) = swto3d(&qq, &dd, &gg, &params);

        nhswps(&qq, &dd, &gg, &params)
    };
    static ref OUTPUT_32_4: Output = {
        let params = Parameters::default();

        let qq = init_pv_strip(&params);
        let (qq, dd, gg) = balinit(qq.as_slice_memory_order().unwrap(), &params);
        let (qq, dd, gg) = swto3d(&qq, &dd, &gg, &params);

        nhswps(&qq, &dd, &gg, &params)
    };
}

mod complete_18_4 {
    use super::*;

    #[test]
    #[ignore]
    fn monitor() {
        assert_eq!(
            include_str!("testdata/complete/18_4/monitor.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_18_4
                .monitor
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[test]
    #[ignore]
    fn ecomp() {
        assert_eq!(
            include_str!("testdata/complete/18_4/ecomp.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_18_4
                .ecomp
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    /*
    // The following tests (and the similar tests for 32_4) have been commented out due to the "spectra.asc" output essentially
    // representing the total accumulated error throughout execution and so cannot be expected to maintain bit-level similarity
    // with the FORTRAN implementation especially when parallelisation is implemented.

    #[cfg(all(target_arch = "x86_64", target_os = "macos",))]
    #[test]
    #[ignore]
    fn spectra() {
        assert_eq!(
            include_str!("testdata/complete/18_4/spectra_x86_64-apple-darwin.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_18_4
                .spectra
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux",))]
    #[test]
    #[ignore]
    fn spectra() {
        assert_eq!(
            include_str!("testdata/complete/18_4/spectra_x86_64-unknown-linux-gnu.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_18_4
                .spectra
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }
    */

    mod _2d {
        use super::*;

        #[test]
        #[ignore]
        fn d() {
            let d2 = include_bytes!("testdata/complete/18_4/2d/d.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let d = OUTPUT_18_4
                .d2d
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in d2.iter().zip(&d) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn g() {
            let g2 = include_bytes!("testdata/complete/18_4/2d/g.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let g = OUTPUT_18_4
                .d2g
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in g2.iter().zip(&g) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn h() {
            let h2 = include_bytes!("testdata/complete/18_4/2d/h.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let h = OUTPUT_18_4
                .d2h
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in h2.iter().zip(&h) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn q() {
            let q2 = include_bytes!("testdata/complete/18_4/2d/q.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let q = OUTPUT_18_4
                .d2q
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in q2.iter().zip(&q) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn zeta() {
            let zeta2 = include_bytes!("testdata/complete/18_4/2d/zeta.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let zeta = OUTPUT_18_4
                .d2zeta
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in zeta2.iter().zip(&zeta) {
                assert_abs_diff_eq!(a, b);
            }
        }
    }

    mod _3d {
        use super::*;

        #[test]
        #[ignore]
        fn d() {
            let d2 = include_bytes!("testdata/complete/18_4/3d/d.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let d = OUTPUT_18_4
                .d3d
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in d2.iter().zip(&d) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn g() {
            let g2 = include_bytes!("testdata/complete/18_4/3d/g.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let g = OUTPUT_18_4
                .d3g
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in g2.iter().zip(&g) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn pn() {
            let pn2 = include_bytes!("testdata/complete/18_4/3d/pn.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let pn = OUTPUT_18_4
                .d3pn
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in pn2.iter().zip(&pn) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn ql() {
            assert_eq!(
                include_bytes!("testdata/complete/18_4/3d/ql.r4")[..],
                OUTPUT_18_4.d3ql[..]
            );
        }

        #[test]
        #[ignore]
        fn r() {
            let r2 = include_bytes!("testdata/complete/18_4/3d/r.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let r = OUTPUT_18_4
                .d3r
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in r2.iter().zip(&r) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn w() {
            let w2 = include_bytes!("testdata/complete/18_4/3d/w.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let w = OUTPUT_18_4
                .d3w
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in w2.iter().zip(&w) {
                assert_abs_diff_eq!(a, b);
            }
        }
    }
}

mod complete_32_4 {
    use super::*;

    #[test]
    #[ignore]
    fn monitor() {
        assert_eq!(
            include_str!("testdata/complete/32_4/monitor.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_32_4
                .monitor
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[test]
    #[ignore]
    fn ecomp() {
        assert_eq!(
            include_str!("testdata/complete/32_4/ecomp.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_32_4
                .ecomp
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    /*
    #[cfg(all(target_arch = "x86_64", target_os = "macos",))]
    #[test]
    #[ignore]
    fn spectra() {
        assert_eq!(
            include_str!("testdata/complete/32_4/spectra_x86_64-apple-darwin.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_32_4
                .spectra
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux",))]
    #[test]
    #[ignore]
    fn spectra() {
        assert_eq!(
            include_str!("testdata/complete/32_4/spectra_x86_64-unknown-linux-gnu.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            OUTPUT_32_4
                .spectra
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }
    */

    mod _2d {
        use super::*;

        #[test]
        #[ignore]
        fn d() {
            let d2 = include_bytes!("testdata/complete/32_4/2d/d.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let d = OUTPUT_32_4
                .d2d
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in d2.iter().zip(&d) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn g() {
            let g2 = include_bytes!("testdata/complete/32_4/2d/g.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let g = OUTPUT_32_4
                .d2g
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in g2.iter().zip(&g) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn h() {
            let h2 = include_bytes!("testdata/complete/32_4/2d/h.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let h = OUTPUT_32_4
                .d2h
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in h2.iter().zip(&h) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn q() {
            let q2 = include_bytes!("testdata/complete/32_4/2d/q.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let q = OUTPUT_32_4
                .d2q
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in q2.iter().zip(&q) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn zeta() {
            let zeta2 = include_bytes!("testdata/complete/32_4/2d/zeta.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let zeta = OUTPUT_32_4
                .d2zeta
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in zeta2.iter().zip(&zeta) {
                assert_abs_diff_eq!(a, b);
            }
        }
    }

    mod _3d {
        use super::*;

        #[test]
        #[ignore]
        fn d() {
            let d2 = include_bytes!("testdata/complete/32_4/3d/d.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let d = OUTPUT_32_4
                .d3d
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in d2.iter().zip(&d) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn g() {
            let g2 = include_bytes!("testdata/complete/32_4/3d/g.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let g = OUTPUT_32_4
                .d3g
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in g2.iter().zip(&g) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn pn() {
            let pn2 = include_bytes!("testdata/complete/32_4/3d/pn.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let pn = OUTPUT_32_4
                .d3pn
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in pn2.iter().zip(&pn) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn ql() {
            assert_eq!(
                include_bytes!("testdata/complete/32_4/3d/ql.r4")[..],
                OUTPUT_32_4.d3ql[..]
            );
        }

        #[test]
        #[ignore]
        fn r() {
            let r2 = include_bytes!("testdata/complete/32_4/3d/r.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let r = OUTPUT_32_4
                .d3r
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in r2.iter().zip(&r) {
                assert_abs_diff_eq!(a, b);
            }
        }

        #[test]
        #[ignore]
        fn w() {
            let w2 = include_bytes!("testdata/complete/32_4/3d/w.r4")
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            let w = OUTPUT_32_4
                .d3w
                .chunks(4)
                .map(LittleEndian::read_f32)
                .collect::<Vec<f32>>();

            for (a, b) in w2.iter().zip(&w) {
                assert_abs_diff_eq!(a, b);
            }
        }
    }
}
