use {
    crate::{
        balinit::balinit, nhswps::nhswps, parameters::Parameters, swto3d::swto3d,
        utils::assert_approx_eq_files_f32, vstrip::init_pv_strip,
    },
    lazy_static::lazy_static,
    std::fs::read_to_string,
    tempdir::TempDir,
};

lazy_static! {
    static ref OUTPUT_18_4: TempDir = {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 4;
        params.numerical.time_step = 1.0 / (18 as f64);
        params.numerical.duration = 2.0;
        params.environment.output_directory = tempdir.path().to_owned();

        init_pv_strip(&params).unwrap();
        balinit(&params).unwrap();
        swto3d(&params).unwrap();
        nhswps(&params).unwrap();

        tempdir
    };
    static ref OUTPUT_32_4: TempDir = {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut params = Parameters::default();
        params.numerical.duration = 1.0;
        params.environment.output_directory = tempdir.path().to_owned();

        init_pv_strip(&params).unwrap();
        balinit(&params).unwrap();
        swto3d(&params).unwrap();
        nhswps(&params).unwrap();

        tempdir
    };
}

mod complete_18_4 {
    use super::*;

    #[test]
    fn monitor() {
        assert_eq!(
            include_str!("testdata/complete/18_4/monitor.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            read_to_string(OUTPUT_18_4.path().join("monitor.asc"))
                .unwrap()
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[test]
    fn ecomp() {
        assert_eq!(
            include_str!("testdata/complete/18_4/ecomp.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            read_to_string(OUTPUT_18_4.path().join("ecomp.asc"))
                .unwrap()
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
        fn d() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/2d/d.r4",
                OUTPUT_18_4.path().join("2d/d.r4"),
            );
        }

        #[test]
        fn g() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/2d/g.r4",
                OUTPUT_18_4.path().join("2d/g.r4"),
            );
        }

        #[test]
        fn h() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/2d/h.r4",
                OUTPUT_18_4.path().join("2d/h.r4"),
            );
        }

        #[test]
        fn q() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/2d/q.r4",
                OUTPUT_18_4.path().join("2d/q.r4"),
            );
        }

        #[test]
        fn zeta() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/2d/zeta.r4",
                OUTPUT_18_4.path().join("2d/zeta.r4"),
            );
        }
    }

    mod _3d {
        use super::*;

        #[test]
        fn d() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/d.r4",
                OUTPUT_18_4.path().join("3d/d.r4"),
            );
        }

        #[test]
        fn g() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/g.r4",
                OUTPUT_18_4.path().join("3d/g.r4"),
            );
        }

        #[test]
        fn pn() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/pn.r4",
                OUTPUT_18_4.path().join("3d/pn.r4"),
            );
        }

        #[test]
        fn ql() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/ql.r4",
                OUTPUT_18_4.path().join("3d/ql.r4"),
            );
        }

        #[test]
        fn r() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/r.r4",
                OUTPUT_18_4.path().join("3d/r.r4"),
            );
        }

        #[test]
        fn w() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/18_4/3d/w.r4",
                OUTPUT_18_4.path().join("3d/w.r4"),
            );
        }
    }
}

mod complete_32_4 {
    use super::*;

    #[test]
    fn monitor() {
        assert_eq!(
            include_str!("testdata/complete/32_4/monitor.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            read_to_string(OUTPUT_32_4.path().join("monitor.asc"))
                .unwrap()
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    #[test]
    fn ecomp() {
        assert_eq!(
            include_str!("testdata/complete/32_4/ecomp.asc")
                .to_string()
                .split_whitespace()
                .collect::<Vec<&str>>(),
            read_to_string(OUTPUT_32_4.path().join("ecomp.asc"))
                .unwrap()
                .replace("\n", " ")
                .split(' ')
                .filter(|&s| !s.is_empty())
                .collect::<Vec<&str>>()
        );
    }

    /*
    // NOTE: Spectra snapshot testing disabled due to the very high variation in output

    #[cfg(all(target_arch = "x86_64", target_os = "macos",))]
    #[test]
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
        fn d() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/2d/d.r4",
                OUTPUT_32_4.path().join("2d/d.r4"),
            );
        }

        #[test]
        fn g() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/2d/g.r4",
                OUTPUT_32_4.path().join("2d/g.r4"),
            );
        }

        #[test]
        fn h() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/2d/h.r4",
                OUTPUT_32_4.path().join("2d/h.r4"),
            );
        }

        #[test]
        fn q() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/2d/q.r4",
                OUTPUT_32_4.path().join("2d/q.r4"),
            );
        }

        #[test]
        fn zeta() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/2d/zeta.r4",
                OUTPUT_32_4.path().join("2d/zeta.r4"),
            );
        }
    }

    mod _3d {
        use super::*;

        #[test]
        fn d() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/d.r4",
                OUTPUT_32_4.path().join("3d/d.r4"),
            );
        }

        #[test]
        fn g() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/g.r4",
                OUTPUT_32_4.path().join("3d/g.r4"),
            );
        }

        #[test]
        fn pn() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/pn.r4",
                OUTPUT_32_4.path().join("3d/pn.r4"),
            );
        }

        #[test]
        fn ql() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/ql.r4",
                OUTPUT_32_4.path().join("3d/ql.r4"),
            );
        }

        #[test]
        fn r() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/r.r4",
                OUTPUT_32_4.path().join("3d/r.r4"),
            );
        }

        #[test]
        fn w() {
            assert_approx_eq_files_f32(
                "src/testdata/complete/32_4/3d/w.r4",
                OUTPUT_32_4.path().join("3d/w.r4"),
            );
        }
    }
}
