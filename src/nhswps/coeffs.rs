use {
    crate::{constants::*, nhswps::State, utils::*},
    ndarray::{ArrayViewMut3, Axis, Zip},
    rayon::prelude::*,
};

/// Calculates the fixed coefficients used in the pressure iteration.
pub fn coeffs(
    state: &State,
    mut sigx: ArrayViewMut3<f64>,
    mut sigy: ArrayViewMut3<f64>,
    mut cpt1: ArrayViewMut3<f64>,
    mut cpt2: ArrayViewMut3<f64>,
) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let qdzi = (1.0 / 4.0) * (1.0 / (HBAR / nz as f64));

    // Compute sigx and sigy and de-alias:
    Zip::from(&mut sigx)
        .and(&state.ri)
        .and(&state.zx)
        .apply(|sigx, ri, zx| *sigx = ri * zx);
    Zip::from(&mut sigy)
        .and(&state.ri)
        .and(&state.zy)
        .apply(|sigy, ri, zy| *sigy = ri * zy);
    state
        .spectral
        .deal3d(sigx.as_slice_memory_order_mut().unwrap());
    state
        .spectral
        .deal3d(sigy.as_slice_memory_order_mut().unwrap());

    // Compute cpt2 and de-alias:
    Zip::from(&mut cpt2)
        .and(&state.ri)
        .and(&sigx)
        .and(&sigy)
        .apply(|cpt2, ri, sigx, sigy| *cpt2 = 1.0 - ri.powf(2.0) - sigx.powf(2.0) - sigy.powf(2.0));

    state
        .spectral
        .deal3d(cpt2.as_slice_memory_order_mut().unwrap());

    // Interior (centred differencing):
    cpt1.axis_iter_mut(Axis(2))
        .into_par_iter()
        .zip(0..(nz as u16))
        .for_each(|(cpt1, iz)| {
            let iz = iz as usize;
            let mut wka = arr2zero(ng);
            let mut wkp = arr2zero(ng);

            if iz == 0 {
                // qdzi=1/(4*dz) is used since 0.5*d/dtheta is being computed.
                // Calculate 0.5*d(cpt2)/dtheta + div(sigx,sigy) and store in cpt1:

                // Lower boundary (use higher order formula):
                Zip::from(cpt1)
                    .and(cpt2.index_axis(Axis(2), 0))
                    .and(cpt2.index_axis(Axis(2), 1))
                    .and(cpt2.index_axis(Axis(2), 2))
                    .apply(|cpt1, cpt2_0, cpt2_1, cpt2_2| {
                        *cpt1 = qdzi * (4.0 * cpt2_1 - 3.0 * cpt2_0 - cpt2_2)
                    });
            } else {
                state.spectral.divs(
                    sigx.index_axis(Axis(2), iz)
                        .as_slice_memory_order()
                        .unwrap(),
                    sigy.index_axis(Axis(2), iz)
                        .as_slice_memory_order()
                        .unwrap(),
                    wka.as_slice_memory_order_mut().unwrap(),
                );
                state.spectral.d2fft.spctop(
                    wka.as_slice_memory_order_mut().unwrap(),
                    wkp.as_slice_memory_order_mut().unwrap(),
                );

                Zip::from(cpt1)
                    .and(cpt2.index_axis(Axis(2), iz + 1))
                    .and(cpt2.index_axis(Axis(2), iz - 1))
                    .and(&wkp)
                    .apply(|cpt1, cpt2_p, cpt2_m, wkp| *cpt1 = qdzi * (cpt2_p - cpt2_m) + wkp);
            }
        });

    // Upper boundary (use higher order formula):
    let mut wkp = arr2zero(ng);
    let mut wka = arr2zero(ng);
    state.spectral.divs(
        sigx.index_axis(Axis(2), nz)
            .as_slice_memory_order()
            .unwrap(),
        sigy.index_axis(Axis(2), nz)
            .as_slice_memory_order()
            .unwrap(),
        wka.as_slice_memory_order_mut().unwrap(),
    );
    state.spectral.d2fft.spctop(
        wka.as_slice_memory_order_mut().unwrap(),
        wkp.as_slice_memory_order_mut().unwrap(),
    );

    Zip::from(cpt1.index_axis_mut(Axis(2), nz))
        .and(cpt2.index_axis(Axis(2), nz))
        .and(cpt2.index_axis(Axis(2), nz - 1))
        .and(cpt2.index_axis(Axis(2), nz - 2))
        .and(&wkp)
        .apply(|cpt1, cpt2_0, cpt2_1, cpt2_2, wkp| {
            *cpt1 = qdzi * (3.0 * cpt2_0 + cpt2_2 - 4.0 * cpt2_1) + wkp;
        });

    // Re-define sigx and sigy to include a factor of 2:
    sigx *= 2.0;
    sigy *= 2.0;
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            array3_from_file,
            nhswps::{Output, Spectral},
        },
        approx::assert_abs_diff_eq,
        byteorder::ByteOrder,
        lazy_static::lazy_static,
        ndarray::{Array3, ShapeBuilder},
    };

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
        let mut sigx = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigx.bin");
        let mut sigy = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigy.bin");
        let mut cpt1 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt1.bin");
        let mut cpt2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt2.bin");
        let sigx2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigx2.bin");

        coeffs(
            &STATE_18_2,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(sigx2, sigx, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_sigy() {
        let mut sigx = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigx.bin");
        let mut sigy = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigy.bin");
        let mut cpt1 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt1.bin");
        let mut cpt2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt2.bin");
        let sigy2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigy2.bin");

        coeffs(
            &STATE_18_2,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(sigy2, sigy, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_cpt1() {
        let mut sigx = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigx.bin");
        let mut sigy = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigy.bin");
        let mut cpt1 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt1.bin");
        let mut cpt2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt2.bin");
        let cpt12 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt12.bin");

        coeffs(
            &STATE_18_2,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&cpt12, &cpt1, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_cpt2() {
        let mut sigx = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigx.bin");
        let mut sigy = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_sigy.bin");
        let mut cpt1 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt1.bin");
        let mut cpt2 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt2.bin");
        let cpt22 = array3_from_file!(18, 18, 3, "testdata/coeffs/18_2_cpt22.bin");

        coeffs(
            &STATE_18_2,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&cpt22, &cpt2, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_sigx() {
        let mut sigx = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigx.bin");
        let mut sigy = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigy.bin");
        let mut cpt1 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt1.bin");
        let mut cpt2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt2.bin");
        let sigx2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigx2.bin");

        coeffs(
            &STATE_32_4,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&sigx2, &sigx, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_sigy() {
        let mut sigx = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigx.bin");
        let mut sigy = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigy.bin");
        let mut cpt1 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt1.bin");
        let mut cpt2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt2.bin");
        let sigy2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigy2.bin");

        coeffs(
            &STATE_32_4,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&sigy2, &sigy, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_cpt1() {
        let mut sigx = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigx.bin");
        let mut sigy = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigy.bin");
        let mut cpt1 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt1.bin");
        let mut cpt2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt2.bin");
        let cpt12 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt12.bin");

        coeffs(
            &STATE_32_4,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&cpt12, &cpt1, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4_cpt2() {
        let mut sigx = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigx.bin");
        let mut sigy = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_sigy.bin");
        let mut cpt1 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt1.bin");
        let mut cpt2 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt2.bin");
        let cpt22 = array3_from_file!(32, 32, 5, "testdata/coeffs/32_4_cpt22.bin");

        coeffs(
            &STATE_32_4,
            sigx.view_mut(),
            sigy.view_mut(),
            cpt1.view_mut(),
            cpt2.view_mut(),
        );

        assert_abs_diff_eq!(&cpt22, &cpt2, epsilon = 1.0E-10);
    }
}
