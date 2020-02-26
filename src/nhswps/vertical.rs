use crate::{constants::*, nhswps::State, utils::*};

/// Calculates layer heights (z), as well as dz/dx & dz/dy (zx & zy),
/// the vertical velocity (w), and the A = grad{u*rho'_theta} (aa).
pub fn vertical(state: &mut State) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let dz2 = (HBAR / nz as f64) / 2.0;

    let mut rsrc = vec![0.0; ng * ng * (nz + 1)];
    let mut wkq = vec![0.0; ng * ng];
    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];
    let mut wkc = vec![0.0; ng * ng];

    // Only need to consider iz > 0 as z = w = 0 for iz = 0:

    // Find z by trapezoidal integration of rho_theta (integrate over
    // rho'_theta then add theta to the result):
    {
        for i in 0..ng {
            for j in 0..ng {
                state.z[[i, j, 1]] = dz2 * (state.r[[i, j, 0]] + state.r[[i, j, 1]]);
            }
        }

        for iz in 1..nz {
            for i in 0..ng {
                for j in 0..ng {
                    state.z[[i, j, iz + 1]] =
                        state.z[[i, j, iz]] + dz2 * (state.r[[i, j, iz]] + state.r[[i, j, iz + 1]]);
                }
            }
        }
    }

    for iz in 1..=nz {
        // Add on theta (a linear function) to complete definition of z:
        for i in 0..ng {
            for j in 0..ng {
                state.z[[i, j, iz]] += state.spectral.theta[iz];
            }
        }

        // Calculate z_x & z_y:
        let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[[i, j]] = state.z[[i, j, iz]];
            }
        }

        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut wkq);
        {
            let wkq_matrix = view2d(&wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    state.zx[[i, j, iz]] = wkq_matrix[[i, j]];
                }
            }
        }
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut wkq);
        {
            let wkq_matrix = view2d(&wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    state.zy[[i, j, iz]] = wkq_matrix[[i, j]];
                }
            }
        }
    }

    // Calculate A = grad{u*rho'_theta} (spectral):
    for iz in 0..=nz {
        // Calculate (u*rho'_theta)_x:
        {
            let mut wkq = viewmut2d(&mut wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkq[[i, j]] = state.u[[i, j, iz]] * state.r[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);

        // Calculate (v*rho'_theta)_y:
        {
            let mut wkq = viewmut2d(&mut wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkq[[i, j]] = state.v[[i, j, iz]] * state.r[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkc);

        // Apply de-aliasing filter and complete definition of A:
        {
            let wkb = view2d(&wkb, ng, ng);
            let wkc = view2d(&wkc, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    state.aa[[i, j, iz]] =
                        state.spectral.filt[[i, j]] * (wkb[[i, j]] + wkc[[i, j]]);
                }
            }
        }
        // Need -(A + delta) in physical space for computing w just below:
        {
            let mut wka = viewmut2d(&mut wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wka[[i, j]] = state.aa[[i, j, iz]] + state.ds[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.spctop(&mut wka, &mut wkq);
        {
            let mut rsrc = viewmut3d(&mut rsrc, ng, ng, nz + 1);
            let wkq = view2d(&wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    rsrc[[i, j, iz]] = -wkq[[i, j]];
                }
            }
        }
    }

    // Calculate vertical velocity (0 at iz = 0):
    {
        let rsrc = view3d(&rsrc, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                state.w[[i, j, 1]] = dz2 * (rsrc[[i, j, 0]] + rsrc[[i, j, 1]]);
            }
        }
    }
    for iz in 1..nz {
        {
            let rsrc = view3d(&rsrc, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    state.w[[i, j, iz + 1]] =
                        state.w[[i, j, iz]] + dz2 * (rsrc[[i, j, iz]] + rsrc[[i, j, iz + 1]]);
                }
            }
        };
    }

    // Complete definition of w by adding u*z_x + v*z_y after de-aliasing:
    for iz in 1..=nz {
        {
            let mut wkq = viewmut2d(&mut wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    wkq[[i, j]] = state.u[[i, j, iz]] * state.zx[[i, j, iz]]
                        + state.v[[i, j, iz]] * state.zy[[i, j, iz]];
                }
            }
        }
        state.spectral.deal2d(&mut wkq);
        {
            let wkq = view2d(&wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    state.w[[i, j, iz]] += wkq[[i, j]];
                }
            }
        }
    }
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
        byteorder::{ByteOrder, NetworkEndian},
        lazy_static::lazy_static,
        ndarray::{Array3, ShapeBuilder},
    };

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
