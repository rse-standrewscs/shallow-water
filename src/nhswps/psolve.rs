use {
    crate::{
        constants::*,
        nhswps::{coeffs::coeffs, cpsource::cpsource, vertical::vertical, State},
        utils::*,
    },
    log::error,
    ndarray::{Axis, Zip},
};

/// Solves for the nonhydrostatic part of the pressure (pn) given
/// the velocity field (u,v,w) together with r = rho'_theta and
/// z = theta + int_0^theta{rho'_theta(s)ds}.
pub fn psolve(state: &mut State) {
    let toler = 1.0E-9;
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let dz = HBAR / (nz as f64);
    let dzi = 1.0 / dz;
    let dz2 = dz / 2.0;
    let dz6 = dz / 6.0;
    let dzisq = (1.0 / dz).powf(2.0);
    let hdzi = (1.0 / 2.0) * (1.0 / (HBAR / nz as f64));

    // Local variables:
    let nitmax: usize = 100;
    // nitmax: maximum number of iterations allowed before stopping

    // Constant part of the pressure source:
    let mut sp0 = vec![0.0; ng * ng * (nz + 1)];

    // Arrays used for pressure inversion (these depend on rho'_theta only):
    let mut sigx = vec![0.0; ng * ng * (nz + 1)];
    let mut sigy = vec![0.0; ng * ng * (nz + 1)];
    let mut cpt1 = vec![0.0; ng * ng * (nz + 1)];
    let mut cpt2 = vec![0.0; ng * ng * (nz + 1)];

    // Physical space arrays:
    let mut dpdt = vec![0.0; ng * ng];
    let mut d2pdxt = vec![0.0; ng * ng];
    let mut d2pdyt = vec![0.0; ng * ng];
    let mut d2pdt2 = vec![0.0; ng * ng];
    let mut wkp = vec![0.0; ng * ng];
    let mut wkq = vec![0.0; ng * ng];

    // Spectral space arrays (all work arrays):
    let mut sp = vec![0.0; ng * ng * (nz + 1)];
    let mut gg = vec![0.0; ng * ng * (nz + 1)];
    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];
    let mut wkc = vec![0.0; ng * ng];
    let mut wkd = vec![0.0; ng * ng];

    // Calculate 1/(1+rho'_theta) and de-aliase:
    Zip::from(&mut state.ri)
        .and(&state.r)
        .apply(|ri, r| *ri = 1.0 / (r + 1.0));
    state
        .spectral
        .deal3d(state.ri.as_slice_memory_order_mut().unwrap());

    // Calcuate layer heights z and z_x & z_y, vertical velocity w
    // and A = grad(u*rho'_theta):
    vertical(state);

    // Define constant coefficients in pressure inversion:
    coeffs(state, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

    // Define constant part of the pressure source (sp0):
    cpsource(state, &mut sp0);

    // Solve for the pressure using previous solution as first guess:
    let mut pna = state.pn.clone();

    // Begin iteration to find (non-hydrostatic part of the) pressure
    let mut errp = 1.0;
    let mut iter = 0;
    while errp > toler && iter < nitmax {
        // Get spectral coefficients for pressure:
        state.spectral.ptospc3d(
            state.pn.as_slice_memory_order().unwrap(),
            state.ps.as_slice_memory_order_mut().unwrap(),
            0,
            nz - 1,
        );
        state.ps.index_axis_mut(Axis(2), nz).fill(0.0);

        // Compute pressure derivatives needed in the non-constant part of the
        // source S_1 and add to S_0 (in sp0) to form total source S (sp):

        // Lower boundary at iz = 0 (use dp/dtheta = 0):
        // d^2p/dtheta^2:
        let mut wkd_matrix = viewmut2d(&mut wkd, ng, ng);
        Zip::from(&mut wkd_matrix)
            .and(&state.ps.index_axis(Axis(2), 0))
            .and(&state.ps.index_axis(Axis(2), 1))
            .and(&state.ps.index_axis(Axis(2), 2))
            .and(&state.ps.index_axis(Axis(2), 3))
            .apply(|wkd, ps0, ps1, ps2, ps3| {
                *wkd = (2.0 * ps0 - 5.0 * ps1 + 4.0 * ps2 - ps3) * dzisq
            });

        // Return to physical space:
        state.spectral.d2fft.spctop(&mut wkd, &mut d2pdt2);
        // Total source:
        {
            let mut wkp_matrix = viewmut2d(&mut wkp, ng, ng);
            let sp0_matrix = view3d(&sp0, ng, ng, nz + 1);
            let cpt2_matrix = view3d(&cpt2, ng, ng, nz + 1);
            let d2pdt2_matrix = view2d(&d2pdt2, ng, ng);

            Zip::from(&mut wkp_matrix)
                .and(sp0_matrix.index_axis(Axis(2), 0))
                .and(cpt2_matrix.index_axis(Axis(2), 0))
                .and(d2pdt2_matrix)
                .apply(|wkp, sp0, cpt2, d2pdt2| *wkp = sp0 + cpt2 * d2pdt2);
        }
        // Transform to spectral space for inversion below:
        state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
        {
            let mut sp_matrix = viewmut3d(&mut sp, ng, ng, nz + 1);
            let wka_matrix = view2d(&wka, ng, ng);
            sp_matrix.index_axis_mut(Axis(2), 0).assign(&wka_matrix);
        }

        // Interior grid points:
        for iz in 1..nz {
            wkq = d2pdt2.clone();
            {
                let mut wka = viewmut2d(&mut wka, ng, ng);

                Zip::from(&mut wka)
                    .and(&state.ps.index_axis(Axis(2), iz + 1))
                    .and(&state.ps.index_axis(Axis(2), iz - 1))
                    .apply(|wka, psp, psm| *wka = (psp - psm) * hdzi);
            }
            {
                let mut wkd = viewmut2d(&mut wkd, ng, ng);

                Zip::from(&mut wkd)
                    .and(&state.ps.index_axis(Axis(2), iz + 1))
                    .and(&state.ps.index_axis(Axis(2), iz))
                    .and(&state.ps.index_axis(Axis(2), iz - 1))
                    .apply(|wkd, psp, ps, psm| *wkd = (psp - 2.0 * ps + psm) * dzisq)
            }

            // Calculate x & y derivatives of dp/dtheta:
            state
                .spectral
                .d2fft
                .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
            state
                .spectral
                .d2fft
                .yderiv(&state.spectral.hrky, &wka, &mut wkc);
            // Return to physical space:
            state.spectral.d2fft.spctop(&mut wka, &mut dpdt);
            state.spectral.d2fft.spctop(&mut wkb, &mut d2pdxt);
            state.spectral.d2fft.spctop(&mut wkc, &mut d2pdyt);
            state.spectral.d2fft.spctop(&mut wkd, &mut d2pdt2);

            // Total source:
            {
                let mut wkp = viewmut2d(&mut wkp, ng, ng);
                let sp0 = view3d(&sp0, ng, ng, nz + 1);
                let sigx = view3d(&sigx, ng, ng, nz + 1);
                let sigy = view3d(&sigy, ng, ng, nz + 1);
                let cpt1 = view3d(&cpt1, ng, ng, nz + 1);
                let cpt2 = view3d(&cpt2, ng, ng, nz + 1);
                let d2pdxt = view2d(&d2pdxt, ng, ng);
                let d2pdyt = view2d(&d2pdyt, ng, ng);
                let d2pdt2 = view2d(&d2pdt2, ng, ng);
                let dpdt = view2d(&dpdt, ng, ng);

                for j in 0..ng {
                    for i in 0..ng {
                        wkp[[i, j]] = sp0[[i, j, iz]]
                            + sigx[[i, j, iz]] * d2pdxt[[i, j]]
                            + sigy[[i, j, iz]] * d2pdyt[[i, j]]
                            + cpt2[[i, j, iz]] * d2pdt2[[i, j]]
                            + cpt1[[i, j, iz]] * dpdt[[i, j]];
                    }
                }
            }

            // Transform to spectral space for inversion below:
            state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
            {
                let mut sp_matrix = viewmut3d(&mut sp, ng, ng, nz + 1);
                let wka_matrix = view2d(&wka, ng, ng);

                sp_matrix.index_axis_mut(Axis(2), iz).assign(&wka_matrix);
            };
        }

        // Upper boundary at iz = nz (use p = 0):
        // Extrapolate to find first and second derivatives there:
        for (i, e) in dpdt.iter_mut().enumerate() {
            *e += dz2 * (3.0 * d2pdt2[i] - wkq[i]);
        }
        for (i, e) in d2pdt2.iter_mut().enumerate() {
            *e = 2.0 * *e - wkq[i];
        }
        wkp = dpdt.clone();
        state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
        // Calculate x & y derivatives of dp/dtheta:
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkc);
        // Return to physical space:
        state.spectral.d2fft.spctop(&mut wkb, &mut d2pdxt);
        state.spectral.d2fft.spctop(&mut wkc, &mut d2pdyt);
        // Total source:
        {
            let mut wkp = viewmut2d(&mut wkp, ng, ng);
            let sp0 = view3d(&sp0, ng, ng, nz + 1);
            let sigx = view3d(&sigx, ng, ng, nz + 1);
            let sigy = view3d(&sigy, ng, ng, nz + 1);
            let cpt1 = view3d(&cpt1, ng, ng, nz + 1);
            let cpt2 = view3d(&cpt2, ng, ng, nz + 1);
            let d2pdxt = view2d(&d2pdxt, ng, ng);
            let d2pdyt = view2d(&d2pdyt, ng, ng);
            let d2pdt2 = view2d(&d2pdt2, ng, ng);
            let dpdt = view2d(&dpdt, ng, ng);

            for j in 0..ng {
                for i in 0..ng {
                    wkp[[i, j]] = sp0[[i, j, nz]]
                        + sigx[[i, j, nz]] * d2pdxt[[i, j]]
                        + sigy[[i, j, nz]] * d2pdyt[[i, j]]
                        + cpt2[[i, j, nz]] * d2pdt2[[i, j]]
                        + cpt1[[i, j, nz]] * dpdt[[i, j]];
                }
            }
        };

        // Transform to spectral space for inversion below:
        state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
        {
            let mut sp_matrix = viewmut3d(&mut sp, ng, ng, nz + 1);
            let wka_matrix = view2d(&wka, ng, ng);

            sp_matrix.index_axis_mut(Axis(2), nz).assign(&wka_matrix);
        };

        // Solve tridiagonal problem for pressure in spectral space:
        let mut gg_matrix = viewmut3d(&mut gg, ng, ng, nz + 1);
        let sp_matrix = view3d(&sp, ng, ng, nz + 1);
        Zip::from(&mut gg_matrix.index_axis_mut(Axis(2), 0))
            .and(sp_matrix.index_axis(Axis(2), 0))
            .and(sp_matrix.index_axis(Axis(2), 1))
            .apply(|gg, sp0, sp1| *gg = (1.0 / 3.0) * sp0 + (1.0 / 6.0) * sp1);

        for iz in 1..nz {
            for i in 0..ng {
                for j in 0..ng {
                    gg_matrix[[i, j, iz]] = (1.0 / 12.0)
                        * (sp_matrix[[i, j, iz - 1]] + sp_matrix[[i, j, iz + 1]])
                        + (5.0 / 6.0) * sp_matrix[[i, j, iz]];
                }
            }
        }

        Zip::from(&mut state.ps.index_axis_mut(Axis(2), 0))
            .and(gg_matrix.index_axis(Axis(2), 0))
            .and(state.spectral.htdv.index_axis(Axis(2), 0))
            .apply(|ps, gg, htdv| *ps = gg * htdv);

        for iz in 1..nz {
            for i in 0..ng {
                for j in 0..ng {
                    state.ps[[i, j, iz]] = (gg_matrix[[i, j, iz]]
                        - state.spectral.ap[[i, j]] * state.ps[[i, j, iz - 1]])
                        * state.spectral.htdv[[i, j, iz]];
                }
            }
        }
        for iz in (0..=nz - 2).rev() {
            for i in 0..ng {
                for j in 0..ng {
                    state.ps[[i, j, iz]] +=
                        state.spectral.etdv[[i, j, iz]] * state.ps[[i, j, iz + 1]];
                }
            }
        }

        state.ps.index_axis_mut(Axis(2), nz).fill(0.0);

        // Transform to physical space:
        state.spectral.spctop3d(
            state.ps.as_slice_memory_order().unwrap(),
            state.pn.as_slice_memory_order_mut().unwrap(),
            0,
            nz - 1,
        );

        state.pn.index_axis_mut(Axis(2), nz).fill(0.0);

        // Monitor convergence
        errp = (state
            .pn
            .iter()
            .zip(&pna)
            .map(|(a, b)| (a - b).powf(2.0))
            .sum::<f64>()
            / (pna.iter().map(|x| x.powf(2.0)).sum::<f64>() + 1.0E-20))
            .sqrt();

        // Stop if not converging:
        if iter > 0 && errp > 1.0 {
            error!("Pressure error too large! Final pressure error = {}", errp);
            quit::with_code(1);
        }

        iter += 1;

        // Reset pna:
        pna = state.pn.clone();
    }

    if iter >= nitmax {
        error!(
            "Exceeded maximum number of iterations to find pressure! Final pressure error = {}",
            errp
        );
        quit::with_code(1);
    }

    // Past this point, we have converged!

    // Calculate 1st derivative of pressure using 4th-order compact differences:
    {
        let mut gg = viewmut3d(&mut gg, ng, ng, nz + 1);
        let sp = view3d(&sp, ng, ng, nz + 1);

        for iz in 1..nz {
            for i in 0..ng {
                for j in 0..ng {
                    gg[[i, j, iz]] = (state.ps[[i, j, iz + 1]] - state.ps[[i, j, iz - 1]]) * hdzi;
                }
            }
        }
        for i in 0..ng {
            for j in 0..ng {
                gg[[i, j, nz]] = dz6 * sp[[i, j, nz]] - state.ps[[i, j, nz - 1]] * dzi;
            }
        }

        for i in 0..ng {
            for j in 0..ng {
                gg[[i, j, 1]] *= state.spectral.htd1[0];
            }
        }
        for iz in 2..nz {
            for i in 0..ng {
                for j in 0..ng {
                    gg[[i, j, iz]] = (gg[[i, j, iz]] - (1.0 / 6.0) * gg[[i, j, iz - 1]])
                        * state.spectral.htd1[iz - 1];
                }
            }
        }

        for i in 0..ng {
            for j in 0..ng {
                gg[[i, j, nz]] = (gg[[i, j, nz]] - (1.0 / 3.0) * gg[[i, j, nz - 1]])
                    * state.spectral.htd1[nz - 1];
            }
        }
        for iz in (1..nz).rev() {
            for i in 0..ng {
                for j in 0..ng {
                    gg[[i, j, iz]] += state.spectral.etd1[iz - 1] * gg[[i, j, iz + 1]];
                }
            }
        }
    }

    // Transform to physical space:
    state
        .spectral
        .spctop3d(&gg, state.dpn.as_slice_memory_order_mut().unwrap(), 1, nz);
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
