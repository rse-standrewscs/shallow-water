use {
    crate::{
        constants::*,
        nhswps::{coeffs::coeffs, cpsource::cpsource, vertical::vertical, State},
        utils::{arr2zero, arr3zero},
    },
    log::error,
    ndarray::{Axis, Zip},
    parking_lot::Mutex,
    rayon::prelude::*,
    std::sync::Arc,
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

    let zero2 = arr2zero(ng);
    let zero3 = arr3zero(ng, nz);

    // Constant part of the pressure source:
    let mut sp0 = zero3.clone();

    // Arrays used for pressure inversion (these depend on rho'_theta only):
    let mut sigx = zero3.clone();
    let mut sigy = zero3.clone();
    let mut cpt1 = zero3.clone();
    let mut cpt2 = zero3.clone();

    // Physical space arrays:
    let mut dpdt = zero2.clone();
    let mut d2pdxt = zero2.clone();
    let mut d2pdyt = zero2.clone();
    let d2pdt2 = zero2.clone();
    let mut wkp = zero2.clone();
    let mut wkq;

    // Spectral space arrays (all work arrays):
    let sp = zero3.clone();
    let mut gg = zero3.clone();
    let mut wka = zero2.clone();
    let mut wkb = zero2.clone();
    let mut wkc = zero2.clone();
    let mut wkd = zero2.clone();

    // Calculate 1/(1+rho'_theta) and de-aliase:
    Zip::from(&mut state.ri)
        .and(&state.r)
        .apply(|ri, r| *ri = 1.0 / (r + 1.0));
    state.spectral.deal3d(state.ri.view_mut());

    // Calcuate layer heights z and z_x & z_y, vertical velocity w
    // and A = grad(u*rho'_theta):
    vertical(state);

    // Define constant coefficients in pressure inversion:
    coeffs(
        state,
        sigx.view_mut(),
        sigy.view_mut(),
        cpt1.view_mut(),
        cpt2.view_mut(),
    );

    // Define constant part of the pressure source (sp0):
    cpsource(state, sp0.view_mut());

    // Solve for the pressure using previous solution as first guess:
    let mut pna = state.pn.clone();

    // Place `sp` inside a Mutex inside an Arc for sharing across threads
    //let mut sp = Arc::new(Mutex::new(sp));
    let sp = Arc::new(Mutex::new(sp));
    let d2pdt2 = Arc::new(Mutex::new(d2pdt2));

    // Begin iteration to find (non-hydrostatic part of the) pressure
    let mut errp = 1.0;
    let mut iter = 0;
    while errp > toler && iter < nitmax {
        // Get spectral coefficients for pressure:
        state
            .spectral
            .ptospc3d(state.pn.view(), state.ps.view_mut(), 0, nz - 1);
        state.ps.index_axis_mut(Axis(2), nz).fill(0.0);

        // Compute pressure derivatives needed in the non-constant part of the
        // source S_1 and add to S_0 (in sp0) to form total source S (sp):

        // Lower boundary at iz = 0 (use dp/dtheta = 0):
        // d^2p/dtheta^2:
        Zip::from(&mut wkd)
            .and(&state.ps.index_axis(Axis(2), 0))
            .and(&state.ps.index_axis(Axis(2), 1))
            .and(&state.ps.index_axis(Axis(2), 2))
            .and(&state.ps.index_axis(Axis(2), 3))
            .apply(|wkd, ps0, ps1, ps2, ps3| {
                *wkd = (2.0 * ps0 - 5.0 * ps1 + 4.0 * ps2 - ps3) * dzisq;
            });

        // Return to physical space:
        state.spectral.d2fft.spctop(
            wkd.as_slice_memory_order_mut().unwrap(),
            d2pdt2.lock().as_slice_memory_order_mut().unwrap(),
        );
        // Total source:
        Zip::from(&mut wkp)
            .and(sp0.index_axis(Axis(2), 0))
            .and(cpt2.index_axis(Axis(2), 0))
            .and(&(*d2pdt2.lock()))
            .apply(|wkp, sp0, cpt2, d2pdt2| *wkp = sp0 + cpt2 * d2pdt2);

        // Transform to spectral space for inversion below:
        state.spectral.d2fft.ptospc(
            wkp.as_slice_memory_order_mut().unwrap(),
            wka.as_slice_memory_order_mut().unwrap(),
        );
        sp.lock().index_axis_mut(Axis(2), 0).assign(&wka);

        // Interior grid points:
        (1..nz - 1).into_par_iter().for_each(|iz| {
            let mut wka = zero2.clone();
            let mut wkb = zero2.clone();
            let mut wkc = zero2.clone();
            let mut wkd = zero2.clone();
            let mut wkp = zero2.clone();

            let mut dpdt = zero2.clone();
            let mut d2pdxt = zero2.clone();
            let mut d2pdyt = zero2.clone();
            let mut d2pdt2_local = zero2.clone();

            Zip::from(&mut wka)
                .and(&state.ps.index_axis(Axis(2), iz + 1))
                .and(&state.ps.index_axis(Axis(2), iz - 1))
                .apply(|wka, psp, psm| *wka = (psp - psm) * hdzi);

            Zip::from(&mut wkd)
                .and(&state.ps.index_axis(Axis(2), iz + 1))
                .and(&state.ps.index_axis(Axis(2), iz))
                .and(&state.ps.index_axis(Axis(2), iz - 1))
                .apply(|wkd, psp, ps, psm| *wkd = (psp - 2.0 * ps + psm) * dzisq);

            // Calculate x & y derivatives of dp/dtheta:
            state.spectral.d2fft.xderiv(
                &state.spectral.hrkx,
                wka.as_slice_memory_order().unwrap(),
                wkb.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.yderiv(
                &state.spectral.hrky,
                wka.as_slice_memory_order().unwrap(),
                wkc.as_slice_memory_order_mut().unwrap(),
            );

            // Return to physical space:
            state.spectral.d2fft.spctop(
                wka.as_slice_memory_order_mut().unwrap(),
                dpdt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkb.as_slice_memory_order_mut().unwrap(),
                d2pdxt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkc.as_slice_memory_order_mut().unwrap(),
                d2pdyt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkd.as_slice_memory_order_mut().unwrap(),
                d2pdt2_local.as_slice_memory_order_mut().unwrap(),
            );

            // Total source:
            Zip::from(&mut wkp)
                .and(sp0.index_axis(Axis(2), iz))
                .and(sigx.index_axis(Axis(2), iz))
                .and(&d2pdxt)
                .and(sigy.index_axis(Axis(2), iz))
                .and(&d2pdyt)
                .apply(|wkp, sp0, sigx, d2pdxt, sigy, d2pdyt| {
                    *wkp = sp0 + sigx * d2pdxt + sigy * d2pdyt
                });
            Zip::from(&mut wkp)
                .and(cpt2.index_axis(Axis(2), iz))
                .and(&d2pdt2_local)
                .and(cpt1.index_axis(Axis(2), iz))
                .and(&dpdt)
                .apply(|wkp, cpt2, d2pdt2, cpt1, dpdt| *wkp += cpt2 * d2pdt2 + cpt1 * dpdt);

            // Transform to spectral space for inversion below:
            state.spectral.d2fft.ptospc(
                wkp.as_slice_memory_order_mut().unwrap(),
                wka.as_slice_memory_order_mut().unwrap(),
            );

            if iz == nz - 2 {
                d2pdt2.lock().assign(&d2pdt2_local);
            }

            sp.lock().index_axis_mut(Axis(2), iz).assign(&wka);
        });

        {
            let iz = nz - 1;

            let mut wka = zero2.clone();
            let mut wkb = zero2.clone();
            let mut wkc = zero2.clone();
            let mut wkd = zero2.clone();
            let mut wkp = zero2.clone();

            wkq = d2pdt2.lock().clone();

            Zip::from(&mut wka)
                .and(&state.ps.index_axis(Axis(2), iz + 1))
                .and(&state.ps.index_axis(Axis(2), iz - 1))
                .apply(|wka, psp, psm| *wka = (psp - psm) * hdzi);

            Zip::from(&mut wkd)
                .and(&state.ps.index_axis(Axis(2), iz + 1))
                .and(&state.ps.index_axis(Axis(2), iz))
                .and(&state.ps.index_axis(Axis(2), iz - 1))
                .apply(|wkd, psp, ps, psm| *wkd = (psp - 2.0 * ps + psm) * dzisq);

            // Calculate x & y derivatives of dp/dtheta:
            state.spectral.d2fft.xderiv(
                &state.spectral.hrkx,
                wka.as_slice_memory_order().unwrap(),
                wkb.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.yderiv(
                &state.spectral.hrky,
                wka.as_slice_memory_order().unwrap(),
                wkc.as_slice_memory_order_mut().unwrap(),
            );

            // Return to physical space:
            state.spectral.d2fft.spctop(
                wka.as_slice_memory_order_mut().unwrap(),
                dpdt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkb.as_slice_memory_order_mut().unwrap(),
                d2pdxt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkc.as_slice_memory_order_mut().unwrap(),
                d2pdyt.as_slice_memory_order_mut().unwrap(),
            );
            state.spectral.d2fft.spctop(
                wkd.as_slice_memory_order_mut().unwrap(),
                d2pdt2.lock().as_slice_memory_order_mut().unwrap(),
            );

            // Total source:
            Zip::from(&mut wkp)
                .and(sp0.index_axis(Axis(2), iz))
                .and(sigx.index_axis(Axis(2), iz))
                .and(&d2pdxt)
                .and(sigy.index_axis(Axis(2), iz))
                .and(&d2pdyt)
                .apply(|wkp, sp0, sigx, d2pdxt, sigy, d2pdyt| {
                    *wkp = sp0 + sigx * d2pdxt + sigy * d2pdyt
                });
            Zip::from(&mut wkp)
                .and(cpt2.index_axis(Axis(2), iz))
                .and(&(*d2pdt2.lock()))
                .and(cpt1.index_axis(Axis(2), iz))
                .and(&dpdt)
                .apply(|wkp, cpt2, d2pdt2, cpt1, dpdt| *wkp += cpt2 * d2pdt2 + cpt1 * dpdt);

            // Transform to spectral space for inversion below:
            state.spectral.d2fft.ptospc(
                wkp.as_slice_memory_order_mut().unwrap(),
                wka.as_slice_memory_order_mut().unwrap(),
            );

            sp.lock().index_axis_mut(Axis(2), iz).assign(&wka);
        }

        // Upper boundary at iz = nz (use p = 0):
        // Extrapolate to find first and second derivatives there:
        Zip::from(&mut dpdt)
            .and(&(*d2pdt2.lock()))
            .and(&wkq)
            .apply(|dpdt, d2pdt2, wkq| *dpdt += dz2 * (3.0 * d2pdt2 - wkq));
        Zip::from(&mut (*d2pdt2.lock()))
            .and(&wkq)
            .apply(|d2pdt2, wkq| *d2pdt2 = 2.0 * *d2pdt2 - wkq);

        wkp = dpdt.clone();

        state.spectral.d2fft.ptospc(
            wkp.as_slice_memory_order_mut().unwrap(),
            wka.as_slice_memory_order_mut().unwrap(),
        );

        // Calculate x & y derivatives of dp/dtheta:
        state.spectral.d2fft.xderiv(
            &state.spectral.hrkx,
            wka.as_slice_memory_order().unwrap(),
            wkb.as_slice_memory_order_mut().unwrap(),
        );
        state.spectral.d2fft.yderiv(
            &state.spectral.hrky,
            wka.as_slice_memory_order().unwrap(),
            wkc.as_slice_memory_order_mut().unwrap(),
        );

        // Return to physical space:
        state.spectral.d2fft.spctop(
            wkb.as_slice_memory_order_mut().unwrap(),
            d2pdxt.as_slice_memory_order_mut().unwrap(),
        );
        state.spectral.d2fft.spctop(
            wkc.as_slice_memory_order_mut().unwrap(),
            d2pdyt.as_slice_memory_order_mut().unwrap(),
        );

        // Total source:
        Zip::from(&mut wkp)
            .and(sp0.index_axis(Axis(2), nz))
            .and(sigx.index_axis(Axis(2), nz))
            .and(&d2pdxt)
            .and(sigy.index_axis(Axis(2), nz))
            .and(&d2pdyt)
            .apply(|wkp, sp0, sigx, d2pdxt, sigy, d2pdyt| {
                *wkp = sp0 + sigx * d2pdxt + sigy * d2pdyt
            });
        Zip::from(&mut wkp)
            .and(cpt2.index_axis(Axis(2), nz))
            .and(&(*d2pdt2.lock()))
            .and(cpt1.index_axis(Axis(2), nz))
            .and(&dpdt)
            .apply(|wkp, cpt2, d2pdt2, cpt1, dpdt| *wkp += cpt2 * d2pdt2 + cpt1 * dpdt);

        // Transform to spectral space for inversion below:
        state.spectral.d2fft.ptospc(
            wkp.as_slice_memory_order_mut().unwrap(),
            wka.as_slice_memory_order_mut().unwrap(),
        );
        sp.lock().index_axis_mut(Axis(2), nz).assign(&wka);

        // Solve tridiagonal problem for pressure in spectral space:
        {
            let sp = sp.lock();

            Zip::from(gg.index_axis_mut(Axis(2), 0))
                .and(sp.index_axis(Axis(2), 0))
                .and(sp.index_axis(Axis(2), 1))
                .apply(|gg, sp0, sp1| *gg = (1.0 / 3.0) * sp0 + (1.0 / 6.0) * sp1);

            for iz in 1..nz {
                Zip::from(gg.index_axis_mut(Axis(2), iz))
                    .and(sp.index_axis(Axis(2), iz - 1))
                    .and(sp.index_axis(Axis(2), iz + 1))
                    .and(sp.index_axis(Axis(2), iz))
                    .apply(|gg, spm, spp, sp| *gg = (1.0 / 12.0) * (spm + spp) + (5.0 / 6.0) * sp);
            }
        }

        Zip::from(state.ps.index_axis_mut(Axis(2), 0))
            .and(gg.index_axis(Axis(2), 0))
            .and(state.spectral.htdv.index_axis(Axis(2), 0))
            .apply(|ps, gg, htdv| *ps = gg * htdv);

        for iz in 1..nz {
            let ps1 = state.ps.index_axis(Axis(2), iz - 1).into_owned();

            Zip::from(state.ps.index_axis_mut(Axis(2), iz))
                .and(gg.index_axis(Axis(2), iz))
                .and(&state.spectral.ap)
                .and(&ps1)
                .and(state.spectral.htdv.index_axis(Axis(2), iz))
                .apply(|ps, gg, ap, ps1, htdv| *ps = (gg - ap * ps1) * htdv);
        }

        for iz in (0..=nz - 2).rev() {
            let ps1 = state.ps.index_axis(Axis(2), iz + 1).into_owned();

            Zip::from(state.ps.index_axis_mut(Axis(2), iz))
                .and(state.spectral.etdv.index_axis(Axis(2), iz))
                .and(&ps1)
                .apply(|ps, etdv, ps1| *ps += etdv * ps1);
        }

        state.ps.index_axis_mut(Axis(2), nz).fill(0.0);

        // Transform to physical space:
        state
            .spectral
            .spctop3d(state.ps.view(), state.pn.view_mut(), 0, nz - 1);

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
        for iz in 1..nz {
            Zip::from(gg.index_axis_mut(Axis(2), iz))
                .and(state.ps.index_axis(Axis(2), iz + 1))
                .and(state.ps.index_axis(Axis(2), iz - 1))
                .apply(|gg, psp, psm| *gg = (psp - psm) * hdzi);
        }

        Zip::from(gg.index_axis_mut(Axis(2), nz))
            .and(sp.lock().index_axis(Axis(2), nz))
            .and(state.ps.index_axis(Axis(2), nz - 1))
            .apply(|gg, sp, ps| *gg = dz6 * sp - ps * dzi);
        Zip::from(gg.index_axis_mut(Axis(2), 1)).apply(|gg| *gg *= state.spectral.htd1[0]);

        for iz in 2..nz {
            let gg1 = gg.index_axis(Axis(2), iz - 1).into_owned();
            Zip::from(gg.index_axis_mut(Axis(2), iz))
                .and(&gg1)
                .apply(|gg, gg1| *gg = (*gg - (1.0 / 6.0) * gg1) * state.spectral.htd1[iz - 1]);
        }

        {
            let gg1 = gg.index_axis(Axis(2), nz - 1).into_owned();
            Zip::from(gg.index_axis_mut(Axis(2), nz))
                .and(&gg1)
                .apply(|gg, gg1| *gg = (*gg - (1.0 / 3.0) * gg1) * state.spectral.htd1[nz - 1]);
        }

        for iz in (1..nz).rev() {
            let gg1 = gg.index_axis(Axis(2), iz + 1).into_owned();
            Zip::from(gg.index_axis_mut(Axis(2), iz))
                .and(&gg1)
                .apply(|gg, gg1| *gg += state.spectral.etd1[iz - 1] * gg1);
        }
    }

    // Transform to physical space:
    state
        .spectral
        .spctop3d(gg.view(), state.dpn.view_mut(), 1, nz);
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
