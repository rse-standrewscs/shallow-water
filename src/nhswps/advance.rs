use {
    crate::{
        constants::*,
        nhswps::{diagnose, psolve, source, Output, State},
        utils::{arr2zero, arr3zero},
    },
    anyhow::Result,
    ndarray::{Axis, Zip},
};

/// Advances fields from time t to t+dt using an iterative implicit
/// method of the form
pub fn advance(state: &mut State, output: &mut Output) -> Result<()> {
    // (F^{n+1}-F^n)/dt = L[(F^{n+1}-F^n)/2] + N[(F^{n+1}-F^n)/2]
    //
    // for a field F, where n refers to the time level, L refers to
    // the linear source terms, and N refers to the nonlinear source
    // terms.  We start with a guess for F^{n+1} in N and iterate
    // niter times (see parameter statement below).

    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let dt = 1.0 / (ng as f64);
    let dt4 = dt / 4.0;
    let dt4i = 1.0 / dt4;

    // Local variables
    let niter = 2;

    // Spectral fields needed in time stepping
    let mut qsi = arr3zero(ng, nz);
    let mut qsm = arr3zero(ng, nz);
    let mut sqs = arr3zero(ng, nz);
    let mut sds = arr3zero(ng, nz);
    let mut nds = arr3zero(ng, nz);
    let mut sgs = arr3zero(ng, nz);
    let mut ngs = arr3zero(ng, nz);

    let mut wka = arr2zero(ng);
    let mut wkb = arr2zero(ng);

    // Invert PV and compute velocity at current time level, say t=t^n:
    if state.ggen {
        state.spectral.main_invert(
            state.qs.view(),
            state.ds.view(),
            state.gs.view(),
            state.r.view_mut(),
            state.u.view_mut(),
            state.v.view_mut(),
            state.zeta.view_mut(),
        );
        psolve(state);
    }

    // If ggen is false, main_invert and psolve were called previously
    // at this time level.

    // Save various diagnostics each time step:
    diagnose(state, output)?;

    //Start with a guess for F^{n+1} for all fields:

    //Calculate the source terms (sqs,sds,sgs) for linearised PV (qs),
    //divergence (ds) and acceleration divergence (gs):
    source(state, sqs.view_mut(), sds.view_mut(), sgs.view_mut());

    //Update PV field:
    qsi.assign(&state.qs);

    Zip::from(&mut qsm)
        .and(&state.qs)
        .and(&sqs)
        .apply(|qsm, qs, sqs| *qsm = qs + dt4 * sqs);

    {
        let mut diss_broadcast = state.spectral.diss.broadcast((nz + 1, ng, ng)).unwrap();
        diss_broadcast.swap_axes(0, 2);
        diss_broadcast.swap_axes(0, 1);

        Zip::from(&mut state.qs)
            .and(diss_broadcast)
            .and(&qsm)
            .and(&sqs)
            .and(&qsi)
            .apply(|qs, diss, qsm, sqs, qsi| *qs = diss * (qsm + dt4 * sqs) - qsi);
    }

    // Update divergence and acceleration divergence:
    let dsi = state.ds.clone();
    let gsi = state.gs.clone();

    Zip::from(&mut nds)
        .and(&sds)
        .and(&dsi)
        .apply(|nds, sds, dsi| *nds = sds + dt4i * dsi);
    Zip::from(&mut ngs)
        .and(&sgs)
        .and(&gsi)
        .apply(|ngs, sgs, gsi| *ngs = sgs + dt4i * gsi);

    // 2*N_tilde_delta
    sds += &nds;
    // 2*N_tilde_gamma
    sgs += &ngs;

    wka.fill(0.0);
    wkb.fill(0.0);
    {
        let mut rdis_broadcast = state.spectral.rdis.broadcast((nz + 1, ng, ng)).unwrap();
        rdis_broadcast.swap_axes(0, 2);
        rdis_broadcast.swap_axes(0, 1);

        Zip::from(&mut state.ds)
            .and(&sgs)
            .and(&rdis_broadcast)
            .and(&sds)
            .apply(|ds, sgs, rdis, sds| *ds = sgs + rdis * sds);

        for iz in 0..=nz {
            Zip::from(&mut wka)
                .and(state.ds.index_axis(Axis(2), iz))
                .apply(|wka, ds| *wka += state.spectral.weight[iz] * ds);

            Zip::from(&mut wkb)
                .and(sds.index_axis(Axis(2), iz))
                .apply(|wkb, sds| *wkb += state.spectral.weight[iz] * sds);
        }
    }

    // fope = F operator
    wka *= &state.spectral.fope;
    // c2g2 = c^2*Lap operator
    wkb *= &state.spectral.c2g2;

    for iz in 0..=nz {
        // simp = (R^2 + f^2)^{-1}
        Zip::from(state.ds.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.simp)
            .and(&wka)
            .and(dsi.index_axis(Axis(2), iz))
            .apply(|ds, simp, wka, dsi| *ds = simp * (*ds - wka) - dsi);

        // 2*T_tilde_gamma
        Zip::from(state.gs.index_axis_mut(Axis(2), iz))
            .and(&wkb)
            .and(sds.index_axis(Axis(2), iz))
            .and(&state.spectral.rdis)
            .and(sgs.index_axis(Axis(2), iz))
            .apply(|gs, wkb, sds, rdis, sgs| *gs = wkb - FSQ * sds + rdis * sgs);
    }

    wka.fill(0.0);
    for iz in 0..=nz {
        Zip::from(&mut wka)
            .and(state.gs.index_axis(Axis(2), iz))
            .apply(|wka, gs| *wka += state.spectral.weight[iz] * gs);
    }

    // fope = F operator in paper
    wka *= &state.spectral.fope;

    for iz in 0..=nz {
        // simp = (R^2 + f^2)^{-1}
        Zip::from(state.gs.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.simp)
            .and(&wka)
            .and(gsi.index_axis(Axis(2), iz))
            .apply(|gs, simp, wka, gsi| *gs = simp * (*gs - wka) - gsi);
    }

    // Iterate to improve estimates of F^{n+1}:
    for _ in 0..niter {
        // Perform inversion at t^{n+1} from estimated quantities:
        state.spectral.main_invert(
            state.qs.view(),
            state.ds.view(),
            state.gs.view(),
            state.r.view_mut(),
            state.u.view_mut(),
            state.v.view_mut(),
            state.zeta.view_mut(),
        );

        // Compute pressure, etc:
        psolve(state);

        // Calculate the source terms (sqs,sds,sgs) for linearised PV (qs),
        // divergence (ds) and acceleration divergence (gs):
        source(state, sqs.view_mut(), sds.view_mut(), sgs.view_mut());

        // Update PV field:
        let mut diss_broadcast = state.spectral.diss.broadcast((nz + 1, ng, ng)).unwrap();
        diss_broadcast.swap_axes(0, 2);
        diss_broadcast.swap_axes(0, 1);
        Zip::from(&mut state.qs)
            .and(&diss_broadcast)
            .and(&qsm)
            .and(&sqs)
            .and(&qsi)
            .apply(|qs, diss, qsm, sqs, qsi| *qs = diss * (qsm + dt4 * sqs) - qsi);

        // Update divergence and acceleration divergence:
        // 2*N_tilde_delta
        sds += &nds;
        // 2*N_tilde_gamma
        sgs += &ngs;

        wka.fill(0.0);
        wkb.fill(0.0);

        // 2*T_tilde_delta
        let mut rdis_broadcast = state.spectral.rdis.broadcast((nz + 1, ng, ng)).unwrap();
        rdis_broadcast.swap_axes(0, 2);
        rdis_broadcast.swap_axes(0, 1);
        Zip::from(&mut state.ds)
            .and(&sgs)
            .and(&rdis_broadcast)
            .and(&sds)
            .apply(|ds, sgs, rdis, sds| *ds = sgs + rdis * sds);

        for iz in 0..=nz {
            Zip::from(&mut wka)
                .and(&state.ds.index_axis(Axis(2), iz))
                .apply(|wka, ds| *wka += state.spectral.weight[iz] * ds);
            Zip::from(&mut wkb)
                .and(&sds.index_axis(Axis(2), iz))
                .apply(|wkb, sds| *wkb += state.spectral.weight[iz] * sds);
        }

        // fope = F operator
        wka *= &state.spectral.fope;
        // c2g2 = c^2*Lap operator
        wkb *= &state.spectral.c2g2;

        for iz in 0..=nz {
            // simp = (R^2 + f^2)^{-1}
            Zip::from(&mut state.ds.index_axis_mut(Axis(2), iz))
                .and(&state.spectral.simp)
                .and(&wka)
                .and(&dsi.index_axis(Axis(2), iz))
                .apply(|ds, simp, wka, dsi| *ds = simp * (*ds - wka) - dsi);

            // 2*T_tilde_gamma
            Zip::from(&mut state.gs.index_axis_mut(Axis(2), iz))
                .and(&wkb)
                .and(&sds.index_axis(Axis(2), iz))
                .and(&state.spectral.rdis)
                .and(&sgs.index_axis(Axis(2), iz))
                .apply(|gs, wkb, sds, rdis, sgs| *gs = wkb - FSQ * sds + rdis * sgs);
        }
        wka.fill(0.0);
        for iz in 0..=nz {
            Zip::from(&mut wka)
                .and(&state.gs.index_axis(Axis(2), iz))
                .apply(|wka, gs| *wka += state.spectral.weight[iz] * gs);
        }
        // fope = F operator
        wka *= &state.spectral.fope;

        let mut simp_broadcast = state.spectral.simp.broadcast((nz + 1, ng, ng)).unwrap();
        simp_broadcast.swap_axes(0, 2);
        simp_broadcast.swap_axes(0, 1);
        let mut wka_broadcast = wka.broadcast((nz + 1, ng, ng)).unwrap();
        wka_broadcast.swap_axes(0, 2);
        wka_broadcast.swap_axes(0, 1);

        Zip::from(&mut state.gs)
            .and(simp_broadcast)
            .and(wka_broadcast)
            .and(&gsi)
            .apply(|gs, simp, wka, gsi| *gs = simp * (*gs - wka) - gsi);
    }

    // Advance time:
    state.t += dt;

    Ok(())
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
        tempdir::TempDir,
    };

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
                };
                let td = TempDir::new("advance").unwrap();
                advance(&mut state, &mut Output::from_path(td.path()).unwrap()).unwrap();
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
                };
                let td = TempDir::new("advance").unwrap();
                advance(&mut state, &mut Output::from_path(td.path()).unwrap()).unwrap();
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
