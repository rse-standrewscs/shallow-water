#![allow(clippy::cognitive_complexity)]

#[cfg(test)]
mod test;

use shallow_water::{
    constants::*,
    spectral::Spectral,
    sta2dfft::{ptospc, spctop, xderiv, yderiv},
    utils::{_2d_to_vec, _3d_to_vec, slice_to_2d, slice_to_3d},
};

#[derive(Debug, Clone, Default)]
pub struct Output {
    // Plain text diagnostics
    pub ecomp: String,   //16
    pub monitor: String, //17

    // 1D vorticity and divergence spectra
    pub spectra: String, //51

    // 3D fields
    pub d3ql: Vec<f64>, //31
    pub d3d: Vec<f64>,  //32
    pub d3g: Vec<f64>,  //33
    pub d3r: Vec<f64>,  //34
    pub d3w: Vec<f64>,  //35
    pub d3pn: Vec<f64>, //36

    // Selected vertically-averaged fields
    pub d2q: Vec<f64>,    //41
    pub d2d: Vec<f64>,    //42
    pub d2g: Vec<f64>,    //43
    pub d2h: Vec<f64>,    //44
    pub d2zeta: Vec<f64>, //45
}

#[derive(Debug, Clone)]
pub struct State {
    pub spectral: Spectral,

    // Velocity field (physical
    u: Vec<f64>,
    v: Vec<f64>,
    w: Vec<f64>,

    // Layer heights and their x & y derivatives (physical)
    z: Vec<f64>,
    zx: Vec<f64>,
    zy: Vec<f64>,

    // Dimensionless layer thickness anomaly and inverse thickness (physical)
    r: Vec<f64>,
    ri: Vec<f64>,

    // A = grad{u*rho'_theta} (spectral):
    aa: Vec<f64>,

    // Relative vertical vorticity component (physical):
    zeta: Vec<f64>,

    // Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
    pn: Vec<f64>,
    dpn: Vec<f64>,

    // Non-hydrostatic pressure (p_n) in spectral space (called ps):
    ps: Vec<f64>,

    // Prognostic fields q_l, delta and gamma (spectral):
    qs: Vec<f64>,
    ds: Vec<f64>,
    gs: Vec<f64>,

    // Time:
    t: f64,

    // Number of time steps between field saves and related indices:
    ngsave: usize,
    itime: usize,
    jtime: usize,

    // Logical for use in calling inversion routine:
    ggen: bool,

    output: Output,
}

#[allow(clippy::many_single_char_names)]
pub fn nhswps(qq: &[f64], dd: &[f64], gg: &[f64], ng: usize, nz: usize) -> Output {
    // Read linearised PV anomaly and convert to spectral space as qs

    // Parameters
    let dt = 1.0 / (ng as f64);
    let tgsave = 0.25;
    let tsim = 25.0;

    let mut state = State {
        spectral: Spectral::new(ng, nz),

        u: vec![0.0; ng * ng * (nz + 1)],
        v: vec![0.0; ng * ng * (nz + 1)],
        w: vec![0.0; ng * ng * (nz + 1)],

        // Layer heights and their x & y derivatives (physical)
        z: vec![0.0; ng * ng * (nz + 1)],
        zx: vec![0.0; ng * ng * (nz + 1)],
        zy: vec![0.0; ng * ng * (nz + 1)],

        // Dimensionless layer thickness anomaly and inverse thickness (physical)
        r: vec![0.0; ng * ng * (nz + 1)],
        ri: vec![0.0; ng * ng * (nz + 1)],

        // A: grad{u*rho'_theta} (spectral):
        aa: vec![0.0; ng * ng * (nz + 1)],

        // Relative vertical vorticity component (physical):
        zeta: vec![0.0; ng * ng * (nz + 1)],

        // Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
        pn: vec![0.0; ng * ng * (nz + 1)],
        dpn: vec![0.0; ng * ng * (nz + 1)],

        // Non-hydrostatic pressure (p_n) in spectral space (called ps):
        ps: vec![0.0; ng * ng * (nz + 1)],

        // Prognostic fields q_l, delta and gamma (spectral):
        qs: vec![0.0; ng * ng * (nz + 1)],
        ds: vec![0.0; ng * ng * (nz + 1)],
        gs: vec![0.0; ng * ng * (nz + 1)],

        // Time:
        t: 0.0,

        // Number of time steps between field saves and related indices:
        ngsave: 0,
        itime: 0,
        jtime: 0,

        // Logical for use in calling inversion routine:
        ggen: false,

        output: Output::default(),
    };

    state.spectral.ptospc3d(qq, &mut state.qs, 0, nz);
    state.qs = {
        let mut qs_matrix = slice_to_3d(&state.qs, ng, ng, nz + 1);

        for i in 0..=nz {
            qs_matrix[0][0][i] = 0.0;
        }

        _3d_to_vec(&qs_matrix)
    };

    state.spectral.ptospc3d(dd, &mut state.ds, 0, nz);
    state.ds = {
        let mut ds_matrix = slice_to_3d(&state.ds, ng, ng, nz + 1);

        for i in 0..=nz {
            ds_matrix[0][0][i] = 0.0;
        }

        _3d_to_vec(&ds_matrix)
    };

    state.spectral.ptospc3d(gg, &mut state.gs, 0, nz);
    state.gs = {
        let mut gs_matrix = slice_to_3d(&state.gs, ng, ng, nz + 1);

        for i in 0..=nz {
            gs_matrix[0][0][i] = 0.0;
        }

        _3d_to_vec(&gs_matrix)
    };

    let mut qs_matrix = slice_to_3d(&state.qs, ng, ng, nz + 1);
    let mut ds_matrix = slice_to_3d(&state.ds, ng, ng, nz + 1);
    let mut gs_matrix = slice_to_3d(&state.gs, ng, ng, nz + 1);

    for iz in 0..=nz {
        for i in 0..ng {
            for j in 0..ng {
                qs_matrix[i][j][iz] *= state.spectral.filt[i][j];
                ds_matrix[i][j][iz] *= state.spectral.filt[i][j];
                gs_matrix[i][j][iz] *= state.spectral.filt[i][j];
            }
        }
    }

    state.qs = _3d_to_vec(&qs_matrix);
    state.ds = _3d_to_vec(&ds_matrix);
    state.gs = _3d_to_vec(&gs_matrix);

    state.spectral.main_invert(
        &state.qs,
        &state.ds,
        &state.gs,
        &mut state.r,
        &mut state.u,
        &mut state.v,
        &mut state.zeta,
    );

    state.ngsave = (tgsave / dt) as usize;

    //Start the time loop:
    while state.t <= tsim {
        // Save data periodically:
        state.itime = (state.t / dt) as usize;
        state.jtime = state.itime / state.ngsave;

        if state.ngsave * state.jtime == state.itime {
            // Invert PV, divergence and acceleration divergence to obtain the
            // dimensionless layer thickness anomaly and horizontal velocity,
            // as well as the relative vertical vorticity (see spectral.f90):
            state.spectral.main_invert(
                &state.qs,
                &state.ds,
                &state.gs,
                &mut state.r,
                &mut state.u,
                &mut state.v,
                &mut state.zeta,
            );
            //Note: qs, ds & gs are in spectral space while
            //      r, u, v and zeta are in physical space.
            //Next find the non-hydrostatic pressure (pn), layer heights (z)
            // and vertical velocity (w):
            psolve(&mut state);
            // Save field data:
            savegrid(state.jtime + 1);

            state.ggen = false;
        } else {
            state.ggen = true;
        }
        // ggen is used to indicate if calling inversion is needed in advance below

        // Advect flow from time t to t + dt:
        advance(&mut state);
    }

    state.itime = (state.t / dt) as usize;
    state.jtime = state.itime / state.ngsave;
    if state.ngsave * state.jtime == state.itime {
        state.spectral.main_invert(
            &state.qs,
            &state.ds,
            &state.gs,
            &mut state.r,
            &mut state.u,
            &mut state.v,
            &mut state.zeta,
        );
        psolve(&mut state);
        savegrid(state.jtime + 1);
    }

    //finalise
    state.output
}

fn savegrid(igrid: usize) {
    dbg!(igrid);
}

/// Computes various quantities every time step to monitor the flow evolution.
fn diagnose(state: &mut State) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    // Compute maximum horizontal speed:
    let umax = state
        .u
        .iter()
        .zip(&state.v)
        .map(|(u, v)| u.powf(2.0) + v.powf(2.0))
        .fold(std::f64::NAN, f64::max)
        .sqrt();

    // Compute maximum vertical vorticity:
    let zmax = state
        .zeta
        .iter()
        .map(|x| x.abs())
        .fold(std::f64::NAN, f64::max);

    // Compute rms vertical vorticity:
    let vsumi = 1.0 / (ng * ng * nz) as f64;

    let zeta = slice_to_3d(&state.zeta, ng, ng, nz + 1);
    let mut sum = 0.0;
    for i in 0..ng {
        for j in 0..ng {
            sum += (1.0 / 2.0) * zeta[i][j][0].powf(2.0);
            sum += (1.0 / 2.0) * zeta[i][j][nz].powf(2.0);
            for k in 1..=nz - 1 {
                sum += zeta[i][j][k].powf(2.0);
            }
        }
    }
    let zrms = (vsumi * sum).sqrt();

    let s = format!(
        "{:.5}   {:.6}   {:.6}   {:.6}   {:.6}\n",
        state.t,
        (1.0 / 2.0) * (zrms.powf(2.0)),
        zrms,
        zmax,
        umax
    );

    dbg!(&s);

    state.output.monitor += &s;
}

/// Advances fields from time t to t+dt using an iterative implicit
/// method of the form
fn advance(state: &mut State) {
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
    let mut qsi = vec![0.0; ng * ng * (nz + 1)];
    let mut qsm = vec![0.0; ng * ng * (nz + 1)];
    let mut sqs = vec![0.0; ng * ng * (nz + 1)];
    let mut sds = vec![0.0; ng * ng * (nz + 1)];
    let mut nds = vec![0.0; ng * ng * (nz + 1)];
    let mut sgs = vec![0.0; ng * ng * (nz + 1)];
    let mut ngs = vec![0.0; ng * ng * (nz + 1)];

    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];

    // Invert PV and compute velocity at current time level, say t=t^n:
    if state.ggen {
        state.spectral.main_invert(
            &state.qs,
            &state.ds,
            &state.gs,
            &mut state.r,
            &mut state.u,
            &mut state.v,
            &mut state.zeta,
        );
        psolve(state);
    }

    // If ggen is false, main_invert and psolve were called previously
    // at this time level.

    // Save various diagnostics each time step:
    diagnose(state);

    //Start with a guess for F^{n+1} for all fields:

    //Calculate the source terms (sqs,sds,sgs) for linearised PV (qs),
    //divergence (ds) and acceleration divergence (gs):
    source(state, &mut sqs, &mut sds, &mut sgs);

    //Update PV field:
    for (i, e) in qsi.iter_mut().enumerate() {
        *e = state.qs[i];
    }
    for (i, e) in qsm.iter_mut().enumerate() {
        *e = state.qs[i] + dt4 * sqs[i];
    }
    state.qs = {
        let mut qs = slice_to_3d(&state.qs, ng, ng, nz + 1);
        let qsm = slice_to_3d(&qsm, ng, ng, nz + 1);
        let sqs = slice_to_3d(&sqs, ng, ng, nz + 1);
        let qsi = slice_to_3d(&qsi, ng, ng, nz + 1);
        for iz in 0..=nz {
            for i in 0..ng {
                for j in 0..ng {
                    qs[i][j][iz] = state.spectral.diss[i][j]
                        * (qsm[i][j][iz] + dt4 * sqs[i][j][iz])
                        - qsi[i][j][iz];
                }
            }
        }
        _3d_to_vec(&qs)
    };

    // Update divergence and acceleration divergence:
    let dsi = state.ds.clone();
    let gsi = state.gs.clone();
    for (i, e) in nds.iter_mut().enumerate() {
        *e = sds[i] + dt4i * dsi[i];
    }
    for (i, e) in ngs.iter_mut().enumerate() {
        *e = sgs[i] + dt4i * gsi[i];
    }
    for (i, e) in sds.iter_mut().enumerate() {
        // 2*N_tilde_delta
        *e += nds[i];
    }
    for (i, e) in sgs.iter_mut().enumerate() {
        // 2*N_tilde_gamma
        *e += ngs[i];
    }
    for e in wka.iter_mut() {
        *e = 0.0;
    }
    for e in wkb.iter_mut() {
        *e = 0.0;
    }
    for iz in 0..=nz {
        let mut ds = slice_to_3d(&state.ds, ng, ng, nz + 1);
        let mut wka_matrix = slice_to_2d(&wka, ng, ng);
        let mut wkb_matrix = slice_to_2d(&wkb, ng, ng);
        let sgs = slice_to_3d(&sgs, ng, ng, nz + 1);
        let sds = slice_to_3d(&sds, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                ds[i][j][iz] = sgs[i][j][iz] + state.spectral.rdis[i][j] * sds[i][j][iz]; // 2*T_tilde_delta
                wka_matrix[i][j] += state.spectral.weight[iz] * ds[i][j][iz];
                wkb_matrix[i][j] += state.spectral.weight[iz] * sds[i][j][iz];
            }
        }

        state.ds = _3d_to_vec(&ds);
        wka = _2d_to_vec(&wka_matrix);
        wkb = _2d_to_vec(&wkb_matrix);
    }
    wka = {
        let mut wka = slice_to_2d(&wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wka[i][j] *= state.spectral.fope[i][j];
            }
        }

        _2d_to_vec(&wka)
    }; // fope = F operator
    wkb = {
        let mut wkb = slice_to_2d(&wkb, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkb[i][j] *= state.spectral.c2g2[i][j];
            }
        }

        _2d_to_vec(&wkb)
    }; // c2g2 = c^2*Lap operator
    for iz in 0..=nz {
        let mut ds = slice_to_3d(&state.ds, ng, ng, nz + 1);
        let mut gs = slice_to_3d(&state.gs, ng, ng, nz + 1);
        let dsi = slice_to_3d(&dsi, ng, ng, nz + 1);
        let sds = slice_to_3d(&sds, ng, ng, nz + 1);
        let sgs = slice_to_3d(&sgs, ng, ng, nz + 1);
        let wka = slice_to_2d(&wka, ng, ng);
        let wkb = slice_to_2d(&wkb, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                // simp = (R^2 + f^2)^{-1}
                ds[i][j][iz] =
                    state.spectral.simp[i][j] * (ds[i][j][iz] - wka[i][j]) - dsi[i][j][iz];
                // 2*T_tilde_gamma
                gs[i][j][iz] =
                    wkb[i][j] - FSQ * sds[i][j][iz] + state.spectral.rdis[i][j] * sgs[i][j][iz];
            }
        }

        state.ds = _3d_to_vec(&ds);
        state.gs = _3d_to_vec(&gs);
    }

    for e in wka.iter_mut() {
        *e = 0.0;
    }
    for iz in 0..=nz {
        let mut wka_matrix = slice_to_2d(&wka, ng, ng);
        let gs = slice_to_3d(&state.gs, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                wka_matrix[i][j] += state.spectral.weight[iz] * gs[i][j][iz];
            }
        }

        wka = _2d_to_vec(&wka_matrix)
    }
    wka = {
        let mut wka = slice_to_2d(&wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wka[i][j] *= state.spectral.fope[i][j];
            }
        }

        _2d_to_vec(&wka)
    }; // fope = F operator in paper
    for iz in 0..=nz {
        let mut gs = slice_to_3d(&state.gs, ng, ng, nz + 1);
        let gsi = slice_to_3d(&gsi, ng, ng, nz + 1);
        let wka = slice_to_2d(&wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                // simp = (R^2 + f^2)^{-1}
                gs[i][j][iz] =
                    state.spectral.simp[i][j] * (gs[i][j][iz] - wka[i][j]) - gsi[i][j][iz];
            }
        }

        state.gs = _3d_to_vec(&gs);
    }

    // Iterate to improve estimates of F^{n+1}:
    for _ in 1..=niter {
        // Perform inversion at t^{n+1} from estimated quantities:
        state.spectral.main_invert(
            &state.qs,
            &state.ds,
            &state.gs,
            &mut state.r,
            &mut state.u,
            &mut state.v,
            &mut state.zeta,
        );

        // Compute pressure, etc:
        psolve(state);

        // Calculate the source terms (sqs,sds,sgs) for linearised PV (qs),
        // divergence (ds) and acceleration divergence (gs):
        source(state, &mut sqs, &mut sds, &mut sgs);

        // Update PV field:
        for iz in 0..=nz {
            let mut qs = slice_to_3d(&state.qs, ng, ng, nz + 1);
            let qsm = slice_to_3d(&qsm, ng, ng, nz + 1);
            let sqs = slice_to_3d(&sqs, ng, ng, nz + 1);
            let qsi = slice_to_3d(&qsi, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    qs[i][j][iz] = state.spectral.diss[i][j]
                        * (qsm[i][j][iz] + dt4 * sqs[i][j][iz])
                        - qsi[i][j][iz];
                }
            }

            state.qs = _3d_to_vec(&qs);
        }

        // Update divergence and acceleration divergence:
        for (i, e) in sds.iter_mut().enumerate() {
            // 2*N_tilde_delta
            *e += nds[i];
        }
        for (i, e) in sgs.iter_mut().enumerate() {
            // 2*N_tilde_gamma
            *e += ngs[i];
        }
        wka = vec![0.0; ng * ng];
        wkb = vec![0.0; ng * ng];
        for iz in 0..=nz {
            let mut ds = slice_to_3d(&state.ds, ng, ng, nz + 1);
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);
            let mut wkb_matrix = slice_to_2d(&wkb, ng, ng);
            let sgs = slice_to_3d(&sgs, ng, ng, nz + 1);
            let sds = slice_to_3d(&sds, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    // 2*T_tilde_delta
                    ds[i][j][iz] = sgs[i][j][iz] + state.spectral.rdis[i][j] * sds[i][j][iz];
                    wka_matrix[i][j] += state.spectral.weight[iz] * ds[i][j][iz];
                    wkb_matrix[i][j] += state.spectral.weight[iz] * sds[i][j][iz];
                }
            }

            state.ds = _3d_to_vec(&ds);
            wka = _2d_to_vec(&wka_matrix);
            wkb = _2d_to_vec(&wkb_matrix);
        }

        {
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);
            let mut wkb_matrix = slice_to_2d(&wkb, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    // fope = F operator
                    wka_matrix[i][j] *= state.spectral.fope[i][j];
                    // c2g2 = c^2*Lap operator
                    wkb_matrix[i][j] *= state.spectral.c2g2[i][j];
                }
                wka = _2d_to_vec(&wka_matrix);
                wkb = _2d_to_vec(&wkb_matrix);
            }
        }
        for iz in 0..=nz {
            let mut ds = slice_to_3d(&state.ds, ng, ng, nz + 1);
            let mut gs = slice_to_3d(&state.gs, ng, ng, nz + 1);
            let dsi = slice_to_3d(&dsi, ng, ng, nz + 1);
            let sgs = slice_to_3d(&sgs, ng, ng, nz + 1);
            let sds = slice_to_3d(&sds, ng, ng, nz + 1);
            let wka = slice_to_2d(&wka, ng, ng);
            let wkb = slice_to_2d(&wkb, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    // simp = (R^2 + f^2)^{-1}
                    ds[i][j][iz] =
                        state.spectral.simp[i][j] * (ds[i][j][iz] - wka[i][j]) - dsi[i][j][iz];
                    // 2*T_tilde_gamma
                    gs[i][j][iz] =
                        wkb[i][j] - FSQ * sds[i][j][iz] + state.spectral.rdis[i][j] * sgs[i][j][iz];
                }
            }

            state.ds = _3d_to_vec(&ds);
            state.gs = _3d_to_vec(&gs);
        }
        wka = vec![0.0; ng * ng];
        for iz in 0..=nz {
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);
            let gs = slice_to_3d(&state.gs, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[i][j] += state.spectral.weight[iz] * gs[i][j][iz];
                }
            }
            wka = _2d_to_vec(&wka_matrix);
        }
        {
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    // fope = F operator in paper
                    wka_matrix[i][j] *= state.spectral.fope[i][j];
                }
            }

            wka = _2d_to_vec(&wka_matrix);
        }
        for iz in 0..=nz {
            let mut gs = slice_to_3d(&state.gs, ng, ng, nz + 1);
            let gsi = slice_to_3d(&gsi, ng, ng, nz + 1);
            let wka = slice_to_2d(&wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    // simp = (R^2 + f^2)^{-1}
                    gs[i][j][iz] =
                        state.spectral.simp[i][j] * (gs[i][j][iz] - wka[i][j]) - gsi[i][j][iz];
                }
            }

            state.gs = _3d_to_vec(&gs);
        }
    }

    // Advance time:
    state.t += dt;
}

/// Solves for the nonhydrostatic part of the pressure (pn) given
/// the velocity field (u,v,w) together with r = rho'_theta and
/// z = theta + int_0^theta{rho'_theta(s)ds}.
fn psolve(state: &mut State) {
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
    for (i, e) in state.ri.iter_mut().enumerate() {
        *e = 1.0 / (1.0 + state.r[i]);
    }
    state.spectral.deal3d(&mut state.ri);

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
        state.spectral.ptospc3d(&state.pn, &mut state.ps, 0, nz - 1);
        {
            let mut ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    ps_matrix[i][j][nz] = 0.0;
                }
            }
            state.ps = _3d_to_vec(&ps_matrix);
        }

        // Compute pressure derivatives needed in the non-constant part of the
        // source S_1 and add to S_0 (in sp0) to form total source S (sp):

        // Lower boundary at iz = 0 (use dp/dtheta = 0):
        // d^2p/dtheta^2:
        {
            let mut wkd_matrix = slice_to_2d(&wkd, ng, ng);
            let ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkd_matrix[i][j] = (2.0 * ps_matrix[i][j][0] - 5.0 * ps_matrix[i][j][1]
                        + 4.0 * ps_matrix[i][j][2]
                        - ps_matrix[i][j][3])
                        * dzisq;
                }
            }
            wkd = _2d_to_vec(&wkd_matrix);
        }

        // Return to physical space:
        spctop(
            ng,
            ng,
            &mut wkd,
            &mut d2pdt2,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        // Total source:
        {
            let mut wkp_matrix = slice_to_2d(&wkp, ng, ng);
            let sp0_matrix = slice_to_3d(&sp0, ng, ng, nz + 1);
            let cpt2_matrix = slice_to_3d(&cpt2, ng, ng, nz + 1);
            let d2pdt2_matrix = slice_to_2d(&d2pdt2, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkp_matrix[i][j] =
                        sp0_matrix[i][j][0] + cpt2_matrix[i][j][0] * d2pdt2_matrix[i][j];
                }
            }

            wkp = _2d_to_vec(&wkp_matrix);
        }
        // Transform to spectral space for inversion below:
        ptospc(
            ng,
            ng,
            &mut wkp,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        {
            let mut sp_matrix = slice_to_3d(&sp, ng, ng, nz + 1);
            let wka_matrix = slice_to_2d(&wka, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    sp_matrix[i][j][0] = wka_matrix[i][j];
                }
            }
            sp = _3d_to_vec(&sp_matrix);
        }

        // Interior grid points:
        for iz in 1..=nz - 1 {
            wkq = d2pdt2.clone();
            wka = {
                let mut wka = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
                let ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);

                for i in 0..ng {
                    for j in 0..ng {
                        wka[i][j] = (ps_matrix[i][j][iz + 1] - ps_matrix[i][j][iz - 1]) * hdzi;
                    }
                }

                _2d_to_vec(&wka)
            };
            wkd = {
                let mut wkd = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
                let ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);

                for i in 0..ng {
                    for j in 0..ng {
                        wkd[i][j] = (ps_matrix[i][j][iz + 1] - 2.0 * ps_matrix[i][j][iz]
                            + ps_matrix[i][j][iz - 1])
                            * dzisq;
                    }
                }

                _2d_to_vec(&wkd)
            };

            // Calculate x & y derivatives of dp/dtheta:
            xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
            yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkc);
            // Return to physical space:
            spctop(
                ng,
                ng,
                &mut wka,
                &mut dpdt,
                &state.spectral.xfactors,
                &state.spectral.yfactors,
                &state.spectral.xtrig,
                &state.spectral.ytrig,
            );
            spctop(
                ng,
                ng,
                &mut wkb,
                &mut d2pdxt,
                &state.spectral.xfactors,
                &state.spectral.yfactors,
                &state.spectral.xtrig,
                &state.spectral.ytrig,
            );
            spctop(
                ng,
                ng,
                &mut wkc,
                &mut d2pdyt,
                &state.spectral.xfactors,
                &state.spectral.yfactors,
                &state.spectral.xtrig,
                &state.spectral.ytrig,
            );
            spctop(
                ng,
                ng,
                &mut wkd,
                &mut d2pdt2,
                &state.spectral.xfactors,
                &state.spectral.yfactors,
                &state.spectral.xtrig,
                &state.spectral.ytrig,
            );

            // Total source:
            wkp = {
                let mut wkp = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
                let sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
                let sigx = slice_to_3d(&sigx, ng, ng, nz + 1);
                let sigy = slice_to_3d(&sigy, ng, ng, nz + 1);
                let cpt1 = slice_to_3d(&cpt1, ng, ng, nz + 1);
                let cpt2 = slice_to_3d(&cpt2, ng, ng, nz + 1);
                let d2pdxt = slice_to_2d(&d2pdxt, ng, ng);
                let d2pdyt = slice_to_2d(&d2pdyt, ng, ng);
                let d2pdt2 = slice_to_2d(&d2pdt2, ng, ng);
                let dpdt = slice_to_2d(&dpdt, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        wkp[i][j] = sp0[i][j][iz]
                            + sigx[i][j][iz] * d2pdxt[i][j]
                            + sigy[i][j][iz] * d2pdyt[i][j]
                            + cpt2[i][j][iz] * d2pdt2[i][j]
                            + cpt1[i][j][iz] * dpdt[i][j];
                    }
                }

                _2d_to_vec(&wkp)
            };

            // Transform to spectral space for inversion below:
            ptospc(
                ng,
                ng,
                &mut wkp,
                &mut wka,
                &state.spectral.xfactors,
                &state.spectral.yfactors,
                &state.spectral.xtrig,
                &state.spectral.ytrig,
            );
            sp = {
                let mut sp_matrix = slice_to_3d(&sp, ng, ng, nz + 1);
                let wka_matrix = slice_to_2d(&wka, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        sp_matrix[i][j][iz] = wka_matrix[i][j];
                    }
                }

                _3d_to_vec(&sp_matrix)
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
        ptospc(
            ng,
            ng,
            &mut wkp,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        // Calculate x & y derivatives of dp/dtheta:
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkc);
        // Return to physical space:
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut d2pdxt,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        spctop(
            ng,
            ng,
            &mut wkc,
            &mut d2pdyt,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        // Total source:
        wkp = {
            let mut wkp = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
            let sigx = slice_to_3d(&sigx, ng, ng, nz + 1);
            let sigy = slice_to_3d(&sigy, ng, ng, nz + 1);
            let cpt1 = slice_to_3d(&cpt1, ng, ng, nz + 1);
            let cpt2 = slice_to_3d(&cpt2, ng, ng, nz + 1);
            let d2pdxt = slice_to_2d(&d2pdxt, ng, ng);
            let d2pdyt = slice_to_2d(&d2pdyt, ng, ng);
            let d2pdt2 = slice_to_2d(&d2pdt2, ng, ng);
            let dpdt = slice_to_2d(&dpdt, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkp[i][j] = sp0[i][j][nz]
                        + sigx[i][j][nz] * d2pdxt[i][j]
                        + sigy[i][j][nz] * d2pdyt[i][j]
                        + cpt2[i][j][nz] * d2pdt2[i][j]
                        + cpt1[i][j][nz] * dpdt[i][j];
                }
            }

            _2d_to_vec(&wkp)
        };

        // Transform to spectral space for inversion below:
        ptospc(
            ng,
            ng,
            &mut wkp,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        sp = {
            let mut sp_matrix = slice_to_3d(&sp, ng, ng, nz + 1);
            let wka_matrix = slice_to_2d(&wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    sp_matrix[i][j][nz] = wka_matrix[i][j];
                }
            }

            _3d_to_vec(&sp_matrix)
        };

        // Solve tridiagonal problem for pressure in spectral space:
        {
            let mut gg_matrix = slice_to_3d(&gg, ng, ng, nz + 1);
            let sp_matrix = slice_to_3d(&sp, ng, ng, nz + 1);
            let mut ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    gg_matrix[i][j][0] =
                        (1.0 / 3.0) * sp_matrix[i][j][0] + (1.0 / 6.0) * sp_matrix[i][j][1];
                }
            }

            for iz in 1..=nz - 1 {
                for i in 0..ng {
                    for j in 0..ng {
                        gg_matrix[i][j][iz] = (1.0 / 12.0)
                            * (sp_matrix[i][j][iz - 1] + sp_matrix[i][j][iz + 1])
                            + (5.0 / 6.0) * sp_matrix[i][j][iz];
                    }
                }
            }

            for i in 0..ng {
                for j in 0..ng {
                    ps_matrix[i][j][0] = gg_matrix[i][j][0] * state.spectral.htdv[i][j][0];
                }
            }
            for iz in 1..=nz - 1 {
                for i in 0..ng {
                    for j in 0..ng {
                        ps_matrix[i][j][iz] = (gg_matrix[i][j][iz]
                            - state.spectral.ap[i][j] * ps_matrix[i][j][iz - 1])
                            * state.spectral.htdv[i][j][iz];
                    }
                }
            }
            for iz in (0..=nz - 2).rev() {
                for i in 0..ng {
                    for j in 0..ng {
                        ps_matrix[i][j][iz] +=
                            state.spectral.etdv[i][j][iz] * ps_matrix[i][j][iz + 1];
                    }
                }
            }
            //dbg!(&_3d_to_vec(&ps_matrix));

            for i in 0..ng {
                for j in 0..ng {
                    ps_matrix[i][j][nz] = 0.0;
                }
            }
            state.ps = _3d_to_vec(&ps_matrix);
            gg = _3d_to_vec(&gg_matrix);

            // Transform to physical space:
            state.spectral.spctop3d(&state.ps, &mut state.pn, 0, nz - 1);

            state.pn = {
                let mut pn_matrix = slice_to_3d(&state.pn, ng, ng, nz + 1);
                for i in 0..ng {
                    for j in 0..ng {
                        pn_matrix[i][j][nz] = 0.0;
                    }
                }
                _3d_to_vec(&pn_matrix)
            };
        }

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
            panic!(format!(
                "Pressure error too large! Final pressure error = {}",
                errp
            ));
        }

        iter += 1;

        // Reset pna:
        pna = state.pn.clone();
    }

    // Past this point, we have converged!

    // Calculate 1st derivative of pressure using 4th-order compact differences:
    gg = {
        let mut gg = slice_to_3d(&gg, ng, ng, nz + 1);
        let ps = slice_to_3d(&state.ps, ng, ng, nz + 1);
        let sp = slice_to_3d(&sp, ng, ng, nz + 1);

        for iz in 1..=nz - 1 {
            for i in 0..ng {
                for j in 0..ng {
                    gg[i][j][iz] = (ps[i][j][iz + 1] - ps[i][j][iz - 1]) * hdzi;
                }
            }
        }
        for i in 0..ng {
            for j in 0..ng {
                gg[i][j][nz] = dz6 * sp[i][j][nz] - ps[i][j][nz - 1] * dzi;
            }
        }

        for i in 0..ng {
            for j in 0..ng {
                gg[i][j][1] *= state.spectral.htd1[0];
            }
        }
        for iz in 2..=nz - 1 {
            for i in 0..ng {
                for j in 0..ng {
                    gg[i][j][iz] = (gg[i][j][iz] - (1.0 / 6.0) * gg[i][j][iz - 1])
                        * state.spectral.htd1[iz - 1];
                }
            }
        }

        for i in 0..ng {
            for j in 0..ng {
                gg[i][j][nz] =
                    (gg[i][j][nz] - (1.0 / 3.0) * gg[i][j][nz - 1]) * state.spectral.htd1[nz - 1];
            }
        }
        for iz in (1..=nz - 1).rev() {
            for i in 0..ng {
                for j in 0..ng {
                    gg[i][j][iz] += state.spectral.etd1[iz - 1] * gg[i][j][iz + 1];
                }
            }
        }

        _3d_to_vec(&gg)
    };

    // Transform to physical space:
    state.spectral.spctop3d(&gg, &mut state.dpn, 1, nz);
}

/// Finds the part of the pressure source which does not vary
/// in the iteration to find the pressure.
fn cpsource(state: &State, sp0: &mut [f64]) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let hdzi = (1.0 / 2.0) * (1.0 / (HBAR / nz as f64));

    // Physical space arrays
    let mut ut = vec![0.0; ng * ng * (nz + 1)];
    let mut vt = vec![0.0; ng * ng * (nz + 1)];
    let mut wt = vec![0.0; ng * ng * (nz + 1)];
    let mut ux = vec![0.0; ng * ng];
    let mut uy = vec![0.0; ng * ng];
    let mut vx = vec![0.0; ng * ng];
    let mut vy = vec![0.0; ng * ng];
    let mut wx = vec![0.0; ng * ng];
    let mut wy = vec![0.0; ng * ng];
    let mut hsrc = vec![0.0; ng * ng];
    let mut wkp = vec![0.0; ng * ng];
    let mut wkq = vec![0.0; ng * ng];
    let mut wkr = vec![0.0; ng * ng];

    // Spectral space arrays (all work arrays)
    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];

    // Calculate part which is independent of z, -g*Lap_h{h}:
    // wkp = h;
    wkp = {
        let z_matrix = slice_to_3d(&state.z, ng, ng, nz + 1);
        let mut wkp_matrix = slice_to_2d(&wkp, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkp_matrix[i][j] = z_matrix[i][j][nz];
            }
        }

        _2d_to_vec(&wkp_matrix)
    };

    // Fourier transform to spectral space:
    ptospc(
        ng,
        ng,
        &mut wkp,
        &mut wka,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    // Apply -g*Lap_h operator:
    wka = {
        let mut wka_matrix = slice_to_2d(&wka, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                wka_matrix[i][j] *= state.spectral.glap[i][j];
            }
        }
        _2d_to_vec(&wka_matrix)
    };
    // Return to physical space:
    spctop(
        ng,
        ng,
        &mut wka,
        &mut hsrc,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    // hsrc contains -g*Lap{h} in physical space.

    // Calculate u_theta, v_theta & w_theta:
    let mut ut_matrix = slice_to_3d(&ut, ng, ng, nz + 1);
    let mut vt_matrix = slice_to_3d(&vt, ng, ng, nz + 1);
    let mut wt_matrix = slice_to_3d(&wt, ng, ng, nz + 1);
    let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);
    let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);
    let w_matrix = slice_to_3d(&state.w, ng, ng, nz + 1);

    // Lower boundary (use higher order formula):
    for i in 0..ng {
        for j in 0..ng {
            ut_matrix[i][j][0] =
                hdzi * (4.0 * u_matrix[i][j][1] - 3.0 * u_matrix[i][j][0] - u_matrix[i][j][2]);
            vt_matrix[i][j][0] =
                hdzi * (4.0 * v_matrix[i][j][1] - 3.0 * v_matrix[i][j][0] - v_matrix[i][j][2]);
            wt_matrix[i][j][0] =
                hdzi * (4.0 * w_matrix[i][j][1] - 3.0 * w_matrix[i][j][0] - w_matrix[i][j][2]);
        }
    }

    // Interior (centred differencing):
    for iz in 1..nz {
        for i in 0..ng {
            for j in 0..ng {
                ut_matrix[i][j][iz] = hdzi * (u_matrix[i][j][iz + 1] - u_matrix[i][j][iz - 1]);
                vt_matrix[i][j][iz] = hdzi * (v_matrix[i][j][iz + 1] - v_matrix[i][j][iz - 1]);
                wt_matrix[i][j][iz] = hdzi * (w_matrix[i][j][iz + 1] - w_matrix[i][j][iz - 1]);
            }
        }
    }

    // Upper boundary (use higher order formula):
    for i in 0..ng {
        for j in 0..ng {
            ut_matrix[i][j][nz] = hdzi
                * (3.0 * u_matrix[i][j][nz] + u_matrix[i][j][nz - 2]
                    - 4.0 * u_matrix[i][j][nz - 1]);
            vt_matrix[i][j][nz] = hdzi
                * (3.0 * v_matrix[i][j][nz] + v_matrix[i][j][nz - 2]
                    - 4.0 * v_matrix[i][j][nz - 1]);
            wt_matrix[i][j][nz] = hdzi
                * (3.0 * w_matrix[i][j][nz] + w_matrix[i][j][nz - 2]
                    - 4.0 * w_matrix[i][j][nz - 1]);
        }
    }

    ut = _3d_to_vec(&ut_matrix);
    vt = _3d_to_vec(&vt_matrix);
    wt = _3d_to_vec(&wt_matrix);

    // Loop over layers and build up source, sp0:

    // iz = 0 is much simpler as z = w = 0 there:
    // Calculate u_x, u_y, v_x & v_y:
    wkq = {
        let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
        let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[i][j] = u_matrix[i][j][0];
            }
        }

        _2d_to_vec(&wkq_matrix)
    };
    ptospc(
        ng,
        ng,
        &mut wkq,
        &mut wka,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
    spctop(
        ng,
        ng,
        &mut wkb,
        &mut ux,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
    spctop(
        ng,
        ng,
        &mut wkb,
        &mut uy,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    wkq = {
        let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
        let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[i][j] = v_matrix[i][j][0];
            }
        }

        _2d_to_vec(&wkq_matrix)
    };
    ptospc(
        ng,
        ng,
        &mut wkq,
        &mut wka,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
    spctop(
        ng,
        ng,
        &mut wkb,
        &mut vx,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
    spctop(
        ng,
        ng,
        &mut wkb,
        &mut vy,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    wkq = {
        let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
        let ri_matrix = slice_to_3d(&state.ri, ng, ng, nz + 1);
        let wt_matrix = slice_to_3d(&wt, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[i][j] = ri_matrix[i][j][0] * wt_matrix[i][j][0];
            }
        }

        _2d_to_vec(&wkq_matrix)
    };
    state.spectral.deal2d(&mut wkq);
    {
        let zeta2d = {
            let mut zeta2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let zeta_matrix = slice_to_3d(&state.zeta, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    zeta2d[i][j] = zeta_matrix[i][j][0];
                }
            }
            _2d_to_vec(&zeta2d)
        };
        let mut d2sp0 = {
            let sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
            let mut d2sp0 = slice_to_2d(&vec![0.0; ng * ng], ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    d2sp0[i][j] = sp0[i][j][0];
                }
            }

            _2d_to_vec(&d2sp0)
        };
        for (i, e) in d2sp0.iter_mut().enumerate() {
            *e = hsrc[i]
                + COF * zeta2d[i]
                + 2.0 * (ux[i] * vy[i] - uy[i] * vx[i] + wkq[i] * (ux[i] + vy[i]));
        }
        let mut d3sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
        let d2sp0 = slice_to_2d(&d2sp0, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                d3sp0[i][j][0] = d2sp0[i][j];
            }
        }
        for (i, e) in _3d_to_vec(&d3sp0).iter().enumerate() {
            sp0[i] = *e;
        }
    }

    for iz in 1..=nz {
        // Calculate u_x, u_y, v_x, v_y, w_x, w_y:
        wkq = {
            let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
            let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[i][j] = u_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&wkq_matrix)
        };
        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut ux,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut uy,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        wkq = {
            let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
            let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[i][j] = v_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&wkq_matrix)
        };
        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut vx,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut vy,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        wkq = {
            let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
            let w_matrix = slice_to_3d(&state.w, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[i][j] = w_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&wkq_matrix)
        };
        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut wx,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut wy,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        // Calculate pressure source:
        let vt2d = {
            let mut vt2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let vt_matrix = slice_to_3d(&vt, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    vt2d[i][j] = vt_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&vt2d)
        };
        let ut2d = {
            let mut ut2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let ut_matrix = slice_to_3d(&ut, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    ut2d[i][j] = ut_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&ut2d)
        };
        let wt2d = {
            let mut wt2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let wt_matrix = slice_to_3d(&wt, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wt2d[i][j] = wt_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&wt2d)
        };
        let zx2d = {
            let mut zx2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let zx_matrix = slice_to_3d(&state.zx, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    zx2d[i][j] = zx_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&zx2d)
        };
        let zy2d = {
            let mut zy2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
            let zy_matrix = slice_to_3d(&state.zy, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    zy2d[i][j] = zy_matrix[i][j][iz];
                }
            }

            _2d_to_vec(&zy2d)
        };
        for (i, e) in wkp.iter_mut().enumerate() {
            *e = vt2d[i] * zx2d[i] - ut2d[i] * zy2d[i];
        }
        state.spectral.deal2d(&mut wkp);

        for (i, e) in wkq.iter_mut().enumerate() {
            *e = uy[i] * vt2d[i] - ut2d[i] * vy[i];
        }
        state.spectral.deal2d(&mut wkq);

        for (i, e) in wkr.iter_mut().enumerate() {
            *e = ut2d[i] * vx[i] - ux[i] * vt2d[i];
        }
        state.spectral.deal2d(&mut wkr);

        for (i, e) in wkq.iter_mut().enumerate() {
            *e = *e * zx2d[i] + wkr[i] * zy2d[i] + (ux[i] + vy[i]) * wt2d[i]
                - wx[i] * ut2d[i]
                - wy[i] * vt2d[i];
        }
        state.spectral.deal2d(&mut wkq);
        //sp0(:,:,iz)=hsrc+cof*(zeta(:,:,iz)-ri(:,:,iz)*wkp)+ two*(ux*vy-uy*vx+ri(:,:,iz)*wkq);
        {
            let zeta2d = {
                let mut zeta2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
                let zeta_matrix = slice_to_3d(&state.zeta, ng, ng, nz + 1);
                for i in 0..ng {
                    for j in 0..ng {
                        zeta2d[i][j] = zeta_matrix[i][j][iz];
                    }
                }
                _2d_to_vec(&zeta2d)
            };

            let ri2d = {
                let mut ri2d = slice_to_2d(&vec![0.0; ng * ng], ng, ng);
                let ri_matrix = slice_to_3d(&state.ri, ng, ng, nz + 1);
                for i in 0..ng {
                    for j in 0..ng {
                        ri2d[i][j] = ri_matrix[i][j][iz];
                    }
                }
                _2d_to_vec(&ri2d)
            };

            let mut d2sp0 = {
                let sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
                let mut d2sp0 = slice_to_2d(&vec![0.0; ng * ng], ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        d2sp0[i][j] = sp0[i][j][0];
                    }
                }

                _2d_to_vec(&d2sp0)
            };

            for (i, e) in d2sp0.iter_mut().enumerate() {
                *e = hsrc[i]
                    + COF * (zeta2d[i] - ri2d[i] * wkp[i])
                    + 2.0 * (ux[i] * vy[i] - uy[i] * vx[i] + ri2d[i] * wkq[i]);
            }
            let mut d3sp0 = slice_to_3d(&sp0, ng, ng, nz + 1);
            let d2sp0 = slice_to_2d(&d2sp0, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    d3sp0[i][j][iz] = d2sp0[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&d3sp0).iter().enumerate() {
                sp0[i] = *e;
            }
        }
    }
}

/// Calculates the fixed coefficients used in the pressure iteration.
fn coeffs(state: &State, sigx: &mut [f64], sigy: &mut [f64], cpt1: &mut [f64], cpt2: &mut [f64]) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let qdzi = (1.0 / 4.0) * (1.0 / (HBAR / nz as f64));
    let mut wkp = vec![0.0; ng * ng];
    let mut wka = vec![0.0; ng * ng];

    // Compute sigx and sigy and de-alias:
    for (i, e) in sigx.iter_mut().enumerate() {
        *e = state.ri[i] * state.zx[i];
    }
    for (i, e) in sigy.iter_mut().enumerate() {
        *e = state.ri[i] * state.zy[i];
    }
    state.spectral.deal3d(sigx);
    state.spectral.deal3d(sigy);

    // Compute cpt2 and de-alias:
    for (i, e) in cpt2.iter_mut().enumerate() {
        *e = 1.0 - state.ri[i].powf(2.0) - sigx[i].powf(2.0) - sigy[i].powf(2.0);
    }
    state.spectral.deal3d(cpt2);

    // Calculate 0.5*d(cpt2)/dtheta + div(sigx,sigy) and store in cpt1:

    // Lower boundary (use higher order formula):
    {
        let mut cpt1_matrix = slice_to_3d(&cpt1, ng, ng, nz + 1);
        let cpt2_matrix = slice_to_3d(&cpt2, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                cpt1_matrix[i][j][0] = qdzi
                    * (4.0 * cpt2_matrix[i][j][1]
                        - 3.0 * cpt2_matrix[i][j][0]
                        - cpt2_matrix[i][j][2]);
            }
        }
        for (i, e) in _3d_to_vec(&cpt1_matrix).iter_mut().enumerate() {
            cpt1[i] = *e;
        }
    }
    // qdzi=1/(4*dz) is used since 0.5*d/dtheta is being computed.

    // Interior (centred differencing):
    for iz in 1..nz {
        let d3_sigx = slice_to_3d(sigx, ng, ng, nz + 1);
        let d3_sigy = slice_to_3d(sigy, ng, ng, nz + 1);
        let mut d2_sigx = vec![vec![0.0; ng]; ng];
        let mut d2_sigy = vec![vec![0.0; ng]; ng];
        for i in 0..ng {
            for j in 0..ng {
                d2_sigx[i][j] = d3_sigx[i][j][iz];
                d2_sigy[i][j] = d3_sigy[i][j][iz];
            }
        }
        state
            .spectral
            .divs(&_2d_to_vec(&d2_sigx), &_2d_to_vec(&d2_sigy), &mut wka);
        spctop(
            ng,
            ng,
            &mut wka,
            &mut wkp,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        {
            let mut cpt1_matrix = slice_to_3d(&cpt1, ng, ng, nz + 1);
            let cpt2_matrix = slice_to_3d(&cpt2, ng, ng, nz + 1);
            let wkp_matrix = slice_to_2d(&wkp, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    cpt1_matrix[i][j][iz] = qdzi
                        * (cpt2_matrix[i][j][iz + 1] - cpt2_matrix[i][j][iz - 1])
                        + wkp_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&cpt1_matrix).iter_mut().enumerate() {
                cpt1[i] = *e;
            }
        }
    }

    // Upper boundary (use higher order formula):
    {
        let d3_sigx = slice_to_3d(sigx, ng, ng, nz + 1);
        let d3_sigy = slice_to_3d(sigy, ng, ng, nz + 1);
        let mut d2_sigx = vec![vec![0.0; ng]; ng];
        let mut d2_sigy = vec![vec![0.0; ng]; ng];
        for i in 0..ng {
            for j in 0..ng {
                d2_sigx[i][j] = d3_sigx[i][j][nz];
                d2_sigy[i][j] = d3_sigy[i][j][nz];
            }
        }
        state
            .spectral
            .divs(&_2d_to_vec(&d2_sigx), &_2d_to_vec(&d2_sigy), &mut wka);
    }
    spctop(
        ng,
        ng,
        &mut wka,
        &mut wkp,
        &state.spectral.xfactors,
        &state.spectral.yfactors,
        &state.spectral.xtrig,
        &state.spectral.ytrig,
    );
    {
        let mut cpt1_matrix = slice_to_3d(&cpt1, ng, ng, nz + 1);
        let cpt2_matrix = slice_to_3d(&cpt2, ng, ng, nz + 1);
        let wkp_matrix = slice_to_2d(&wkp, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                cpt1_matrix[i][j][nz] = qdzi
                    * (3.0 * cpt2_matrix[i][j][nz] + cpt2_matrix[i][j][nz - 2]
                        - 4.0 * cpt2_matrix[i][j][nz - 1])
                    + wkp_matrix[i][j];
            }
        }
        for (i, e) in _3d_to_vec(&cpt1_matrix).iter_mut().enumerate() {
            cpt1[i] = *e;
        }
    };
    // Re-define sigx and sigy to include a factor of 2:
    for e in sigx.iter_mut() {
        *e *= 2.0;
    }
    for e in sigy.iter_mut() {
        *e *= 2.0;
    }
}

/// Calculates layer heights (z), as well as dz/dx & dz/dy (zx & zy),
/// the vertical velocity (w), and the A = grad{u*rho'_theta} (aa).
fn vertical(state: &mut State) {
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
    state.z = {
        let mut z_matrix = slice_to_3d(&state.z, ng, ng, nz + 1);
        let r_matrix = slice_to_3d(&state.r, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                z_matrix[i][j][1] = dz2 * (r_matrix[i][j][0] + r_matrix[i][j][1]);
            }
        }

        for iz in 1..nz {
            for i in 0..ng {
                for j in 0..ng {
                    z_matrix[i][j][iz + 1] =
                        z_matrix[i][j][iz] + dz2 * (r_matrix[i][j][iz] + r_matrix[i][j][iz + 1]);
                }
            }
        }

        _3d_to_vec(&z_matrix)
    };

    for iz in 1..=nz {
        // Add on theta (a linear function) to complete definition of z:
        let mut z_matrix = slice_to_3d(&state.z, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                z_matrix[i][j][iz] += state.spectral.theta[iz];
            }
        }
        state.z = _3d_to_vec(&z_matrix);

        // Calculate z_x & z_y:
        let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
        let z_matrix = slice_to_3d(&state.z, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[i][j] = z_matrix[i][j][iz];
            }
        }
        wkq = _2d_to_vec(&wkq_matrix);

        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut wkq,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        state.zx = {
            let mut zx_matrix = slice_to_3d(&state.zx, ng, ng, nz + 1);
            let wkq_matrix = slice_to_2d(&wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    zx_matrix[i][j][iz] = wkq_matrix[i][j];
                }
            }
            _3d_to_vec(&zx_matrix)
        };
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkb);
        spctop(
            ng,
            ng,
            &mut wkb,
            &mut wkq,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        state.zy = {
            let mut zy_matrix = slice_to_3d(&state.zy, ng, ng, nz + 1);
            let wkq_matrix = slice_to_2d(&wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    zy_matrix[i][j][iz] = wkq_matrix[i][j];
                }
            }
            _3d_to_vec(&zy_matrix)
        };
    }

    // Calculate A = grad{u*rho'_theta} (spectral):
    for iz in 0..=nz {
        // Calculate (u*rho'_theta)_x:
        wkq = {
            let mut wkq = slice_to_2d(&wkq, ng, ng);
            let u = slice_to_3d(&state.u, ng, ng, nz + 1);
            let r = slice_to_3d(&state.r, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wkq[i][j] = u[i][j][iz] * r[i][j][iz];
                }
            }
            _2d_to_vec(&wkq)
        };
        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        xderiv(ng, ng, &state.spectral.hrkx, &wka, &mut wkb);

        // Calculate (v*rho'_theta)_y:
        wkq = {
            let mut wkq = slice_to_2d(&wkq, ng, ng);
            let v = slice_to_3d(&state.v, ng, ng, nz + 1);
            let r = slice_to_3d(&state.r, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wkq[i][j] = v[i][j][iz] * r[i][j][iz];
                }
            }
            _2d_to_vec(&wkq)
        };
        ptospc(
            ng,
            ng,
            &mut wkq,
            &mut wka,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        yderiv(ng, ng, &state.spectral.hrky, &wka, &mut wkc);

        // Apply de-aliasing filter and complete definition of A:
        state.aa = {
            let mut aa = slice_to_3d(&state.aa, ng, ng, nz + 1);
            let wkb = slice_to_2d(&wkb, ng, ng);
            let wkc = slice_to_2d(&wkc, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    aa[i][j][iz] = state.spectral.filt[i][j] * (wkb[i][j] + wkc[i][j]);
                }
            }

            _3d_to_vec(&aa)
        };
        // Need -(A + delta) in physical space for computing w just below:
        wka = {
            let mut wka = slice_to_2d(&wka, ng, ng);
            let aa = slice_to_3d(&state.aa, ng, ng, nz + 1);
            let ds = slice_to_3d(&state.ds, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wka[i][j] = aa[i][j][iz] + ds[i][j][iz];
                }
            }

            _2d_to_vec(&wka)
        };
        spctop(
            ng,
            ng,
            &mut wka,
            &mut wkq,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        rsrc = {
            let mut rsrc = slice_to_3d(&rsrc, ng, ng, nz + 1);
            let wkq = slice_to_2d(&wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    rsrc[i][j][iz] = -wkq[i][j];
                }
            }

            _3d_to_vec(&rsrc)
        }
    }

    // Calculate vertical velocity (0 at iz = 0):
    state.w = {
        let mut w = slice_to_3d(&state.w, ng, ng, nz + 1);
        let rsrc = slice_to_3d(&rsrc, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                w[i][j][1] = dz2 * (rsrc[i][j][0] + rsrc[i][j][1]);
            }
        }
        _3d_to_vec(&w)
    };
    for iz in 1..nz {
        state.w = {
            let mut w = slice_to_3d(&state.w, ng, ng, nz + 1);
            let rsrc = slice_to_3d(&rsrc, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    w[i][j][iz + 1] = w[i][j][iz] + dz2 * (rsrc[i][j][iz] + rsrc[i][j][iz + 1]);
                }
            }
            _3d_to_vec(&w)
        };
    }

    // Complete definition of w by adding u*z_x + v*z_y after de-aliasing:
    for iz in 1..=nz {
        wkq = {
            let mut wkq = slice_to_2d(&wkq, ng, ng);
            let u = slice_to_3d(&state.u, ng, ng, nz + 1);
            let v = slice_to_3d(&state.v, ng, ng, nz + 1);
            let zx = slice_to_3d(&state.zx, ng, ng, nz + 1);
            let zy = slice_to_3d(&state.zy, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkq[i][j] = u[i][j][iz] * zx[i][j][iz] + v[i][j][iz] * zy[i][j][iz];
                }
            }

            _2d_to_vec(&wkq)
        };
        state.spectral.deal2d(&mut wkq);
        state.w = {
            let mut w = slice_to_3d(&state.w, ng, ng, nz + 1);
            let wkq = slice_to_2d(&wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    w[i][j][iz] += wkq[i][j];
                }
            }

            _3d_to_vec(&w)
        }
    }
}

/// Gets the nonlinear source terms for linearised PV, divergence and
/// acceleration divergence  --- all in spectral space.  These are
/// returned in sqs, sds and sgs respectively.
///
/// Note that (sds,sgs) only include the nonlinear terms for a
/// semi-implicit treatment, closely analogous to that described in
/// the appendix of Mohebalhojeh & Dritschel (2004).
///
/// The spectral fields qs, ds and gs are all spectrally truncated.
/// Note: u, v & zeta obtained by main_invert, and z obtained by psolve
/// (which calls vertical) before calling this routine are all
/// spectrally truncated.
fn source(state: &State, sqs: &mut [f64], sds: &mut [f64], sgs: &mut [f64]) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let mut dd = vec![0.0; ng * ng];
    let mut ff = vec![0.0; ng * ng];
    let mut wkp = vec![0.0; ng * ng];
    let mut wkq = vec![0.0; ng * ng];

    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];
    let mut wkc = vec![0.0; ng * ng];
    let mut wkd = vec![0.0; ng * ng];

    wkd = {
        //Calculate vertically-independent part of gs source (wkd):
        let aa_matrix = slice_to_3d(&state.aa, ng, ng, nz + 1);
        let mut wkd_matrix = slice_to_2d(&wkd, ng, ng);
        for iz in 0..=nz {
            for i in 0..ng {
                for j in 0..ng {
                    wkd_matrix[i][j] += state.spectral.weight[iz] * aa_matrix[i][j][iz];
                }
            }
        }

        //Note: aa contains div(u*rho_theta) in spectral space
        for i in 0..ng {
            for j in 0..ng {
                wkd_matrix[i][j] *= state.spectral.c2g2[i][j];
            }
        }

        _2d_to_vec(&wkd_matrix)
    };

    //Loop over layers:
    for iz in 0..=nz {
        // qs source:

        // Compute div(ql*u,ql*v) (wka in spectral space):
        wka = {
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);
            let qs_matrix = slice_to_3d(&state.qs, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[i][j] = qs_matrix[i][j][iz];
                }
            }
            _2d_to_vec(&wka_matrix)
        };
        spctop(
            ng,
            ng,
            &mut wka,
            &mut wkq,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );
        // wkq contains the linearised PV in physical space
        let mut wkp_matrix = slice_to_2d(&wkp, ng, ng);
        let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
        let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);
        let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                wkp_matrix[i][j] = wkq_matrix[i][j] * u_matrix[i][j][iz];
                wkq_matrix[i][j] *= v_matrix[i][j][iz];
            }
        }
        wkp = _2d_to_vec(&wkp_matrix);
        wkq = _2d_to_vec(&wkq_matrix);

        // Compute spectral divergence from physical fields:
        state.spectral.divs(&wkp, &wkq, &mut wka);

        // Compute Jacobian of F = (1/rho_theta)*dP'/dtheta & z (wkb, spectral):
        ff = {
            let mut ff_matrix = slice_to_2d(&ff, ng, ng);
            let ri_matrix = slice_to_3d(&state.ri, ng, ng, nz + 1);
            let dpn_matrix = slice_to_3d(&state.dpn, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    ff_matrix[i][j] = ri_matrix[i][j][iz] * dpn_matrix[i][j][iz];
                }
            }
            _2d_to_vec(&ff_matrix)
        };
        state.spectral.deal2d(&mut ff);
        wkq = {
            let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
            let z_matrix = slice_to_3d(&state.z, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[i][j] = z_matrix[i][j][iz];
                }
            }
            _2d_to_vec(&wkq_matrix)
        };
        state.spectral.jacob(&ff, &wkq, &mut wkb);

        // Sum to get qs source:
        {
            let mut sqs_matrix = slice_to_3d(&sqs, ng, ng, nz + 1);
            let wka_matrix = slice_to_2d(&wka, ng, ng);
            let wkb_matrix = slice_to_2d(&wkb, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    sqs_matrix[i][j][iz] =
                        state.spectral.filt[i][j] * (wkb_matrix[i][j] - wka_matrix[i][j]);
                }
            }
            for (i, e) in _3d_to_vec(&sqs_matrix).iter_mut().enumerate() {
                sqs[i] = *e;
            }
        }

        // Nonlinear part of ds source:

        // Compute J(u,v) (wkc in spectral space):
        {
            let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);
            let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);
            let mut d2u = vec![vec![0.0; ng]; ng];
            let mut d2v = vec![vec![0.0; ng]; ng];
            for i in 0..ng {
                for j in 0..ng {
                    d2u[i][j] = u_matrix[i][j][iz];
                    d2v[i][j] = v_matrix[i][j][iz];
                }
            }
            state
                .spectral
                .jacob(&_2d_to_vec(&d2u), &_2d_to_vec(&d2v), &mut wkc);
        }

        // Convert ds to physical space as dd:
        wka = {
            let mut wka_matrix = slice_to_2d(&wka, ng, ng);
            let ds_matrix = slice_to_3d(&state.ds, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[i][j] = ds_matrix[i][j][iz];
                }
            }
            _2d_to_vec(&wka_matrix)
        };

        spctop(
            ng,
            ng,
            &mut wka,
            &mut dd,
            &state.spectral.xfactors,
            &state.spectral.yfactors,
            &state.spectral.xtrig,
            &state.spectral.ytrig,
        );

        // Compute div(F*grad{z}-delta*{u,v}) (wkb in spectral space):
        {
            let mut wkp_matrix = slice_to_2d(&wkp, ng, ng);
            let mut wkq_matrix = slice_to_2d(&wkq, ng, ng);
            let ff_matrix = slice_to_2d(&ff, ng, ng);
            let dd_matrix = slice_to_2d(&dd, ng, ng);
            let zx_matrix = slice_to_3d(&state.zx, ng, ng, nz + 1);
            let zy_matrix = slice_to_3d(&state.zy, ng, ng, nz + 1);
            let u_matrix = slice_to_3d(&state.u, ng, ng, nz + 1);
            let v_matrix = slice_to_3d(&state.v, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wkp_matrix[i][j] = ff_matrix[i][j] * zx_matrix[i][j][iz]
                        - dd_matrix[i][j] * u_matrix[i][j][iz];
                    wkq_matrix[i][j] = ff_matrix[i][j] * zy_matrix[i][j][iz]
                        - dd_matrix[i][j] * v_matrix[i][j][iz];
                }
            }
            wkp = _2d_to_vec(&wkp_matrix);
            wkq = _2d_to_vec(&wkq_matrix);
            state.spectral.divs(&wkp, &wkq, &mut wkb);
        }

        // Add Lap(P') and complete definition of ds source:
        {
            let mut sds_matrix = slice_to_3d(&sds, ng, ng, nz + 1);
            let wkc_matrix = slice_to_2d(&wkc, ng, ng);
            let wkb_matrix = slice_to_2d(&wkb, ng, ng);
            let ps_matrix = slice_to_3d(&state.ps, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    sds_matrix[i][j][iz] = state.spectral.filt[i][j]
                        * (2.0 * wkc_matrix[i][j] + wkb_matrix[i][j]
                            - state.spectral.hlap[i][j] * ps_matrix[i][j][iz]);
                }
            }
            for (i, e) in _3d_to_vec(&sds_matrix).iter().enumerate() {
                sds[i] = *e;
            }
        }

        // Nonlinear part of gs source:
        {
            let mut sgs_matrix = slice_to_3d(&sgs, ng, ng, nz + 1);
            let sqs_matrix = slice_to_3d(&sqs, ng, ng, nz + 1);
            let wkd_matrix = slice_to_2d(&wkd, ng, ng);
            let aa_matrix = slice_to_3d(&state.aa, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    sgs_matrix[i][j][iz] =
                        COF * sqs_matrix[i][j][iz] + wkd_matrix[i][j] - FSQ * aa_matrix[i][j][iz];
                }
            }

            for (i, e) in _3d_to_vec(&sgs_matrix).iter().enumerate() {
                sgs[i] = *e;
            }
        }
    }
}
