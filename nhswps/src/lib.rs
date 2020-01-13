#![allow(clippy::cognitive_complexity)]

#[cfg(test)]
mod test;

use {
    byteorder::{ByteOrder, LittleEndian},
    ndarray::{Array1, Array2, Array3, ArrayView1, Axis, ShapeBuilder, Zip},
    shallow_water::{constants::*, spectral::Spectral, sta2dfft::D2FFT, utils::*},
    std::f64::consts::PI,
};

#[derive(Debug, Clone, Default)]
pub struct Output {
    // Plain text diagnostics
    pub ecomp: String,   //16
    pub monitor: String, //17

    // 1D vorticity and divergence spectra
    pub spectra: String, //51

    // 3D fields
    pub d3ql: Vec<u8>, //31
    pub d3d: Vec<u8>,  //32
    pub d3g: Vec<u8>,  //33
    pub d3r: Vec<u8>,  //34
    pub d3w: Vec<u8>,  //35
    pub d3pn: Vec<u8>, //36

    // Selected vertically-averaged fields
    pub d2q: Vec<u8>,    //41
    pub d2d: Vec<u8>,    //42
    pub d2g: Vec<u8>,    //43
    pub d2h: Vec<u8>,    //44
    pub d2zeta: Vec<u8>, //45
}

#[derive(Debug, Clone)]
pub struct State {
    pub spectral: Spectral,

    // Velocity field (physical
    pub u: Array3<f64>,
    pub v: Array3<f64>,
    pub w: Array3<f64>,

    // Layer heights and their x & y derivatives (physical)
    pub z: Array3<f64>,
    pub zx: Array3<f64>,
    pub zy: Array3<f64>,

    // Dimensionless layer thickness anomaly and inverse thickness (physical)
    pub r: Array3<f64>,
    pub ri: Array3<f64>,

    // A = grad{u*rho'_theta} (spectral):
    pub aa: Array3<f64>,

    // Relative vertical vorticity component (physical):
    pub zeta: Array3<f64>,

    // Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
    pub pn: Array3<f64>,
    pub dpn: Array3<f64>,

    // Non-hydrostatic pressure (p_n) in spectral space (called ps):
    pub ps: Array3<f64>,

    // Prognostic fields q_l, delta and gamma (spectral):
    pub qs: Array3<f64>,
    pub ds: Array3<f64>,
    pub gs: Array3<f64>,

    // Time:
    pub t: f64,

    // Number of time steps between field saves and related indices:
    pub ngsave: usize,
    pub itime: usize,
    pub jtime: usize,

    // Logical for use in calling inversion routine:
    pub ggen: bool,

    pub output: Output,
}

pub fn nhswps(qq: &[f64], dd: &[f64], gg: &[f64], ng: usize, nz: usize) -> Output {
    // Read linearised PV anomaly and convert to spectral space as qs

    // Parameters
    let dt = 1.0 / (ng as f64);
    let tgsave = 0.25;
    let tsim = 25.0;

    let arr3zero = Array3::<f64>::from_shape_vec(
        (ng, ng, nz + 1).strides((1, ng, ng * ng)),
        vec![0.0; ng * ng * (nz + 1)],
    )
    .unwrap();

    let mut state = State {
        spectral: Spectral::new(ng, nz),

        u: arr3zero.clone(),
        v: arr3zero.clone(),
        w: arr3zero.clone(),

        // Layer heights and their x & y derivatives (physical)
        z: arr3zero.clone(),
        zx: arr3zero.clone(),
        zy: arr3zero.clone(),

        // Dimensionless layer thickness anomaly and inverse thickness (physical)
        r: arr3zero.clone(),
        ri: arr3zero.clone(),

        // A: grad{u*rho'_theta} (spectral):
        aa: arr3zero.clone(),

        // Relative vertical vorticity component (physical):
        zeta: arr3zero.clone(),

        // Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
        pn: arr3zero.clone(),
        dpn: arr3zero.clone(),

        // Non-hydrostatic pressure (p_n) in spectral space (called ps):
        ps: arr3zero.clone(),

        // Prognostic fields q_l, delta and gamma (spectral):
        qs: arr3zero.clone(),
        ds: arr3zero.clone(),
        gs: arr3zero,

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

    state
        .spectral
        .ptospc3d(qq, state.qs.as_slice_memory_order_mut().unwrap(), 0, nz);
    {
        for i in 0..=nz {
            state.qs[[0, 0, i]] = 0.0;
        }
    };

    state
        .spectral
        .ptospc3d(dd, state.ds.as_slice_memory_order_mut().unwrap(), 0, nz);
    {
        for i in 0..=nz {
            state.ds[[0, 0, i]] = 0.0;
        }
    };

    state
        .spectral
        .ptospc3d(gg, state.gs.as_slice_memory_order_mut().unwrap(), 0, nz);
    {
        for i in 0..=nz {
            state.gs[[0, 0, i]] = 0.0;
        }
    };

    for iz in 0..=nz {
        for i in 0..ng {
            for j in 0..ng {
                state.qs[[i, j, iz]] *= state.spectral.filt[[i, j]];
                state.ds[[i, j, iz]] *= state.spectral.filt[[i, j]];
                state.gs[[i, j, iz]] *= state.spectral.filt[[i, j]];
            }
        }
    }

    state.spectral.main_invert(
        state.qs.as_slice_memory_order().unwrap(),
        state.ds.as_slice_memory_order().unwrap(),
        state.gs.as_slice_memory_order().unwrap(),
        state.r.as_slice_memory_order_mut().unwrap(),
        state.u.as_slice_memory_order_mut().unwrap(),
        state.v.as_slice_memory_order_mut().unwrap(),
        state.zeta.as_slice_memory_order_mut().unwrap(),
    );

    state.ngsave = (tgsave / dt).round() as usize;

    //Start the time loop:
    while state.t <= tsim {
        // Save data periodically:
        state.itime = (state.t / dt).round() as usize;
        state.jtime = state.itime / state.ngsave;

        if state.ngsave * state.jtime == state.itime {
            // Invert PV, divergence and acceleration divergence to obtain the
            // dimensionless layer thickness anomaly and horizontal velocity,
            // as well as the relative vertical vorticity (see spectral.f90):
            state.spectral.main_invert(
                state.qs.as_slice_memory_order().unwrap(),
                state.ds.as_slice_memory_order().unwrap(),
                state.gs.as_slice_memory_order().unwrap(),
                state.r.as_slice_memory_order_mut().unwrap(),
                state.u.as_slice_memory_order_mut().unwrap(),
                state.v.as_slice_memory_order_mut().unwrap(),
                state.zeta.as_slice_memory_order_mut().unwrap(),
            );

            //Note: qs, ds & gs are in spectral space while
            //      r, u, v and zeta are in physical space.
            //Next find the non-hydrostatic pressure (pn), layer heights (z)
            // and vertical velocity (w):
            psolve(&mut state);

            // Save field data:
            savegrid(&mut state);

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
            state.qs.as_slice_memory_order().unwrap(),
            state.ds.as_slice_memory_order().unwrap(),
            state.gs.as_slice_memory_order().unwrap(),
            state.r.as_slice_memory_order_mut().unwrap(),
            state.u.as_slice_memory_order_mut().unwrap(),
            state.v.as_slice_memory_order_mut().unwrap(),
            state.zeta.as_slice_memory_order_mut().unwrap(),
        );
        psolve(&mut state);
        savegrid(&mut state);
    }

    //finalise
    state.output
}

fn savegrid(state: &mut State) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let arr2zero =
        Array2::<f64>::from_shape_vec((ng, ng).strides((1, ng)), vec![0.0; ng * ng]).unwrap();

    let mut v3d = Array3::<f32>::from_shape_vec(
        (ng, ng, nz + 1).strides((1, ng, ng * ng)),
        vec![0.0; ng * ng * (nz + 1)],
    )
    .unwrap();
    let mut wkp = arr2zero.clone();
    let mut wkq = arr2zero.clone();
    let mut wks = arr2zero;
    let mut zspec = Array1::<f64>::zeros(ng + 1);
    let mut dspec = Array1::<f64>::zeros(ng + 1);
    let mut gspec = Array1::<f64>::zeros(ng + 1);

    let mut ekin: f64 = 0.0;
    let mut epot: f64 = 0.0;
    let mut etot: f64 = 0.0;

    // Compute kinetic energy:
    Zip::from(&mut wkp)
        .and(&state.r.index_axis(Axis(2), 0))
        .and(&state.u.index_axis(Axis(2), 0))
        .and(&state.v.index_axis(Axis(2), 0))
        .apply(|wkp, &r, &u, &v| {
            *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0));
        });

    ekin = (1.0 / 2.0) * wkp.sum();

    Zip::from(&mut wkp)
        .and(&state.r.index_axis(Axis(2), nz))
        .and(&state.u.index_axis(Axis(2), nz))
        .and(&state.v.index_axis(Axis(2), nz))
        .and(&state.w.index_axis(Axis(2), nz))
        .apply(|wkp, &r, &u, &v, &w| {
            *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0) + w.powf(2.0));
        });

    ekin += (1.0 / 2.0) * wkp.sum();

    for iz in 1..=nz - 1 {
        Zip::from(&mut wkp)
            .and(&state.r.index_axis(Axis(2), iz))
            .and(&state.u.index_axis(Axis(2), iz))
            .and(&state.v.index_axis(Axis(2), iz))
            .and(&state.w.index_axis(Axis(2), iz))
            .apply(|wkp, &r, &u, &v, &w| {
                *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0) + w.powf(2.0));
            });
        ekin += wkp.sum();
    }

    let gl = (2.0 * PI) / ng as f64;
    let dz = HBAR / nz as f64;
    let gvol = gl * gl * dz;
    ekin *= (1.0 / 2.0) * gvol * (1.0 / HBAR);

    // Compute potential energy (same as SW expression):
    for i in 0..ng {
        for j in 0..ng {
            wkp[[i, j]] = ((1.0 / HBAR) * state.z[[i, j, nz]] - 1.0).powf(2.0);
        }
    }

    epot = (1.0 / 2.0) * (gl * gl) * CSQ * wkp.sum();

    // Compute total energy:
    etot = ekin + epot;

    // Write energies to ecomp.asc:
    //write(16,'(f13.6,5(1x,f16.9))') t,zero,ekin,ekin,epot,etot
    let s = format!(
        "{:.6} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
        state.t, 0.0, ekin, ekin, epot, etot
    );
    state.output.ecomp += &s;

    // Compute vertically-averaged 1d vorticity, divergence and
    // acceleration divergence spectra:
    zspec.fill(0.0);
    dspec.fill(0.0);
    gspec.fill(0.0);

    let d2fft = D2FFT::new(
        ng,
        ng,
        2.0 * PI,
        2.0 * PI,
        &mut vec![0.0; ng],
        &mut vec![0.0; ng],
    );
    let mut tmpspec = Array1::<f64>::zeros(ng + 1);
    for iz in 0..=nz {
        for i in 0..ng {
            for j in 0..ng {
                wkp[[i, j]] = state.zeta[[i, j, iz]];
            }
        }
        d2fft.ptospc(
            wkp.as_slice_memory_order_mut().unwrap(),
            wks.as_slice_memory_order_mut().unwrap(),
        );
        state.spectral.spec1d(
            wks.as_slice_memory_order().unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        zspec = zspec + state.spectral.weight[iz] * &tmpspec;
        state.spectral.spec1d(
            state
                .ds
                .index_axis(Axis(2), iz)
                .as_slice_memory_order()
                .unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        dspec = dspec + state.spectral.weight[iz] * &tmpspec;
        state.spectral.spec1d(
            state
                .gs
                .index_axis(Axis(2), iz)
                .as_slice_memory_order()
                .unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        gspec = gspec + state.spectral.weight[iz] * &tmpspec;
    }
    // Normalise to take into account uneven sampling of wavenumbers
    // in each shell [k-1/2,k+1/2]:
    let spmf = ArrayView1::from_shape(ng + 1, &state.spectral.spmf).unwrap();
    zspec = zspec * spmf;
    dspec = dspec * spmf;
    gspec = gspec * spmf;

    let s = format!("{:.6} {}\n", state.t, state.spectral.kmaxred);
    state.output.spectra += &s;
    for k in 1..=state.spectral.kmaxred {
        let s = format!(
            "{:.8} {:.8} {:.8} {:.8}\n",
            state.spectral.alk[k - 1],
            zspec[k].log10(),
            (dspec[k] + 1.0E-32).log10(),
            (gspec[k] + 1.0E-32).log10()
        );
        state.output.spectra += &s;
    }

    // Write various 3D gridded fields to direct access files:
    // PV field:
    for iz in 0..=nz {
        wks.assign(&state.qs.index_axis(Axis(2), iz));
        d2fft.spctop(
            wks.as_slice_memory_order_mut().unwrap(),
            wkp.as_slice_memory_order_mut().unwrap(),
        );
        for i in 0..ng {
            for j in 0..ng {
                v3d[[i, j, iz]] = wkp[[i, j]] as f32;
            }
        }
    }
    append_output(
        &mut state.output.d3ql,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    );

    // Divergence field:
    for iz in 0..=nz {
        wks.assign(&state.ds.index_axis(Axis(2), iz));
        d2fft.spctop(
            wks.as_slice_memory_order_mut().unwrap(),
            wkp.as_slice_memory_order_mut().unwrap(),
        );
        for i in 0..ng {
            for j in 0..ng {
                v3d[[i, j, iz]] = wkp[[i, j]] as f32;
            }
        }
    }
    append_output(
        &mut state.output.d3d,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    );

    // Acceleration divergence field:
    for iz in 0..=nz {
        wks.assign(&state.gs.index_axis(Axis(2), iz));
        d2fft.spctop(
            wks.as_slice_memory_order_mut().unwrap(),
            wkp.as_slice_memory_order_mut().unwrap(),
        );
        for i in 0..ng {
            for j in 0..ng {
                v3d[[i, j, iz]] = wkp[[i, j]] as f32;
            }
        }
    }
    append_output(
        &mut state.output.d3g,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    );

    // Dimensionless thickness anomaly:
    let r_f32 = state
        .r
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d3g, state.t, &r_f32);

    // Vertical velocity:
    let w_f32 = state
        .w
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d3w, state.t, &w_f32);

    // Non-hydrostatic pressure:
    let pn_f32 = state
        .pn
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d3pn, state.t, &pn_f32);

    // Write various vertically-integrated 2D fields to direct access files:
    // Divergence:
    Zip::from(&mut wkp)
        .and(&state.w.index_axis(Axis(2), nz))
        .and(&state.z.index_axis(Axis(2), nz))
        .apply(|wkp, &w, &z| {
            *wkp = -w / z;
        });
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d2d, state.t, &wkp_f32);

    // Relative vorticity:
    wkp.fill(0.0);
    for iz in 0..=nz {
        Zip::from(&mut wkp)
            .and(&state.zeta.index_axis(Axis(2), iz))
            .and(&state.r.index_axis(Axis(2), iz))
            .apply(|wkp, zeta, r| {
                *wkp += state.spectral.weight[iz] * zeta * (1.0 + r);
            });
    }
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .apply(|wkp, z| {
            *wkp *= HBAR / z;
        });
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d2zeta, state.t, &wkp_f32);

    // PV anomaly:
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .apply(|wkp, z| *wkp = HBAR * (*wkp + COF) / z - COF);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d2q, state.t, &wkp_f32);

    // Acceleration divergence:
    wkp.fill(0.0);
    for iz in 0..=nz {
        wks.assign(&state.gs.index_axis(Axis(2), nz));
        d2fft.spctop(
            wks.as_slice_memory_order_mut().unwrap(),
            wkq.as_slice_memory_order_mut().unwrap(),
        );
        Zip::from(&mut wkp)
            .and(&wkq)
            .and(&state.r.index_axis(Axis(2), iz))
            .apply(|wkp, wkq, r| *wkp += state.spectral.weight[iz] * wkq * (1.0 + r));
    }
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .apply(|wkp, z| *wkp *= HBAR / z);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d2g, state.t, &wkp_f32);

    // Dimensionless height anomaly:
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .apply(|wkp, z| *wkp = (1.0 / HBAR) * z - 1.0);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut state.output.d2h, state.t, &wkp_f32);
}

fn append_output(field: &mut Vec<u8>, t: f64, data: &[f32]) {
    let mut buf = [0u8; 4];
    LittleEndian::write_f32(&mut buf, t as f32);
    field.extend_from_slice(&buf);
    for e in data {
        LittleEndian::write_f32(&mut buf, *e);
        field.extend_from_slice(&buf);
    }
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

    let mut sum = 0.0;
    for i in 0..ng {
        for j in 0..ng {
            sum += (1.0 / 2.0) * state.zeta[[i, j, 0]].powf(2.0);
            sum += (1.0 / 2.0) * state.zeta[[i, j, nz]].powf(2.0);
            for k in 1..=nz - 1 {
                sum += state.zeta[[i, j, k]].powf(2.0);
            }
        }
    }
    let zrms = (vsumi * sum).sqrt();

    let s = format!(
        "{:.5} {:.6} {:.6} {:.6} {:.6}\n",
        state.t,
        (1.0 / 2.0) * (zrms.powf(2.0)),
        zrms,
        zmax,
        umax
    );

    //println!("{}", &s);
    state.output.monitor += &s;
}

/// Advances fields from time t to t+dt using an iterative implicit
/// method of the form
pub fn advance(state: &mut State) {
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
            state.qs.as_slice_memory_order().unwrap(),
            state.ds.as_slice_memory_order().unwrap(),
            state.gs.as_slice_memory_order().unwrap(),
            state.r.as_slice_memory_order_mut().unwrap(),
            state.u.as_slice_memory_order_mut().unwrap(),
            state.v.as_slice_memory_order_mut().unwrap(),
            state.zeta.as_slice_memory_order_mut().unwrap(),
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
    let qs_slice = state.qs.as_slice_memory_order().unwrap();
    for (i, e) in qsi.iter_mut().enumerate() {
        *e = qs_slice[i];
    }
    for (i, e) in qsm.iter_mut().enumerate() {
        *e = qs_slice[i] + dt4 * sqs[i];
    }
    {
        let qsm = view3d(&qsm, ng, ng, nz + 1);
        let sqs = view3d(&sqs, ng, ng, nz + 1);
        let qsi = view3d(&qsi, ng, ng, nz + 1);
        for iz in 0..=nz {
            for i in 0..ng {
                for j in 0..ng {
                    state.qs[[i, j, iz]] = state.spectral.diss[[i, j]]
                        * (qsm[[i, j, iz]] + dt4 * sqs[[i, j, iz]])
                        - qsi[[i, j, iz]];
                }
            }
        }
    }

    // Update divergence and acceleration divergence:
    let dsi = state.ds.clone();
    let dsi_slice = dsi.as_slice_memory_order().unwrap();
    let gsi = state.gs.clone();
    let gsi_slice = gsi.as_slice_memory_order().unwrap();
    for (i, e) in nds.iter_mut().enumerate() {
        *e = sds[i] + dt4i * dsi_slice[i];
    }
    for (i, e) in ngs.iter_mut().enumerate() {
        *e = sgs[i] + dt4i * gsi_slice[i];
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
        let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
        let mut wkb_matrix = viewmut2d(&mut wkb, ng, ng);
        let sgs = view3d(&sgs, ng, ng, nz + 1);
        let sds = view3d(&sds, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                state.ds[[i, j, iz]] =
                    sgs[[i, j, iz]] + state.spectral.rdis[[i, j]] * sds[[i, j, iz]]; // 2*T_tilde_delta
                wka_matrix[[i, j]] += state.spectral.weight[iz] * state.ds[[i, j, iz]];
                wkb_matrix[[i, j]] += state.spectral.weight[iz] * sds[[i, j, iz]];
            }
        }
    }
    {
        let mut wka = viewmut2d(&mut wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wka[[i, j]] *= state.spectral.fope[[i, j]];
            }
        }
    } // fope = F operator
    {
        let mut wkb = viewmut2d(&mut wkb, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkb[[i, j]] *= state.spectral.c2g2[[i, j]];
            }
        }
    }; // c2g2 = c^2*Lap operator
    for iz in 0..=nz {
        let sds = view3d(&sds, ng, ng, nz + 1);
        let sgs = view3d(&sgs, ng, ng, nz + 1);
        let wka = view2d(&wka, ng, ng);
        let wkb = view2d(&wkb, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                // simp = (R^2 + f^2)^{-1}
                state.ds[[i, j, iz]] = state.spectral.simp[[i, j]]
                    * (state.ds[[i, j, iz]] - wka[[i, j]])
                    - dsi[[i, j, iz]];
                // 2*T_tilde_gamma
                state.gs[[i, j, iz]] = wkb[[i, j]] - FSQ * sds[[i, j, iz]]
                    + state.spectral.rdis[[i, j]] * sgs[[i, j, iz]];
            }
        }
    }

    for e in wka.iter_mut() {
        *e = 0.0;
    }
    for iz in 0..=nz {
        let mut wka_matrix = viewmut2d(&mut wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wka_matrix[[i, j]] += state.spectral.weight[iz] * state.gs[[i, j, iz]];
            }
        }
    }
    {
        let mut wka = viewmut2d(&mut wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wka[[i, j]] *= state.spectral.fope[[i, j]];
            }
        }
    } // fope = F operator in paper
    for iz in 0..=nz {
        let wka = view2d(&wka, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                // simp = (R^2 + f^2)^{-1}
                state.gs[[i, j, iz]] = state.spectral.simp[[i, j]]
                    * (state.gs[[i, j, iz]] - wka[[i, j]])
                    - gsi[[i, j, iz]];
            }
        }
    }

    // Iterate to improve estimates of F^{n+1}:
    for _ in 1..=niter {
        // Perform inversion at t^{n+1} from estimated quantities:
        state.spectral.main_invert(
            state.qs.as_slice_memory_order().unwrap(),
            state.ds.as_slice_memory_order().unwrap(),
            state.gs.as_slice_memory_order().unwrap(),
            state.r.as_slice_memory_order_mut().unwrap(),
            state.u.as_slice_memory_order_mut().unwrap(),
            state.v.as_slice_memory_order_mut().unwrap(),
            state.zeta.as_slice_memory_order_mut().unwrap(),
        );

        // Compute pressure, etc:
        psolve(state);

        // Calculate the source terms (sqs,sds,sgs) for linearised PV (qs),
        // divergence (ds) and acceleration divergence (gs):
        source(state, &mut sqs, &mut sds, &mut sgs);

        // Update PV field:
        for iz in 0..=nz {
            let qsm = view3d(&qsm, ng, ng, nz + 1);
            let sqs = view3d(&sqs, ng, ng, nz + 1);
            let qsi = view3d(&qsi, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    state.qs[[i, j, iz]] = state.spectral.diss[[i, j]]
                        * (qsm[[i, j, iz]] + dt4 * sqs[[i, j, iz]])
                        - qsi[[i, j, iz]];
                }
            }
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
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
            let mut wkb_matrix = viewmut2d(&mut wkb, ng, ng);
            let sgs = view3d(&sgs, ng, ng, nz + 1);
            let sds = view3d(&sds, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    // 2*T_tilde_delta
                    state.ds[[i, j, iz]] =
                        sgs[[i, j, iz]] + state.spectral.rdis[[i, j]] * sds[[i, j, iz]];
                    wka_matrix[[i, j]] += state.spectral.weight[iz] * state.ds[[i, j, iz]];
                    wkb_matrix[[i, j]] += state.spectral.weight[iz] * sds[[i, j, iz]];
                }
            }
        }
        {
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
            let mut wkb_matrix = viewmut2d(&mut wkb, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    // fope = F operator
                    wka_matrix[[i, j]] *= state.spectral.fope[[i, j]];
                    // c2g2 = c^2*Lap operator
                    wkb_matrix[[i, j]] *= state.spectral.c2g2[[i, j]];
                }
            }
        }
        for iz in 0..=nz {
            let sgs = view3d(&sgs, ng, ng, nz + 1);
            let sds = view3d(&sds, ng, ng, nz + 1);
            let wka = view2d(&wka, ng, ng);
            let wkb = view2d(&wkb, ng, ng);

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
        wka = vec![0.0; ng * ng];
        for iz in 0..=nz {
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[[i, j]] += state.spectral.weight[iz] * state.gs[[i, j, iz]];
                }
            }
        }
        {
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    // fope = F operator in paper
                    wka_matrix[[i, j]] *= state.spectral.fope[[i, j]];
                }
            }
        }
        for iz in 0..=nz {
            let wka = view2d(&wka, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    // simp = (R^2 + f^2)^{-1}
                    state.gs[[i, j, iz]] = state.spectral.simp[[i, j]]
                        * (state.gs[[i, j, iz]] - wka[[i, j]])
                        - gsi[[i, j, iz]];
                }
            }
        }
    }

    // Advance time:
    state.t += dt;
}

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
    for i in 0..ng {
        for j in 0..ng {
            for k in 0..=nz {
                state.ri[[i, j, k]] = 1.0 / (1.0 + state.r[[i, j, k]]);
            }
        }
    }
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
        {
            for i in 0..ng {
                for j in 0..ng {
                    state.ps[[i, j, nz]] = 0.0;
                }
            }
        }

        // Compute pressure derivatives needed in the non-constant part of the
        // source S_1 and add to S_0 (in sp0) to form total source S (sp):

        // Lower boundary at iz = 0 (use dp/dtheta = 0):
        // d^2p/dtheta^2:
        {
            let mut wkd_matrix = viewmut2d(&mut wkd, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    wkd_matrix[[i, j]] = (2.0 * state.ps[[i, j, 0]] - 5.0 * state.ps[[i, j, 1]]
                        + 4.0 * state.ps[[i, j, 2]]
                        - state.ps[[i, j, 3]])
                        * dzisq;
                }
            }
        }

        // Return to physical space:
        state.spectral.d2fft.spctop(&mut wkd, &mut d2pdt2);
        // Total source:
        {
            let mut wkp_matrix = viewmut2d(&mut wkp, ng, ng);
            let sp0_matrix = view3d(&sp0, ng, ng, nz + 1);
            let cpt2_matrix = view3d(&cpt2, ng, ng, nz + 1);
            let d2pdt2_matrix = view2d(&d2pdt2, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkp_matrix[[i, j]] =
                        sp0_matrix[[i, j, 0]] + cpt2_matrix[[i, j, 0]] * d2pdt2_matrix[[i, j]];
                }
            }
        }
        // Transform to spectral space for inversion below:
        state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
        {
            let mut sp_matrix = viewmut3d(&mut sp, ng, ng, nz + 1);
            let wka_matrix = view2d(&wka, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    sp_matrix[[i, j, 0]] = wka_matrix[[i, j]];
                }
            }
        }

        // Interior grid points:
        for iz in 1..=nz - 1 {
            wkq = d2pdt2.clone();
            {
                let mut wka = viewmut2d(&mut wka, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        wka[[i, j]] = (state.ps[[i, j, iz + 1]] - state.ps[[i, j, iz - 1]]) * hdzi;
                    }
                }
            }
            {
                let mut wkd = viewmut2d(&mut wkd, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        wkd[[i, j]] = (state.ps[[i, j, iz + 1]] - 2.0 * state.ps[[i, j, iz]]
                            + state.ps[[i, j, iz - 1]])
                            * dzisq;
                    }
                }
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

                for i in 0..ng {
                    for j in 0..ng {
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

                for i in 0..ng {
                    for j in 0..ng {
                        sp_matrix[[i, j, iz]] = wka_matrix[[i, j]];
                    }
                }
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

            for i in 0..ng {
                for j in 0..ng {
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

            for i in 0..ng {
                for j in 0..ng {
                    sp_matrix[[i, j, nz]] = wka_matrix[[i, j]];
                }
            }
        };

        // Solve tridiagonal problem for pressure in spectral space:
        {
            let mut gg_matrix = viewmut3d(&mut gg, ng, ng, nz + 1);
            let sp_matrix = view3d(&sp, ng, ng, nz + 1);
            for i in 0..ng {
                for j in 0..ng {
                    gg_matrix[[i, j, 0]] =
                        (1.0 / 3.0) * sp_matrix[[i, j, 0]] + (1.0 / 6.0) * sp_matrix[[i, j, 1]];
                }
            }

            for iz in 1..=nz - 1 {
                for i in 0..ng {
                    for j in 0..ng {
                        gg_matrix[[i, j, iz]] = (1.0 / 12.0)
                            * (sp_matrix[[i, j, iz - 1]] + sp_matrix[[i, j, iz + 1]])
                            + (5.0 / 6.0) * sp_matrix[[i, j, iz]];
                    }
                }
            }

            for i in 0..ng {
                for j in 0..ng {
                    state.ps[[i, j, 0]] = gg_matrix[[i, j, 0]] * state.spectral.htdv[[i, j, 0]];
                }
            }
            for iz in 1..=nz - 1 {
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

            for i in 0..ng {
                for j in 0..ng {
                    state.ps[[i, j, nz]] = 0.0;
                }
            }

            // Transform to physical space:
            state.spectral.spctop3d(
                state.ps.as_slice_memory_order().unwrap(),
                state.pn.as_slice_memory_order_mut().unwrap(),
                0,
                nz - 1,
            );

            {
                for i in 0..ng {
                    for j in 0..ng {
                        state.pn[[i, j, nz]] = 0.0;
                    }
                }
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
    {
        let mut gg = viewmut3d(&mut gg, ng, ng, nz + 1);
        let sp = view3d(&sp, ng, ng, nz + 1);

        for iz in 1..=nz - 1 {
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
        for iz in 2..=nz - 1 {
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
        for iz in (1..=nz - 1).rev() {
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

/// Finds the part of the pressure source which does not vary
/// in the iteration to find the pressure.
pub fn cpsource(state: &State, sp0: &mut [f64]) {
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
    {
        let mut wkp_matrix = viewmut2d(&mut wkp, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkp_matrix[[i, j]] = state.z[[i, j, nz]];
            }
        }
    }

    // Fourier transform to spectral space:
    state.spectral.d2fft.ptospc(&mut wkp, &mut wka);
    // Apply -g*Lap_h operator:
    {
        let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                wka_matrix[[i, j]] *= state.spectral.glap[[i, j]];
            }
        }
    }
    // Return to physical space:
    state.spectral.d2fft.spctop(&mut wka, &mut hsrc);
    // hsrc contains -g*Lap{h} in physical space.

    // Calculate u_theta, v_theta & w_theta:
    let mut ut_matrix = viewmut3d(&mut ut, ng, ng, nz + 1);
    let mut vt_matrix = viewmut3d(&mut vt, ng, ng, nz + 1);
    let mut wt_matrix = viewmut3d(&mut wt, ng, ng, nz + 1);

    // Lower boundary (use higher order formula):
    for i in 0..ng {
        for j in 0..ng {
            ut_matrix[[i, j, 0]] =
                hdzi * (4.0 * state.u[[i, j, 1]] - 3.0 * state.u[[i, j, 0]] - state.u[[i, j, 2]]);
            vt_matrix[[i, j, 0]] =
                hdzi * (4.0 * state.v[[i, j, 1]] - 3.0 * state.v[[i, j, 0]] - state.v[[i, j, 2]]);
            wt_matrix[[i, j, 0]] =
                hdzi * (4.0 * state.w[[i, j, 1]] - 3.0 * state.w[[i, j, 0]] - state.w[[i, j, 2]]);
        }
    }

    // Interior (centred differencing):
    for iz in 1..nz {
        for i in 0..ng {
            for j in 0..ng {
                ut_matrix[[i, j, iz]] = hdzi * (state.u[[i, j, iz + 1]] - state.u[[i, j, iz - 1]]);
                vt_matrix[[i, j, iz]] = hdzi * (state.v[[i, j, iz + 1]] - state.v[[i, j, iz - 1]]);
                wt_matrix[[i, j, iz]] = hdzi * (state.w[[i, j, iz + 1]] - state.w[[i, j, iz - 1]]);
            }
        }
    }

    // Upper boundary (use higher order formula):
    for i in 0..ng {
        for j in 0..ng {
            ut_matrix[[i, j, nz]] = hdzi
                * (3.0 * state.u[[i, j, nz]] + state.u[[i, j, nz - 2]]
                    - 4.0 * state.u[[i, j, nz - 1]]);
            vt_matrix[[i, j, nz]] = hdzi
                * (3.0 * state.v[[i, j, nz]] + state.v[[i, j, nz - 2]]
                    - 4.0 * state.v[[i, j, nz - 1]]);
            wt_matrix[[i, j, nz]] = hdzi
                * (3.0 * state.w[[i, j, nz]] + state.w[[i, j, nz - 2]]
                    - 4.0 * state.w[[i, j, nz - 1]]);
        }
    }

    // Loop over layers and build up source, sp0:

    // iz = 0 is much simpler as z = w = 0 there:
    // Calculate u_x, u_y, v_x & v_y:
    {
        let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[[i, j]] = state.u[[i, j, 0]];
            }
        }
    };
    state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
    state
        .spectral
        .d2fft
        .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
    state.spectral.d2fft.spctop(&mut wkb, &mut ux);
    state
        .spectral
        .d2fft
        .yderiv(&state.spectral.hrky, &wka, &mut wkb);
    state.spectral.d2fft.spctop(&mut wkb, &mut uy);
    {
        let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[[i, j]] = state.v[[i, j, 0]];
            }
        }
    };
    state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
    state
        .spectral
        .d2fft
        .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
    state.spectral.d2fft.spctop(&mut wkb, &mut vx);
    state
        .spectral
        .d2fft
        .yderiv(&state.spectral.hrky, &wka, &mut wkb);
    state.spectral.d2fft.spctop(&mut wkb, &mut vy);
    {
        let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);
        let wt_matrix = view3d(&wt, ng, ng, nz + 1);

        for i in 0..ng {
            for j in 0..ng {
                wkq_matrix[[i, j]] = state.ri[[i, j, 0]] * wt_matrix[[i, j, 0]];
            }
        }
    };
    state.spectral.deal2d(&mut wkq);
    {
        let mut zeta2d = vec![0.0; ng * ng];
        {
            let mut zeta2d = viewmut2d(&mut zeta2d, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    zeta2d[[i, j]] = state.zeta[[i, j, 0]];
                }
            }
        }
        let mut d2sp0 = vec![0.0; ng * ng];
        {
            let sp0 = view3d(&sp0, ng, ng, nz + 1);
            let mut d2sp0 = viewmut2d(&mut d2sp0, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    d2sp0[[i, j]] = sp0[[i, j, 0]];
                }
            }
        };
        for (i, e) in d2sp0.iter_mut().enumerate() {
            *e = hsrc[i]
                + COF * zeta2d[i]
                + 2.0 * (ux[i] * vy[i] - uy[i] * vx[i] + wkq[i] * (ux[i] + vy[i]));
        }
        let mut d3sp0 = viewmut3d(sp0, ng, ng, nz + 1);
        let d2sp0 = view2d(&d2sp0, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                d3sp0[[i, j, 0]] = d2sp0[[i, j]];
            }
        }
    }

    for iz in 1..=nz {
        // Calculate u_x, u_y, v_x, v_y, w_x, w_y:
        {
            let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[[i, j]] = state.u[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut ux);
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut uy);
        {
            let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[[i, j]] = state.v[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut vx);
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut vy);
        {
            let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[[i, j]] = state.w[[i, j, iz]];
                }
            }
        }
        state.spectral.d2fft.ptospc(&mut wkq, &mut wka);
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut wx);
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, &wka, &mut wkb);
        state.spectral.d2fft.spctop(&mut wkb, &mut wy);
        // Calculate pressure source:
        let mut vt2d = vec![0.0; ng * ng];
        {
            let mut vt2d = viewmut2d(&mut vt2d, ng, ng);
            let vt_matrix = view3d(&vt, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    vt2d[[i, j]] = vt_matrix[[i, j, iz]];
                }
            }
        };
        let mut ut2d = vec![0.0; ng * ng];
        {
            let mut ut2d = viewmut2d(&mut ut2d, ng, ng);
            let ut_matrix = view3d(&ut, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    ut2d[[i, j]] = ut_matrix[[i, j, iz]];
                }
            }
        };
        let mut wt2d = vec![0.0; ng * ng];
        {
            let mut wt2d = viewmut2d(&mut wt2d, ng, ng);
            let wt_matrix = view3d(&wt, ng, ng, nz + 1);

            for i in 0..ng {
                for j in 0..ng {
                    wt2d[[i, j]] = wt_matrix[[i, j, iz]];
                }
            }
        }
        let mut zx2d = vec![0.0; ng * ng];
        {
            let mut zx2d = viewmut2d(&mut zx2d, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    zx2d[[i, j]] = state.zx[[i, j, iz]];
                }
            }
        }
        let mut zy2d = vec![0.0; ng * ng];
        {
            let mut zy2d = viewmut2d(&mut zy2d, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    zy2d[[i, j]] = state.zy[[i, j, iz]];
                }
            }
        }
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
            let mut zeta2d = vec![0.0; ng * ng];
            {
                let mut zeta2d = viewmut2d(&mut zeta2d, ng, ng);
                for i in 0..ng {
                    for j in 0..ng {
                        zeta2d[[i, j]] = state.zeta[[i, j, iz]];
                    }
                }
            };

            let mut ri2d = vec![0.0; ng * ng];
            {
                let mut ri2d = viewmut2d(&mut ri2d, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        ri2d[[i, j]] = state.ri[[i, j, iz]];
                    }
                }
            }

            let mut d2sp0 = vec![0.0; ng * ng];
            {
                let sp0 = view3d(&sp0, ng, ng, nz + 1);
                let mut d2sp0 = viewmut2d(&mut d2sp0, ng, ng);

                for i in 0..ng {
                    for j in 0..ng {
                        d2sp0[[i, j]] = sp0[[i, j, 0]];
                    }
                }
            }

            for (i, e) in d2sp0.iter_mut().enumerate() {
                *e = hsrc[i]
                    + COF * (zeta2d[i] - ri2d[i] * wkp[i])
                    + 2.0 * (ux[i] * vy[i] - uy[i] * vx[i] + ri2d[i] * wkq[i]);
            }
            let mut d3sp0 = viewmut3d(sp0, ng, ng, nz + 1);
            let d2sp0 = view2d(&d2sp0, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    d3sp0[[i, j, iz]] = d2sp0[[i, j]];
                }
            }
        }
    }
}

/// Calculates the fixed coefficients used in the pressure iteration.
pub fn coeffs(
    state: &State,
    sigx: &mut [f64],
    sigy: &mut [f64],
    cpt1: &mut [f64],
    cpt2: &mut [f64],
) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let qdzi = (1.0 / 4.0) * (1.0 / (HBAR / nz as f64));
    let mut wkp = vec![0.0; ng * ng];
    let mut wka = vec![0.0; ng * ng];

    // Compute sigx and sigy and de-alias:
    let ri_slice = state.ri.as_slice_memory_order().unwrap();
    let zx_slice = state.zx.as_slice_memory_order().unwrap();
    for (i, e) in sigx.iter_mut().enumerate() {
        *e = ri_slice[i] * zx_slice[i];
    }
    let zy_slice = state.zy.as_slice_memory_order().unwrap();
    for (i, e) in sigy.iter_mut().enumerate() {
        *e = ri_slice[i] * zy_slice[i];
    }
    state.spectral.deal3d(sigx);
    state.spectral.deal3d(sigy);

    // Compute cpt2 and de-alias:
    for (i, e) in cpt2.iter_mut().enumerate() {
        *e = 1.0 - ri_slice[i].powf(2.0) - sigx[i].powf(2.0) - sigy[i].powf(2.0);
    }
    state.spectral.deal3d(cpt2);

    // Calculate 0.5*d(cpt2)/dtheta + div(sigx,sigy) and store in cpt1:

    // Lower boundary (use higher order formula):
    {
        let mut cpt1_matrix = viewmut3d(cpt1, ng, ng, nz + 1);
        let cpt2_matrix = view3d(&cpt2, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                cpt1_matrix[[i, j, 0]] = qdzi
                    * (4.0 * cpt2_matrix[[i, j, 1]]
                        - 3.0 * cpt2_matrix[[i, j, 0]]
                        - cpt2_matrix[[i, j, 2]]);
            }
        }
    }
    // qdzi=1/(4*dz) is used since 0.5*d/dtheta is being computed.

    // Interior (centred differencing):
    for iz in 1..nz {
        let d3_sigx = view3d(sigx, ng, ng, nz + 1);
        let d3_sigy = view3d(sigy, ng, ng, nz + 1);
        let mut d2_sigx = vec![0.0; ng * ng];
        let mut d2_sigy = vec![0.0; ng * ng];
        {
            let mut d2_sigx = viewmut2d(&mut d2_sigx, ng, ng);
            let mut d2_sigy = viewmut2d(&mut d2_sigy, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    d2_sigx[[i, j]] = d3_sigx[[i, j, iz]];
                    d2_sigy[[i, j]] = d3_sigy[[i, j, iz]];
                }
            }
        }
        state.spectral.divs(&d2_sigx, &d2_sigy, &mut wka);
        state.spectral.d2fft.spctop(&mut wka, &mut wkp);
        {
            let mut cpt1_matrix = viewmut3d(cpt1, ng, ng, nz + 1);
            let cpt2_matrix = view3d(&cpt2, ng, ng, nz + 1);
            let wkp_matrix = view2d(&wkp, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    cpt1_matrix[[i, j, iz]] = qdzi
                        * (cpt2_matrix[[i, j, iz + 1]] - cpt2_matrix[[i, j, iz - 1]])
                        + wkp_matrix[[i, j]];
                }
            }
        }
    }

    // Upper boundary (use higher order formula):
    let mut d2_sigx = vec![0.0; ng * ng];
    let mut d2_sigy = vec![0.0; ng * ng];
    {
        let mut d2_sigx = viewmut2d(&mut d2_sigx, ng, ng);
        let mut d2_sigy = viewmut2d(&mut d2_sigy, ng, ng);
        let d3_sigx = view3d(sigx, ng, ng, nz + 1);
        let d3_sigy = view3d(sigy, ng, ng, nz + 1);
        for i in 0..ng {
            for j in 0..ng {
                d2_sigx[[i, j]] = d3_sigx[[i, j, nz]];
                d2_sigy[[i, j]] = d3_sigy[[i, j, nz]];
            }
        }
    }
    state.spectral.divs(&d2_sigx, &d2_sigy, &mut wka);
    state.spectral.d2fft.spctop(&mut wka, &mut wkp);
    {
        let mut cpt1_matrix = viewmut3d(cpt1, ng, ng, nz + 1);
        let cpt2_matrix = view3d(&cpt2, ng, ng, nz + 1);
        let wkp_matrix = view2d(&wkp, ng, ng);
        for i in 0..ng {
            for j in 0..ng {
                cpt1_matrix[[i, j, nz]] = qdzi
                    * (3.0 * cpt2_matrix[[i, j, nz]] + cpt2_matrix[[i, j, nz - 2]]
                        - 4.0 * cpt2_matrix[[i, j, nz - 1]])
                    + wkp_matrix[[i, j]];
            }
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
pub fn source(state: &State, sqs: &mut [f64], sds: &mut [f64], sgs: &mut [f64]) {
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

    {
        //Calculate vertically-independent part of gs source (wkd):
        let mut wkd_matrix = viewmut2d(&mut wkd, ng, ng);
        for iz in 0..=nz {
            for i in 0..ng {
                for j in 0..ng {
                    wkd_matrix[[i, j]] += state.spectral.weight[iz] * state.aa[[i, j, iz]];
                }
            }
        }

        //Note: aa contains div(u*rho_theta) in spectral space
        for i in 0..ng {
            for j in 0..ng {
                wkd_matrix[[i, j]] *= state.spectral.c2g2[[i, j]];
            }
        }
    };

    //Loop over layers:
    for iz in 0..=nz {
        // qs source:

        // Compute div(ql*u,ql*v) (wka in spectral space):
        {
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[[i, j]] = state.qs[[i, j, iz]];
                }
            }
        };
        state.spectral.d2fft.spctop(&mut wka, &mut wkq);
        // wkq contains the linearised PV in physical space
        let mut wkp_matrix = viewmut2d(&mut wkp, ng, ng);
        let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);

        for i in 0..ng {
            for j in 0..ng {
                wkp_matrix[[i, j]] = wkq_matrix[[i, j]] * state.u[[i, j, iz]];
                wkq_matrix[[i, j]] *= state.v[[i, j, iz]];
            }
        }
        // Compute spectral divergence from physical fields:
        state.spectral.divs(&wkp, &wkq, &mut wka);

        // Compute Jacobian of F = (1/rho_theta)*dP'/dtheta & z (wkb, spectral):
        {
            let mut ff_matrix = viewmut2d(&mut ff, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    ff_matrix[[i, j]] = state.ri[[i, j, iz]] * state.dpn[[i, j, iz]];
                }
            }
        }
        state.spectral.deal2d(&mut ff);
        {
            let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    wkq_matrix[[i, j]] = state.z[[i, j, iz]];
                }
            }
        }
        state.spectral.jacob(&ff, &wkq, &mut wkb);

        // Sum to get qs source:
        {
            let mut sqs_matrix = viewmut3d(sqs, ng, ng, nz + 1);
            let wka_matrix = view2d(&wka, ng, ng);
            let wkb_matrix = view2d(&wkb, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    sqs_matrix[[i, j, iz]] =
                        state.spectral.filt[[i, j]] * (wkb_matrix[[i, j]] - wka_matrix[[i, j]]);
                }
            }
        }

        // Nonlinear part of ds source:

        // Compute J(u,v) (wkc in spectral space):
        let mut d2u = vec![0.0; ng * ng];
        let mut d2v = vec![0.0; ng * ng];
        {
            let mut d2u = viewmut2d(&mut d2u, ng, ng);
            let mut d2v = viewmut2d(&mut d2v, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    d2u[[i, j]] = state.u[[i, j, iz]];
                    d2v[[i, j]] = state.v[[i, j, iz]];
                }
            }
        }
        state.spectral.jacob(&d2u, &d2v, &mut wkc);

        // Convert ds to physical space as dd:
        {
            let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    wka_matrix[[i, j]] = state.ds[[i, j, iz]];
                }
            }
        }

        state.spectral.d2fft.spctop(&mut wka, &mut dd);

        // Compute div(F*grad{z}-delta*{u,v}) (wkb in spectral space):
        {
            let mut wkp_matrix = viewmut2d(&mut wkp, ng, ng);
            let mut wkq_matrix = viewmut2d(&mut wkq, ng, ng);
            let ff_matrix = view2d(&ff, ng, ng);
            let dd_matrix = view2d(&dd, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    wkp_matrix[[i, j]] = ff_matrix[[i, j]] * state.zx[[i, j, iz]]
                        - dd_matrix[[i, j]] * state.u[[i, j, iz]];
                    wkq_matrix[[i, j]] = ff_matrix[[i, j]] * state.zy[[i, j, iz]]
                        - dd_matrix[[i, j]] * state.v[[i, j, iz]];
                }
            }
            state.spectral.divs(&wkp, &wkq, &mut wkb);
        }

        // Add Lap(P') and complete definition of ds source:
        {
            let mut sds_matrix = viewmut3d(sds, ng, ng, nz + 1);
            let wkc_matrix = view2d(&wkc, ng, ng);
            let wkb_matrix = view2d(&wkb, ng, ng);
            for i in 0..ng {
                for j in 0..ng {
                    sds_matrix[[i, j, iz]] = state.spectral.filt[[i, j]]
                        * (2.0 * wkc_matrix[[i, j]] + wkb_matrix[[i, j]]
                            - state.spectral.hlap[[i, j]] * state.ps[[i, j, iz]]);
                }
            }
        }

        // Nonlinear part of gs source:
        {
            let mut sgs_matrix = viewmut3d(sgs, ng, ng, nz + 1);
            let sqs_matrix = view3d(&sqs, ng, ng, nz + 1);
            let wkd_matrix = view2d(&wkd, ng, ng);

            for i in 0..ng {
                for j in 0..ng {
                    sgs_matrix[[i, j, iz]] = COF * sqs_matrix[[i, j, iz]] + wkd_matrix[[i, j]]
                        - FSQ * state.aa[[i, j, iz]];
                }
            }
        }
    }
}
