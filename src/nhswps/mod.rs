pub mod advance;
pub mod coeffs;
pub mod cpsource;
pub mod psolve;
pub mod source;
pub mod vertical;

use {
    crate::{
        constants::*,
        parameters::Parameters,
        spectral::Spectral,
        sta2dfft::D2FFT,
        utils::{arr2zero, arr3zero, view3d},
    },
    advance::advance,
    anyhow::Result,
    byteorder::{ByteOrder, LittleEndian},
    log::info,
    ndarray::{Array1, Array3, ArrayView1, Axis, ShapeBuilder, Zip},
    psolve::psolve,
    serde::{Deserialize, Serialize},
    source::source,
    std::{
        f64::consts::PI,
        fs::{create_dir_all, File},
        io::{Read, Write},
        path::Path,
    },
};

#[derive(Debug)]
pub struct Output {
    // Plain text diagnostics
    pub ecomp: File,   //16
    pub monitor: File, //17

    // 1D vorticity and divergence spectra
    pub spectra: File, //51

    // 3D fields
    pub d3ql: File, //31
    pub d3d: File,  //32
    pub d3g: File,  //33
    pub d3r: File,  //34
    pub d3w: File,  //35
    pub d3pn: File, //36

    // Selected vertically-averaged fields
    pub d2q: File,    //41
    pub d2d: File,    //42
    pub d2g: File,    //43
    pub d2h: File,    //44
    pub d2zeta: File, //45
}

impl Output {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let d2 = path.as_ref().join("2d/");
        let d3 = path.as_ref().join("3d/");

        create_dir_all(&d2)?;
        create_dir_all(&d3)?;

        Ok(Self {
            ecomp: File::create(path.as_ref().join("ecomp.asc"))?,
            monitor: File::create(path.as_ref().join("monitor.asc"))?,

            spectra: File::create(path.as_ref().join("spectra.asc"))?,

            d3ql: File::create(d3.join("ql.r4"))?,
            d3d: File::create(d3.join("d.r4"))?,
            d3g: File::create(d3.join("g.r4"))?,
            d3r: File::create(d3.join("r.r4"))?,
            d3w: File::create(d3.join("w.r4"))?,
            d3pn: File::create(d3.join("pn.r4"))?,

            d2q: File::create(d2.join("q.r4"))?,
            d2d: File::create(d2.join("d.r4"))?,
            d2g: File::create(d2.join("g.r4"))?,
            d2h: File::create(d2.join("h.r4"))?,
            d2zeta: File::create(d2.join("zeta.r4"))?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

impl State {
    /// Reinitializes all arrays so that each has contiguous memory
    pub fn defragment(&mut self) {
        let ds = self.ds.clone();
        self.ds = arr3zero(self.spectral.ng, self.spectral.nz);
        self.ds.assign(&ds);

        let gs = self.gs.clone();
        self.gs = arr3zero(self.spectral.ng, self.spectral.nz);
        self.gs.assign(&gs);

        //TODO: All other arrays
    }
}

pub fn nhswps(parameters: &Parameters) -> Result<()> {
    // Parameters
    let ng = parameters.numerical.grid_resolution;
    let nz = parameters.numerical.vertical_layers;
    let dt = parameters.numerical.time_step;
    let tgsave = parameters.numerical.save_interval;
    let tsim = parameters.numerical.duration;
    let out_dir = parameters.environment.output_directory.clone();

    // Read linearised PV anomaly and convert to spectral space as qs
    let qq = {
        let mut f = File::open(out_dir.join("qq_init.r8"))?;
        let mut qq = Vec::new();
        f.read_to_end(&mut qq)?;
        qq.chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>()
    };

    let dd = {
        let mut f = File::open(out_dir.join("dd_init.r8"))?;
        let mut dd = Vec::new();
        f.read_to_end(&mut dd)?;
        dd.chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>()
    };

    let gg = {
        let mut f = File::open(out_dir.join("gg_init.r8"))?;
        let mut gg = Vec::new();
        f.read_to_end(&mut gg)?;
        gg.chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>()
    };

    let mut state = State {
        spectral: Spectral::new(ng, nz),

        u: arr3zero(ng, nz),
        v: arr3zero(ng, nz),
        w: arr3zero(ng, nz),

        // Layer heights and their x & y derivatives (physical)
        z: arr3zero(ng, nz),
        zx: arr3zero(ng, nz),
        zy: arr3zero(ng, nz),

        // Dimensionless layer thickness anomaly and inverse thickness (physical)
        r: arr3zero(ng, nz),
        ri: arr3zero(ng, nz),

        // A: grad{u*rho'_theta} (spectral):
        aa: arr3zero(ng, nz),

        // Relative vertical vorticity component (physical):
        zeta: arr3zero(ng, nz),

        // Non-hydrostatic pressure (p_n) and its first derivative wrt theta:
        pn: arr3zero(ng, nz),
        dpn: arr3zero(ng, nz),

        // Non-hydrostatic pressure (p_n) in spectral space (called ps):
        ps: arr3zero(ng, nz),

        // Prognostic fields q_l, delta and gamma (spectral):
        qs: arr3zero(ng, nz),
        ds: arr3zero(ng, nz),
        gs: arr3zero(ng, nz),

        // Time:
        t: 0.0,

        // Number of time steps between field saves and related indices:
        ngsave: 0,
        itime: 0,
        jtime: 0,

        // Logical for use in calling inversion routine:
        ggen: false,
    };

    let mut output = Output::from_path(out_dir)?;

    state
        .spectral
        .ptospc3d(view3d(&qq, ng, ng, nz + 1), state.qs.view_mut(), 0, nz);
    {
        for i in 0..=nz {
            state.qs[[0, 0, i]] = 0.0;
        }
    };

    state
        .spectral
        .ptospc3d(view3d(&dd, ng, ng, nz + 1), state.ds.view_mut(), 0, nz);
    {
        for i in 0..=nz {
            state.ds[[0, 0, i]] = 0.0;
        }
    };

    state
        .spectral
        .ptospc3d(view3d(&gg, ng, ng, nz + 1), state.gs.view_mut(), 0, nz);
    {
        for i in 0..=nz {
            state.gs[[0, 0, i]] = 0.0;
        }
    };

    for iz in 0..=nz {
        Zip::from(state.qs.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.filt)
            .for_each(|qs, filt| *qs *= filt);
        Zip::from(state.ds.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.filt)
            .for_each(|ds, filt| *ds *= filt);
        Zip::from(state.gs.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.filt)
            .for_each(|gs, filt| *gs *= filt);
    }

    state.spectral.main_invert(
        state.qs.view(),
        state.ds.view(),
        state.gs.view(),
        state.r.view_mut(),
        state.u.view_mut(),
        state.v.view_mut(),
        state.zeta.view_mut(),
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
                state.qs.view(),
                state.ds.view(),
                state.gs.view(),
                state.r.view_mut(),
                state.u.view_mut(),
                state.v.view_mut(),
                state.zeta.view_mut(),
            );

            //Note: qs, ds & gs are in spectral space while
            //      r, u, v and zeta are in physical space.
            //Next find the non-hydrostatic pressure (pn), layer heights (z)
            // and vertical velocity (w):
            psolve(&mut state);

            // Save field data:
            savegrid(&mut state, &mut output)?;

            state.ggen = false;
        } else {
            state.ggen = true;
        }
        // ggen is used to indicate if calling inversion is needed in advance below

        // Advect flow from time t to t + dt:
        advance(&mut state, &mut output)?;
    }

    state.itime = (state.t / dt) as usize;
    state.jtime = state.itime / state.ngsave;
    if state.ngsave * state.jtime == state.itime {
        state.spectral.main_invert(
            state.qs.view(),
            state.ds.view(),
            state.gs.view(),
            state.r.view_mut(),
            state.u.view_mut(),
            state.v.view_mut(),
            state.zeta.view_mut(),
        );
        psolve(&mut state);
        savegrid(&mut state, &mut output)?;
    }

    Ok(())
}

pub fn savegrid(state: &mut State, output: &mut Output) -> Result<()> {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let arr2zero = arr2zero(ng);

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

    // Compute kinetic energy:
    Zip::from(&mut wkp)
        .and(&state.r.index_axis(Axis(2), 0))
        .and(&state.u.index_axis(Axis(2), 0))
        .and(&state.v.index_axis(Axis(2), 0))
        .par_for_each(|wkp, &r, &u, &v| {
            *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0));
        });

    let mut ekin = (1.0 / 2.0) * wkp.sum();

    Zip::from(&mut wkp)
        .and(&state.r.index_axis(Axis(2), nz))
        .and(&state.u.index_axis(Axis(2), nz))
        .and(&state.v.index_axis(Axis(2), nz))
        .and(&state.w.index_axis(Axis(2), nz))
        .par_for_each(|wkp, &r, &u, &v, &w| {
            *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0) + w.powf(2.0));
        });

    ekin += (1.0 / 2.0) * wkp.sum();

    for iz in 1..nz {
        Zip::from(&mut wkp)
            .and(&state.r.index_axis(Axis(2), iz))
            .and(&state.u.index_axis(Axis(2), iz))
            .and(&state.v.index_axis(Axis(2), iz))
            .and(&state.w.index_axis(Axis(2), iz))
            .par_for_each(|wkp, &r, &u, &v, &w| {
                *wkp = (1.0 + r) * (u.powf(2.0) + v.powf(2.0) + w.powf(2.0));
            });
        ekin += wkp.sum();
    }

    let gl = (2.0 * PI) / ng as f64;
    let dz = HBAR / nz as f64;
    let gvol = gl * gl * dz;
    ekin *= (1.0 / 2.0) * gvol * (1.0 / HBAR);

    // Compute potential energy (same as SW expression):
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .par_for_each(|wkp, &z| *wkp = ((1.0 / HBAR) * z - 1.0).powf(2.0));

    let epot = (1.0 / 2.0) * (gl * gl) * CSQ * wkp.sum();

    // Compute total energy:
    let etot = ekin + epot;

    // Write energies to ecomp.asc:
    //write(16,'(f13.6,5(1x,f16.9))') t,zero,ekin,ekin,epot,etot
    write!(
        output.ecomp,
        "{:.6} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
        state.t, 0.0, ekin, ekin, epot, etot
    )?;

    info!("t = {}, E_tot = {}", state.t, etot);

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
        wkp.assign(&state.zeta.index_axis(Axis(2), iz));

        d2fft.ptospc(wkp.view_mut(), wks.view_mut());
        state.spectral.spec1d(
            wks.as_slice_memory_order().unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        zspec += &(state.spectral.weight[iz] * &tmpspec);
        state.spectral.spec1d(
            state
                .ds
                .index_axis(Axis(2), iz)
                .as_slice_memory_order()
                .unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        dspec += &(state.spectral.weight[iz] * &tmpspec);
        state.spectral.spec1d(
            state
                .gs
                .index_axis(Axis(2), iz)
                .as_slice_memory_order()
                .unwrap(),
            tmpspec.as_slice_memory_order_mut().unwrap(),
        );
        gspec += &(state.spectral.weight[iz] * &tmpspec);
    }
    // Normalise to take into account uneven sampling of wavenumbers
    // in each shell [k-1/2,k+1/2]:
    let spmf = ArrayView1::from_shape(ng + 1, &state.spectral.spmf).unwrap();
    zspec *= &spmf;
    dspec *= &spmf;
    gspec *= &spmf;

    write!(
        output.spectra,
        "{:.6} {}\n",
        state.t, state.spectral.kmaxred
    )?;

    for k in 1..=state.spectral.kmaxred {
        write!(
            output.spectra,
            "{:.8} {:.8} {:.8} {:.8}\n",
            state.spectral.alk[k - 1],
            zspec[k].log10(),
            (dspec[k] + 1.0E-32).log10(),
            (gspec[k] + 1.0E-32).log10()
        )?;
    }

    // Write various 3D gridded fields to direct access files:
    // PV field:
    for iz in 0..=nz {
        wks.assign(&state.qs.index_axis(Axis(2), iz));

        d2fft.spctop(wks.view_mut(), wkp.view_mut());

        v3d.index_axis_mut(Axis(2), iz)
            .assign(&wkp.mapv(|f| f as f32));
    }
    append_output(
        &mut output.d3ql,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    )?;

    // Divergence field:
    for iz in 0..=nz {
        wks.assign(&state.ds.index_axis(Axis(2), iz));

        d2fft.spctop(wks.view_mut(), wkp.view_mut());

        v3d.index_axis_mut(Axis(2), iz)
            .assign(&wkp.mapv(|f| f as f32));
    }
    append_output(
        &mut output.d3d,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    )?;

    // Acceleration divergence field:
    for iz in 0..=nz {
        wks.assign(&state.gs.index_axis(Axis(2), iz));

        d2fft.spctop(wks.view_mut(), wkp.view_mut());

        v3d.index_axis_mut(Axis(2), iz)
            .assign(&wkp.mapv(|f| f as f32));
    }
    append_output(
        &mut output.d3g,
        state.t,
        v3d.as_slice_memory_order().unwrap(),
    )?;

    // Dimensionless thickness anomaly:
    let r_f32 = state
        .r
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d3r, state.t, &r_f32)?;

    // Vertical velocity:
    let w_f32 = state
        .w
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d3w, state.t, &w_f32)?;

    // Non-hydrostatic pressure:
    let pn_f32 = state
        .pn
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d3pn, state.t, &pn_f32)?;

    // Write various vertically-integrated 2D fields to direct access files:
    // Divergence:
    Zip::from(&mut wkp)
        .and(&state.w.index_axis(Axis(2), nz))
        .and(&state.z.index_axis(Axis(2), nz))
        .for_each(|wkp, &w, &z| {
            *wkp = -w / z;
        });
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d2d, state.t, &wkp_f32)?;

    // Relative vorticity:
    wkp.fill(0.0);
    for iz in 0..=nz {
        Zip::from(&mut wkp)
            .and(&state.zeta.index_axis(Axis(2), iz))
            .and(&state.r.index_axis(Axis(2), iz))
            .for_each(|wkp, zeta, r| {
                *wkp += state.spectral.weight[iz] * zeta * (1.0 + r);
            });
    }
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .for_each(|wkp, z| {
            *wkp *= HBAR / z;
        });
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d2zeta, state.t, &wkp_f32)?;

    // PV anomaly:
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .for_each(|wkp, z| *wkp = HBAR * (*wkp + COF) / z - COF);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d2q, state.t, &wkp_f32)?;

    // Acceleration divergence:
    wkp.fill(0.0);
    for iz in 0..=nz {
        wks.assign(&state.gs.index_axis(Axis(2), iz));
        d2fft.spctop(wks.view_mut(), wkq.view_mut());
        Zip::from(&mut wkp)
            .and(&wkq)
            .and(&state.r.index_axis(Axis(2), iz))
            .for_each(|wkp, wkq, r| *wkp += state.spectral.weight[iz] * wkq * (1.0 + r));
    }
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .for_each(|wkp, z| *wkp *= HBAR / z);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d2g, state.t, &wkp_f32)?;

    // Dimensionless height anomaly:
    Zip::from(&mut wkp)
        .and(&state.z.index_axis(Axis(2), nz))
        .for_each(|wkp, z| *wkp = (1.0 / HBAR) * z - 1.0);
    let wkp_f32 = wkp
        .as_slice_memory_order()
        .unwrap()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>();
    append_output(&mut output.d2h, state.t, &wkp_f32)?;

    Ok(())
}

fn append_output(field: &mut File, t: f64, data: &[f32]) -> Result<()> {
    let mut buf = [0u8; 4];
    LittleEndian::write_f32(&mut buf, t as f32);
    field.write_all(&buf)?;
    for e in data {
        LittleEndian::write_f32(&mut buf, *e);
        field.write_all(&buf)?;
    }
    Ok(())
}

/// Computes various quantities every time step to monitor the flow evolution.
fn diagnose(state: &mut State, output: &mut Output) -> Result<()> {
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
            for k in 1..nz {
                sum += state.zeta[[i, j, k]].powf(2.0);
            }
        }
    }
    let zrms = (vsumi * sum).sqrt();

    write!(
        output.monitor,
        "{:.5} {:.6} {:.6} {:.6} {:.6}\n",
        state.t,
        (1.0 / 2.0) * (zrms.powf(2.0)),
        zrms,
        zmax,
        umax
    )?;

    Ok(())
}
