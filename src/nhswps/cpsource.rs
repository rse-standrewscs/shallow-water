use {
    crate::{
        constants::*,
        nhswps::State,
        utils::{arr2zero, arr3zero},
    },
    ndarray::{ArrayViewMut3, Axis, Zip},
    rayon::prelude::*,
};

/// Finds the part of the pressure source which does not vary
/// in the iteration to find the pressure.
pub fn cpsource(state: &State, mut sp0: ArrayViewMut3<f64>) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;
    let hdzi = (1.0 / 2.0) * (1.0 / (HBAR / nz as f64));

    // Physical space arrays
    let mut ut = arr3zero(ng, nz);
    let mut vt = arr3zero(ng, nz);
    let mut wt = arr3zero(ng, nz);
    let mut hsrc = arr2zero(ng);
    let wkq = arr2zero(ng);

    // Spectral space arrays (all work arrays)
    let mut wka = arr2zero(ng);
    let wkb = arr2zero(ng);

    // Calculate part which is independent of z, -g*Lap_h{h}:
    // wkp = h;
    let mut wkp = state.z.index_axis(Axis(2), nz).to_owned();

    // Fourier transform to spectral space:
    state.spectral.d2fft.ptospc(wkp.view_mut(), wka.view_mut());

    // Apply -g*Lap_h operator:
    wka *= &state.spectral.glap;

    // Return to physical space:
    state.spectral.d2fft.spctop(wka.view_mut(), hsrc.view_mut());
    // hsrc contains -g*Lap{h} in physical space.

    // Calculate u_theta, v_theta & w_theta:

    // Lower boundary (use higher order formula):
    (0..=nz as u16)
        .into_par_iter()
        .zip(ut.axis_iter_mut(Axis(2)).into_par_iter())
        .zip(vt.axis_iter_mut(Axis(2)).into_par_iter())
        .zip(wt.axis_iter_mut(Axis(2)).into_par_iter())
        .for_each(|(((iz, ut), vt), wt)| {
            let iz = iz as usize;

            if iz == 0 {
                // Lower boundary (use higher order formula):
                Zip::from(ut)
                    .and(state.u.index_axis(Axis(2), 0))
                    .and(state.u.index_axis(Axis(2), 1))
                    .and(state.u.index_axis(Axis(2), 2))
                    .apply(|ut, u0, u1, u2| *ut = hdzi * (4.0 * u1 - 3.0 * u0 - u2));

                Zip::from(vt)
                    .and(state.v.index_axis(Axis(2), 0))
                    .and(state.v.index_axis(Axis(2), 1))
                    .and(state.v.index_axis(Axis(2), 2))
                    .apply(|vt, v0, v1, v2| *vt = hdzi * (4.0 * v1 - 3.0 * v0 - v2));

                Zip::from(wt)
                    .and(state.w.index_axis(Axis(2), 0))
                    .and(state.w.index_axis(Axis(2), 1))
                    .and(state.w.index_axis(Axis(2), 2))
                    .apply(|wt, w0, w1, w2| *wt = hdzi * (4.0 * w1 - 3.0 * w0 - w2));
            } else if iz == nz {
                // Upper boundary (use higher order formula):
                Zip::from(ut)
                    .and(state.u.index_axis(Axis(2), nz))
                    .and(state.u.index_axis(Axis(2), nz - 1))
                    .and(state.u.index_axis(Axis(2), nz - 2))
                    .apply(|ut, u, u1, u2| *ut = hdzi * (3.0 * u + u2 - 4.0 * u1));
                Zip::from(vt)
                    .and(state.v.index_axis(Axis(2), nz))
                    .and(state.v.index_axis(Axis(2), nz - 1))
                    .and(state.v.index_axis(Axis(2), nz - 2))
                    .apply(|vt, v, v1, v2| *vt = hdzi * (3.0 * v + v2 - 4.0 * v1));
                Zip::from(wt)
                    .and(state.w.index_axis(Axis(2), nz))
                    .and(state.w.index_axis(Axis(2), nz - 1))
                    .and(state.w.index_axis(Axis(2), nz - 2))
                    .apply(|wt, w, w1, w2| *wt = hdzi * (3.0 * w + w2 - 4.0 * w1));
            } else {
                // Interior (centred differencing):
                Zip::from(ut)
                    .and(state.u.index_axis(Axis(2), iz + 1))
                    .and(state.u.index_axis(Axis(2), iz - 1))
                    .apply(|ut, up, um| *ut = hdzi * (up - um));
                Zip::from(vt)
                    .and(state.v.index_axis(Axis(2), iz + 1))
                    .and(state.v.index_axis(Axis(2), iz - 1))
                    .apply(|vt, vp, vm| *vt = hdzi * (vp - vm));
                Zip::from(wt)
                    .and(state.w.index_axis(Axis(2), iz + 1))
                    .and(state.w.index_axis(Axis(2), iz - 1))
                    .apply(|wt, wp, wm| *wt = hdzi * (wp - wm));
            }
        });

    // Loop over layers and build up source, sp0:

    (0..=nz as u16)
        .into_par_iter()
        .zip(sp0.axis_iter_mut(Axis(2)).into_par_iter())
        .for_each(|(iz, mut sp0)| {
            let iz = iz as usize;

            let mut wka = wka.clone();
            let mut wkb = wkb.clone();
            let mut wkp = wkp.clone();
            let mut wkq = wkq.clone();
            let mut wkr = arr2zero(ng);
            let mut ux = arr2zero(ng);
            let mut uy = arr2zero(ng);
            let mut vx = arr2zero(ng);
            let mut vy = arr2zero(ng);
            let mut wx = arr2zero(ng);
            let mut wy = arr2zero(ng);

            // Calculate u_x, u_y, v_x & v_y:
            wkq.assign(&state.u.index_axis(Axis(2), iz));
            state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
            state
                .spectral
                .d2fft
                .xderiv(&state.spectral.hrkx, wka.view(), wkb.view_mut());
            state.spectral.d2fft.spctop(wkb.view_mut(), ux.view_mut());
            state
                .spectral
                .d2fft
                .yderiv(&state.spectral.hrky, wka.view(), wkb.view_mut());
            state.spectral.d2fft.spctop(wkb.view_mut(), uy.view_mut());

            wkq.assign(&state.v.index_axis(Axis(2), iz));

            state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
            state
                .spectral
                .d2fft
                .xderiv(&state.spectral.hrkx, wka.view(), wkb.view_mut());
            state.spectral.d2fft.spctop(wkb.view_mut(), vx.view_mut());
            state
                .spectral
                .d2fft
                .yderiv(&state.spectral.hrky, wka.view(), wkb.view_mut());
            state.spectral.d2fft.spctop(wkb.view_mut(), vy.view_mut());

            if iz == 0 {
                // iz = 0 is much simpler as z = w = 0 there:

                Zip::from(&mut wkq)
                    .and(state.ri.index_axis(Axis(2), 0))
                    .and(wt.index_axis(Axis(2), 0))
                    .apply(|wkq, ri, wt| *wkq = ri * wt);

                state.spectral.deal2d(wkq.view_mut());

                Zip::from(&mut sp0)
                    .and(&hsrc)
                    .and(state.zeta.index_axis(Axis(2), 0))
                    .apply(|sp0, hsrc, zeta| *sp0 = hsrc + COF * zeta);

                Zip::from(&mut sp0)
                    .and(&wkq)
                    .and(&ux)
                    .and(&uy)
                    .and(&vx)
                    .and(&vy)
                    .apply(|sp0, wkq, ux, uy, vx, vy| {
                        *sp0 += 2.0 * (ux * vy - uy * vx + wkq * (ux + vy))
                    });
            } else {
                // Calculate w_x, w_y:

                wkq.assign(&state.w.index_axis(Axis(2), iz));

                state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
                state
                    .spectral
                    .d2fft
                    .xderiv(&state.spectral.hrkx, wka.view(), wkb.view_mut());
                state.spectral.d2fft.spctop(wkb.view_mut(), wx.view_mut());
                state
                    .spectral
                    .d2fft
                    .yderiv(&state.spectral.hrky, wka.view(), wkb.view_mut());
                state.spectral.d2fft.spctop(wkb.view_mut(), wy.view_mut());

                // Calculate pressure source:
                Zip::from(&mut wkp)
                    .and(vt.index_axis(Axis(2), iz))
                    .and(ut.index_axis(Axis(2), iz))
                    .and(state.zx.index_axis(Axis(2), iz))
                    .and(state.zy.index_axis(Axis(2), iz))
                    .apply(|wkp, vt, ut, zx, zy| *wkp = vt * zx - ut * zy);

                state.spectral.deal2d(wkp.view_mut());

                Zip::from(&mut wkq)
                    .and(&uy)
                    .and(vt.index_axis(Axis(2), iz))
                    .and(ut.index_axis(Axis(2), iz))
                    .and(&vy)
                    .apply(|wkq, uy, vt, ut, vy| *wkq = uy * vt - ut * vy);

                state.spectral.deal2d(wkq.view_mut());

                Zip::from(&mut wkr)
                    .and(ut.index_axis(Axis(2), iz))
                    .and(vt.index_axis(Axis(2), iz))
                    .and(&vx)
                    .and(&ux)
                    .apply(|wkr, ut, vt, vx, ux| *wkr = ut * vx - ux * vt);

                state.spectral.deal2d(wkr.view_mut());

                Zip::from(&mut wkq)
                    .and(&wkr)
                    .and(state.zx.index_axis(Axis(2), iz))
                    .and(state.zy.index_axis(Axis(2), iz))
                    .and(&wy)
                    .and(vt.index_axis(Axis(2), iz))
                    .apply(|wkq, wkr, zx, zy, wy, vt| *wkq = *wkq * zx + wkr * zy - wy * vt);

                Zip::from(&mut wkq)
                    .and(&ux)
                    .and(&vy)
                    .and(ut.index_axis(Axis(2), iz))
                    .and(wt.index_axis(Axis(2), iz))
                    .and(&wx)
                    .apply(|wkq, ux, vy, ut, wt, wx| *wkq += (ux + vy) * wt - wx * ut);

                state.spectral.deal2d(wkq.view_mut());

                //sp0(:,:,iz)=hsrc+cof*(zeta(:,:,iz)-ri(:,:,iz)*wkp)+ two*(ux*vy-uy*vx+ri(:,:,iz)*wkq);
                Zip::from(&mut sp0)
                    .and(&hsrc)
                    .and(state.zeta.index_axis(Axis(2), iz))
                    .and(state.ri.index_axis(Axis(2), iz))
                    .and(&wkp)
                    .and(&wkq)
                    .apply(|sp0, hsrc, zeta, ri, wkp, wkq| {
                        *sp0 = hsrc + COF * (zeta - ri * wkp) + 2.0 * (ri * wkq)
                    });

                Zip::from(&mut sp0)
                    .and(&ux)
                    .and(&vy)
                    .and(&uy)
                    .and(&vx)
                    .apply(|sp0, ux, vy, uy, vx| *sp0 += 2.0 * (ux * vy - uy * vx));
            }
        });
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            array3_from_file,
            nhswps::{Output, Spectral},
            utils::*,
        },
        byteorder::{ByteOrder, NetworkEndian},
        lazy_static::lazy_static,
        ndarray::{Array3, ShapeBuilder},
    };

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_ri.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/18_2_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w,
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zeta,
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
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_ri.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_v.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_w.bin");
            let zeta = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zeta.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/cpsource/32_4_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w,
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zeta,
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
    fn _18_2() {
        let mut sp0 = include_bytes!("testdata/cpsource/18_2_sp0.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sp02 = include_bytes!("testdata/cpsource/18_2_sp02.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        cpsource(&STATE_18_2, viewmut3d(&mut sp0, 18, 18, 3));

        assert_approx_eq_slice(&sp02, &sp0);
    }

    #[test]
    fn _32_4() {
        let mut sp0 = include_bytes!("testdata/cpsource/32_4_sp0.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sp02 = include_bytes!("testdata/cpsource/32_4_sp02.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        cpsource(&STATE_32_4, viewmut3d(&mut sp0, 32, 32, 5));

        assert_approx_eq_slice(&sp02, &sp0);
    }
}
