use crate::{constants::*, nhswps::State, utils::*};

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

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            array3_from_file,
            nhswps::{Output, Spectral},
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

        cpsource(&STATE_18_2, &mut sp0);

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

        cpsource(&STATE_32_4, &mut sp0);

        assert_approx_eq_slice(&sp02, &sp0);
    }
}
