use crate::{constants::*, nhswps::State, utils::*};

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
                    wkd_matrix[[i, j]] += state.spectral.weight[iz] * state.aa[[i, j, iz]]
                }
            }
        }

        //Note: aa contains div(u*rho_theta) in spectral space
        for i in 0..ng {
            for j in 0..ng {
                wkd_matrix[[i, j]] *= state.spectral.c2g2[[i, j]]
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

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn,
                ps,
                qs,
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

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn,
                ps,
                qs,
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
    fn _18_2_sqs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/18_2_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _18_2_sds() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/18_2_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _18_2_sgs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/18_2_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_18_2, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sgs2, &sgs);
    }

    #[test]
    fn _32_4_sqs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/32_4_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _32_4_sds() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/32_4_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _32_4_sgs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/32_4_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(&STATE_32_4, &mut sqs, &mut sds, &mut sgs);

        assert_approx_eq_slice(&sgs2, &sgs);
    }
}
