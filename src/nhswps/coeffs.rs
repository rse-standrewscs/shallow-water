use crate::{constants::*, nhswps::State, utils::*};

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

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_ri.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/18_2_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u: Array3::<f64>::zeros((ng, ng, nz + 1)),
                v: Array3::<f64>::zeros((ng, ng, nz + 1)),
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
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

            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_ri.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/coeffs/32_4_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u: Array3::<f64>::zeros((ng, ng, nz + 1)),
                v: Array3::<f64>::zeros((ng, ng, nz + 1)),
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa: Array3::<f64>::zeros((ng, ng, nz + 1)),
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
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
    fn _18_2_sigx() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigx2 = include_bytes!("testdata/coeffs/18_2_sigx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigx2, &sigx);
    }

    #[test]
    fn _18_2_sigy() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigy2 = include_bytes!("testdata/coeffs/18_2_sigy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigy2, &sigy);
    }

    #[test]
    fn _18_2_cpt1() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt12 = include_bytes!("testdata/coeffs/18_2_cpt12.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt12, &cpt1);
    }

    #[test]
    fn _18_2_cpt2() {
        let mut sigx = include_bytes!("testdata/coeffs/18_2_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/18_2_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/18_2_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/18_2_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt22 = include_bytes!("testdata/coeffs/18_2_cpt22.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_18_2, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt22, &cpt2);
    }

    #[test]
    fn _32_4_sigx() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigx2 = include_bytes!("testdata/coeffs/32_4_sigx2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigx2, &sigx);
    }

    #[test]
    fn _32_4_sigy() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sigy2 = include_bytes!("testdata/coeffs/32_4_sigy2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&sigy2, &sigy);
    }

    #[test]
    fn _32_4_cpt1() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt12 = include_bytes!("testdata/coeffs/32_4_cpt12.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt12, &cpt1);
    }

    #[test]
    fn _32_4_cpt2() {
        let mut sigx = include_bytes!("testdata/coeffs/32_4_sigx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sigy = include_bytes!("testdata/coeffs/32_4_sigy.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt1 = include_bytes!("testdata/coeffs/32_4_cpt1.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut cpt2 = include_bytes!("testdata/coeffs/32_4_cpt2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let cpt22 = include_bytes!("testdata/coeffs/32_4_cpt22.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        coeffs(&STATE_32_4, &mut sigx, &mut sigy, &mut cpt1, &mut cpt2);

        assert_approx_eq_slice(&cpt22, &cpt2);
    }
}
