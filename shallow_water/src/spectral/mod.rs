//! Module containing subroutines for spectral operations, inversion, etc

#[cfg(test)]
mod test;

use {
    crate::{constants::*, sta2dfft::D2FFT, utils::*},
    core::f64::consts::PI,
    ndarray::{Array2, Array3},
};

#[derive(Debug, PartialEq, Clone)]
pub struct Spectral {
    // Spectral operators
    pub hlap: Array2<f64>,
    pub glap: Array2<f64>,
    pub rlap: Array2<f64>,
    pub helm: Array2<f64>,

    pub c2g2: Array2<f64>,
    pub simp: Array2<f64>,
    pub rope: Array2<f64>,
    pub fope: Array2<f64>,

    pub filt: Array2<f64>,
    pub diss: Array2<f64>,
    pub opak: Array2<f64>,
    pub rdis: Array2<f64>,

    // Tridiagonal arrays for the pressure Poisson equation
    pub etdv: Array3<f64>,
    pub htdv: Array3<f64>,
    pub ap: Array2<f64>,

    // Tridiagonal arrays for the compact difference calculation of d/dz
    pub etd1: Vec<f64>,
    pub htd1: Vec<f64>,

    // Array for theta and vertical weights for integration:
    pub theta: Vec<f64>,
    pub weight: Vec<f64>,

    // For 2D FFTs
    pub hrkx: Vec<f64>,
    pub hrky: Vec<f64>,
    pub rk: Vec<f64>,
    pub d2fft: D2FFT,

    pub spmf: Vec<f64>,
    pub alk: Vec<f64>,
    pub kmag: Array2<usize>,
    pub kmax: usize,
    pub kmaxred: usize,

    pub ng: usize,
    pub nz: usize,
}

impl Spectral {
    pub fn new(ng: usize, nz: usize) -> Self {
        let dt = 1.0 / (ng as f64);
        let dt2 = dt * (1.0 / 2.0);
        let dt2i = 1.0 / dt2;
        let dz = HBAR / (nz as f64);
        let dzi = 1.0 / dz;
        let dzisq = dzi.powf(2.0);

        let mut hlap = Array2::<f64>::zeros((ng, ng));
        let mut glap = Array2::<f64>::zeros((ng, ng));
        let mut rlap = Array2::<f64>::zeros((ng, ng));
        let mut helm = Array2::<f64>::zeros((ng, ng));
        let mut c2g2 = Array2::<f64>::zeros((ng, ng));
        let mut simp = Array2::<f64>::zeros((ng, ng));
        let mut rope = Array2::<f64>::zeros((ng, ng));
        let mut fope = Array2::<f64>::zeros((ng, ng));
        let mut filt = Array2::<f64>::zeros((ng, ng));
        let mut diss = Array2::<f64>::zeros((ng, ng));
        let mut opak = Array2::<f64>::zeros((ng, ng));
        let mut rdis = Array2::<f64>::zeros((ng, ng));

        let mut etdv = Array3::<f64>::zeros((ng, ng, nz));
        let mut htdv = Array3::<f64>::zeros((ng, ng, nz));
        let mut ap = Array2::<f64>::zeros((ng, ng));
        let mut etd1 = vec![0.0; nz];
        let mut htd1 = vec![0.0; nz];
        let mut theta = vec![0.0; nz + 1];
        let mut weight = vec![0.0; nz + 1];
        let mut hrkx = vec![0.0; ng];
        let mut hrky = vec![0.0; ng];
        let mut rk = vec![0.0; ng];
        let mut spmf = vec![0.0; ng + 1];
        let mut alk = vec![0.0; ng];
        let mut kmag = Array2::<usize>::zeros((ng, ng));
        let kmax;
        let kmaxred;

        let mut a0 = Array2::<f64>::zeros((ng, ng));
        let mut a0b = Array2::<f64>::zeros((ng, ng));
        let mut apb = Array2::<f64>::zeros((ng, ng));

        let rkmax: f64;
        let mut rks: f64;
        let snorm: f64;
        let mut rrsq: f64;

        let anu: f64;
        let rkfsq: f64;

        let d2fft = D2FFT::new(ng, ng, 2.0 * PI, 2.0 * PI, &mut hrkx, &mut hrky);

        //Define wavenumbers and filtered wavenumbers:
        rk[0] = 0.0;
        for k in 1..ng / 2 {
            rk[k] = hrkx[2 * k - 1];
            rk[ng - k] = hrkx[2 * k - 1];
        }
        rk[ng / 2] = hrkx[ng - 1];

        //Initialise arrays for computing the spectrum of any field:
        rkmax = (ng / 2) as f64;
        kmax = (rkmax * 2.0f64.sqrt()).ceil() as usize;
        for e in spmf.iter_mut().take(kmax + 1) {
            *e = 0.0;
        }
        for ky in 0..ng {
            for kx in 0..ng {
                let k = ((rk[kx].powf(2.0) + rk[ky].powf(2.0)).sqrt()).round() as usize;
                kmag[[kx, ky]] = k;
                spmf[k] += 1.0;
            }
        }

        //Compute spectrum multiplication factor (spmf) to account for unevenly
        //sampled shells and normalise spectra by 8/(ng*ng) so that the sum
        //of the spectrum is equal to the L2 norm of the original field:
        snorm = 4.0 * PI / (ng * ng) as f64;
        spmf[0] = 0.0;
        for k in 1..=kmax {
            spmf[k] = snorm * k as f64 / spmf[k];
            alk[k - 1] = (k as f64).log10();
        }

        //Only output shells which are fully occupied (k <= kmaxred):
        kmaxred = ng / 2;

        //Define a variety of spectral operators:

        //Hyperviscosity coefficient (Dritschel, Gottwald & Oliver, JFM (2017)):
        anu = CDAMP * COF / rkmax.powf(2.0 * NNU);
        //Assumes Burger number = 1.

        //Used for de-aliasing filter below:
        rkfsq = ((ng as f64) / 3.0).powf(2.0);

        for ky in 0..ng {
            for kx in 0..ng {
                rks = rk[kx].powf(2.0) + rk[ky].powf(2.0);
                //grad^2:
                hlap[[kx, ky]] = -rks;
                //Spectral c^2*grad^2 - f^2 operator (G in paper):
                opak[[kx, ky]] = -(FSQ + CSQ * rks);
                //Hyperviscous operator:
                diss[[kx, ky]] = anu * rks.powf(NNU);
                //De-aliasing filter:
                if rks > rkfsq {
                    filt[[kx, ky]] = 0.0;
                    glap[[kx, ky]] = 0.0;
                    c2g2[[kx, ky]] = 0.0;
                    rlap[[kx, ky]] = 0.0;
                    helm[[kx, ky]] = 0.0;
                    rope[[kx, ky]] = 0.0;
                    rdis[[kx, ky]] = 0.0;
                } else {
                    filt[[kx, ky]] = 1.0;
                    //-g*grad^2:
                    glap[[kx, ky]] = GRAVITY * rks;
                    //c^2*grad^2:
                    c2g2[[kx, ky]] = -CSQ * rks;
                    //grad^{-2} (inverse Laplacian):
                    rlap[[kx, ky]] = -1.0 / (rks + 1.0E-20);
                    //(c^2*grad^2 - f^2)^{-1} (G^{-1} in paper):
                    helm[[kx, ky]] = 1.0 / opak[[kx, ky]];
                    //c^2*grad^2/(c^2*grad^2 - f^2) (used in layer thickness inversion):
                    rope[[kx, ky]] = c2g2[[kx, ky]] * helm[[kx, ky]];
                    rdis[[kx, ky]] = dt2i + diss[[kx, ky]];
                }
                //Operators needed for semi-implicit time stepping:
                rrsq = (dt2i + diss[[kx, ky]]).powf(2.0);
                fope[[kx, ky]] = -c2g2[[kx, ky]] / (rrsq - opak[[kx, ky]]);
                //Semi-implicit operator for inverting divergence:
                simp[[kx, ky]] = 1.0 / (rrsq + FSQ);
                //Re-define damping operator for use in qd evolution:
                diss[[kx, ky]] = 2.0 / (1.0 + dt2 * diss[[kx, ky]]);
            }
        }

        //Ensure area averages remain zero:
        rlap[[0, 0]] = 0.0;

        //Define theta and the vertical weight for trapezoidal integration:
        for iz in 0..=nz {
            theta[iz] = dz * (iz as f64);
            weight[iz] = 1.0;
        }
        weight[0] = 0.5;
        weight[nz] = 0.5;
        for e in weight.iter_mut() {
            *e /= nz as f64;
        }

        //Tridiagonal coefficients depending only on kx and ky:
        for ky in 0..ng {
            for kx in 0..ng {
                rks = rk[kx].powf(2.0) + rk[ky].powf(2.0);
                a0[[kx, ky]] = -2.0 * dzisq - (5.0 / 6.0) * rks;
                a0b[[kx, ky]] = -dzisq - (1.0 / 3.0) * rks;
                ap[[kx, ky]] = dzisq - (1.0 / 12.0) * rks;
                apb[[kx, ky]] = dzisq - (1.0 / 6.0) * rks;
            }
        }
        //Tridiagonal arrays for the pressure:

        for i in 0..ng {
            for j in 0..ng {
                htdv[[i, j, 0]] = filt[[i, j]] / a0b[[i, j]];
                etdv[[i, j, 0]] = -apb[[i, j]] * htdv[[i, j, 0]];
                for iz in 1..=nz - 2 {
                    htdv[[i, j, iz]] =
                        filt[[i, j]] / (a0[[i, j]] + ap[[i, j]] * etdv[[i, j, iz - 1]]);
                    etdv[[i, j, iz]] = -ap[[i, j]] * htdv[[i, j, iz]];
                }
                htdv[[i, j, nz - 1]] =
                    filt[[i, j]] / (a0[[i, j]] + ap[[i, j]] * etdv[[i, j, nz - 2]]);
            }
        }

        //Tridiagonal arrays for the compact difference calculation of d/dz:
        htd1[0] = 1.0 / (2.0 / 3.0);
        etd1[0] = -(1.0 / 6.0) * htd1[0];
        for iz in 2..nz {
            htd1[iz - 1] = 1.0 / ((2.0 / 3.0) + (1.0 / 6.0) * etd1[iz - 2]);
            etd1[iz - 1] = -(1.0 / 6.0) * htd1[iz - 1];
        }
        htd1[nz - 1] = 1.0 / ((2.0 / 3.0) + (1.0 / 3.0) * etd1[nz - 2]);

        Self {
            hlap,
            glap,
            rlap,
            helm,
            c2g2,
            simp,
            rope,
            fope,
            filt,
            diss,
            opak,
            rdis,
            etdv,
            htdv,
            ap,
            etd1,
            htd1,
            theta,
            weight,
            hrkx,
            hrky,
            rk,
            d2fft,
            spmf,
            alk,
            kmag,
            kmax,
            kmaxred,
            ng,
            nz,
        }
    }

    /// Given the PV anomaly qs, divergence ds and acceleration divergence gs
    /// (all in spectral space), this routine computes the dimensionless
    /// layer thickness anomaly and horizontal velocity, as well as the
    /// relative vertical vorticity in physical space.
    #[allow(clippy::cognitive_complexity)]
    pub fn main_invert(
        &self,
        qs: &[f64],
        ds: &[f64],
        gs: &[f64],
        r: &mut [f64],
        u: &mut [f64],
        v: &mut [f64],
        zeta: &mut [f64],
    ) {
        let dsumi = 1.0 / (self.ng * self.ng) as f64;

        let mut es = vec![0.0; self.ng * self.ng * (self.nz + 1)];

        let mut wka = vec![0.0; self.ng * self.ng];
        let mut wkb = vec![0.0; self.ng * self.ng];
        let mut wkc = vec![0.0; self.ng * self.ng];
        let mut wkd = vec![0.0; self.ng * self.ng];
        let mut wke = vec![0.0; self.ng * self.ng];
        let mut wkf = vec![0.0; self.ng * self.ng];
        let mut wkg = vec![0.0; self.ng * self.ng];
        let mut wkh = vec![0.0; self.ng * self.ng];

        let mut uio: f64;
        let mut vio: f64;

        //Define eta = gamma_l/f^2 - q_l/f (spectral):
        for (i, e) in es.iter_mut().enumerate() {
            *e = COFI * (COFI * gs[i] - qs[i]);
        }

        //Compute vertical average of eta (store in wkh):
        for e in wkh.iter_mut() {
            *e = 0.0;
        }

        let mut wkh_matrix = viewmut2d(&mut wkh, self.ng, self.ng);
        let es_matrix = view3d(&es, self.ng, self.ng, self.nz + 1);

        for iz in 0..=self.nz {
            for j in 0..self.ng {
                for i in 0..self.ng {
                    wkh_matrix[[i, j]] += self.weight[iz] * es_matrix[[i, j, iz]];
                }
            }
        }

        //Multiply by F = c^2*k^2/(f^2+c^2k^2) in spectral space:
        for i in 0..self.ng {
            for j in 0..self.ng {
                wkh_matrix[[i, j]] *= self.rope[[i, j]];
            }
        }

        //Initialise mean flow:
        uio = 0.0;
        vio = 0.0;

        //Complete inversion:
        for iz in 0..=self.nz {
            //Obtain layer thickness anomaly (spectral, in wka):
            let es_matrix = view3d(&es, self.ng, self.ng, self.nz + 1);
            let wkh_matrix = view2d(&wkh, self.ng, self.ng);
            let mut wka_matrix = viewmut2d(&mut wka, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wka_matrix[[i, j]] = es_matrix[[i, j, iz]] - wkh_matrix[[i, j]];
                }
            }

            //Obtain relative vorticity (spectral, in wkb):
            //wkb=qs(:,:,iz)+COF*wka;
            let qs_matrix = view3d(&qs, self.ng, self.ng, self.nz + 1);
            let wka_matrix = view2d(&wka, self.ng, self.ng);
            let mut wkb_matrix = viewmut2d(&mut wkb, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkb_matrix[[i, j]] = qs_matrix[[i, j, iz]] + COF * wka_matrix[[i, j]];
                }
            }

            //Invert Laplace operator on zeta & delta to define velocity:
            let wkb_matrix = view2d(&wkb, self.ng, self.ng);
            let ds_matrix = view3d(&ds, self.ng, self.ng, self.nz + 1);
            let mut wkc_matrix = viewmut2d(&mut wkc, self.ng, self.ng);
            let mut wkd_matrix = viewmut2d(&mut wkd, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkc_matrix[[i, j]] = self.rlap[[i, j]] * wkb_matrix[[i, j]];
                    wkd_matrix[[i, j]] = self.rlap[[i, j]] * ds_matrix[[i, j, iz]];
                }
            }

            //Calculate derivatives spectrally:
            self.d2fft.xderiv(&self.hrkx, &wkd, &mut wke);
            self.d2fft.yderiv(&self.hrky, &wkd, &mut wkf);
            self.d2fft.xderiv(&self.hrkx, &wkc, &mut wkd);
            self.d2fft.yderiv(&self.hrky, &wkc, &mut wkg);

            //Define velocity components:
            for (e, g) in wke.iter_mut().zip(&wkg) {
                *e -= g;
            }
            //wke=wke-wkg;
            for (f, d) in wkf.iter_mut().zip(&wkd) {
                *f += d;
            }
            //wkf=wkf+wkd;

            //Bring quantities back to physical space and store:
            self.d2fft.spctop(&mut wka, &mut wkc);

            let wkc_matrix = view2d(&wkc, self.ng, self.ng);
            let mut r_matrix = viewmut3d(r, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    r_matrix[[i, j, iz]] = wkc_matrix[[i, j]];
                }
            }

            self.d2fft.spctop(&mut wkb, &mut wkd);

            let wkd_matrix = view2d(&wkd, self.ng, self.ng);
            let mut zeta_matrix = viewmut3d(zeta, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    zeta_matrix[[i, j, iz]] = wkd_matrix[[i, j]];
                }
            }

            self.d2fft.spctop(&mut wke, &mut wka);

            let wka_matrix = view2d(&wka, self.ng, self.ng);
            let mut u_matrix = viewmut3d(u, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    u_matrix[[i, j, iz]] = wka_matrix[[i, j]];
                }
            }

            self.d2fft.spctop(&mut wkf, &mut wkb);

            let wkb_matrix = view2d(&wkb, self.ng, self.ng);
            let mut v_matrix = viewmut3d(v, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    v_matrix[[i, j, iz]] = wkb_matrix[[i, j]];
                }
            }

            //Accumulate mean flow (uio,vio):
            let sum_ca = wkc.iter().zip(&wka).map(|(a, b)| a * b).sum::<f64>();
            let sum_cb = wkc.iter().zip(&wkb).map(|(a, b)| a * b).sum::<f64>();

            uio -= self.weight[iz] * sum_ca * dsumi;
            vio -= self.weight[iz] * sum_cb * dsumi;
        }

        //Add mean flow:
        for e in u.iter_mut() {
            *e += uio;
        }
        for e in v.iter_mut() {
            *e += vio;
        }
    }

    /// Computes the (xy) Jacobian of aa and bb and returns it in cs.
    /// aa and bb are in physical space while cs is in spectral space
    ///
    /// NOTE: aa and bb are assumed to be spectrally truncated (de-aliased).
    pub fn jacob(&self, aa: &[f64], bb: &[f64], cs: &mut [f64]) {
        let mut ax = vec![0.0; self.ng * self.ng];
        let mut ay = vec![0.0; self.ng * self.ng];
        let mut bx = vec![0.0; self.ng * self.ng];
        let mut by = vec![0.0; self.ng * self.ng];
        let mut wka = vec![0.0; self.ng * self.ng];
        let mut wkb = vec![0.0; self.ng * self.ng];

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = aa[i];
        }

        self.d2fft.ptospc(&mut wkb, &mut wka);
        //Get derivatives of aa:
        self.d2fft.xderiv(&self.hrkx, &wka, &mut wkb);
        self.d2fft.spctop(&mut wkb, &mut ax);
        self.d2fft.yderiv(&self.hrky, &wka, &mut wkb);
        self.d2fft.spctop(&mut wkb, &mut ay);

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = bb[i];
        }

        self.d2fft.ptospc(&mut wkb, &mut wka);
        //Get derivatives of bb:
        self.d2fft.xderiv(&self.hrkx, &wka, &mut wkb);
        self.d2fft.spctop(&mut wkb, &mut bx);
        self.d2fft.yderiv(&self.hrky, &wka, &mut wkb);
        self.d2fft.spctop(&mut wkb, &mut by);

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = ax[i] * by[i] - ay[i] * bx[i];
        }
        self.d2fft.ptospc(&mut wkb, cs);
    }

    /// Computes the divergence of (aa,bb) and returns it in cs.
    /// Both aa and bb in physical space but cs is in spectral space.
    pub fn divs(&self, aa: &[f64], bb: &[f64], cs: &mut [f64]) {
        let mut wkp = vec![0.0; self.ng * self.ng];
        let mut wka = vec![0.0; self.ng * self.ng];
        let mut wkb = vec![0.0; self.ng * self.ng];

        for (i, e) in aa.iter().enumerate() {
            wkp[i] = *e;
        }

        self.d2fft.ptospc(&mut wkp, &mut wka);
        self.d2fft.xderiv(&self.hrkx, &wka, &mut wkb);

        for (i, e) in bb.iter().enumerate() {
            wkp[i] = *e;
        }

        self.d2fft.ptospc(&mut wkp, &mut wka);
        self.d2fft.yderiv(&self.hrky, &wka, cs);

        for (i, e) in cs.iter_mut().enumerate() {
            *e += wkb[i];
        }
    }

    /// Transforms a physical 3d field fp to spectral space (horizontally)
    /// as the array fs.
    pub fn ptospc3d(&self, fp: &[f64], fs: &mut [f64], izbeg: usize, izend: usize) {
        let mut wkp = vec![0.0; self.ng * self.ng];
        let mut wks = vec![0.0; self.ng * self.ng];

        for iz in izbeg..=izend {
            let mut wkp_matrix = viewmut2d(&mut wkp, self.ng, self.ng);
            let fp_matrix = view3d(fp, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkp_matrix[[i, j]] = fp_matrix[[i, j, iz]];
                }
            }

            self.d2fft.ptospc(&mut wkp, &mut wks);

            let wks_matrix = view2d(&wks, self.ng, self.ng);
            let mut fs_matrix = viewmut3d(fs, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    fs_matrix[[i, j, iz]] = wks_matrix[[i, j]];
                }
            }
        }
    }

    /// Transforms a spectral 3d field fs to physical space (horizontally)
    /// as the array fp.
    pub fn spctop3d(&self, fs: &[f64], fp: &mut [f64], izbeg: usize, izend: usize) {
        let mut wks = vec![0.0; self.ng * self.ng];
        let mut wkp = vec![0.0; self.ng * self.ng];

        for iz in izbeg..=izend {
            let mut wks_matrix = viewmut2d(&mut wks, self.ng, self.ng);
            let fs_matrix = view3d(fs, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wks_matrix[[i, j]] = fs_matrix[[i, j, iz]];
                }
            }

            self.d2fft.spctop(&mut wks, &mut wkp);

            let wkp_matrix = view2d(&wkp, self.ng, self.ng);
            let mut fp_matrix = viewmut3d(fp, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    fp_matrix[[i, j, iz]] = wkp_matrix[[i, j]];
                }
            }
        }
    }

    /// Filters (horizontally) a physical 3d field fp (overwrites fp)
    pub fn deal3d(&self, fp: &mut [f64]) {
        let mut fp_matrix = viewmut3d(fp, self.ng, self.ng, self.nz + 1);

        let mut wkp_matrix = Array2::<f64>::zeros((self.ng, self.ng));
        let mut wks_matrix = Array2::<f64>::zeros((self.ng, self.ng));

        for iz in 0..=self.nz {
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkp_matrix[[i, j]] = fp_matrix[[i, j, iz]];
                }
            }

            self.d2fft.ptospc(
                wkp_matrix.as_slice_mut().unwrap(),
                wks_matrix.as_slice_mut().unwrap(),
            );

            for i in 0..self.ng {
                for j in 0..self.ng {
                    wks_matrix[[i, j]] *= self.filt[[i, j]];
                }
            }

            self.d2fft.spctop(
                wks_matrix.as_slice_mut().unwrap(),
                wkp_matrix.as_slice_mut().unwrap(),
            );

            for i in 0..self.ng {
                for j in 0..self.ng {
                    fp_matrix[[i, j, iz]] = wkp_matrix[[i, j]];
                }
            }
        }
    }

    /// Filters (horizontally) a physical 2d field fp (overwrites fp)
    pub fn deal2d(&self, fp: &mut [f64]) {
        let mut fs = Array2::<f64>::zeros((self.ng, self.ng));

        self.d2fft.ptospc(fp, fs.as_slice_mut().unwrap());

        for i in 0..self.ng {
            for j in 0..self.ng {
                fs[[i, j]] *= self.filt[[i, j]]
            }
        }

        self.d2fft.spctop(fs.as_slice_mut().unwrap(), fp);
    }

    /// Computes the 1d spectrum of a spectral field ss and returns the
    /// result in spec
    pub fn spec1d(&self, ss: &[f64], spec: &mut [f64]) {
        assert_eq!(self.ng * self.ng, ss.len());
        assert_eq!(self.ng + 1, spec.len());

        let ss = view2d(ss, self.ng, self.ng);

        for k in 0..=self.kmax {
            spec[k] = 0.0;
        }

        // x and y-independent mode:
        let k = self.kmag[[0, 0]];
        spec[k] += (1.0 / 4.0) * ss[[0, 0]].powf(2.0);

        // y-independent mode:
        for kx in 1..self.ng {
            let k = self.kmag[[kx, 0]];
            spec[k] += (1.0 / 2.0) * ss[[kx, 0]].powf(2.0)
        }

        // x-independent mode:
        for ky in 1..self.ng {
            let k = self.kmag[[0, ky]];
            spec[k] += (1.0 / 2.0) * ss[[0, ky]].powf(2.0)
        }

        // All other modes:
        for ky in 1..self.ng {
            for kx in 1..self.ng {
                let k = self.kmag[[kx, ky]];
                spec[k] += ss[[kx, ky]].powf(2.0)
            }
        }
    }
}
