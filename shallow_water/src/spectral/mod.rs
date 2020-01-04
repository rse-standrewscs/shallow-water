//! Module containing subroutines for spectral operations, inversion, etc

#[cfg(test)]
mod test;

use {
    crate::{
        constants::*,
        sta2dfft::{init2dfft, ptospc, spctop, xderiv, yderiv},
        utils::{_2d_to_vec, _3d_to_vec, slice_to_2d, slice_to_3d},
    },
    core::f64::consts::PI,
};

#[derive(Debug, PartialEq, Clone)]
pub struct Spectral {
    // Spectral operators
    pub hlap: Vec<Vec<f64>>,
    pub glap: Vec<Vec<f64>>,
    pub rlap: Vec<Vec<f64>>,
    pub helm: Vec<Vec<f64>>,

    pub c2g2: Vec<Vec<f64>>,
    pub simp: Vec<Vec<f64>>,
    pub rope: Vec<Vec<f64>>,
    pub fope: Vec<Vec<f64>>,

    pub filt: Vec<Vec<f64>>,
    pub diss: Vec<Vec<f64>>,
    pub opak: Vec<Vec<f64>>,
    pub rdis: Vec<Vec<f64>>,

    // Tridiagonal arrays for the pressure Poisson equation
    pub etdv: Vec<Vec<Vec<f64>>>,
    pub htdv: Vec<Vec<Vec<f64>>>,
    pub ap: Vec<Vec<f64>>,

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
    pub xtrig: Vec<f64>,
    pub ytrig: Vec<f64>,
    pub xfactors: [usize; 5],
    pub yfactors: [usize; 5],

    pub spmf: Vec<f64>,
    pub alk: Vec<f64>,
    pub kmag: Vec<Vec<usize>>,
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

        let mut hlap = vec![vec![0.0; ng]; ng];
        let mut glap = vec![vec![0.0; ng]; ng];
        let mut rlap = vec![vec![0.0; ng]; ng];
        let mut helm = vec![vec![0.0; ng]; ng];
        let mut c2g2 = vec![vec![0.0; ng]; ng];
        let mut simp = vec![vec![0.0; ng]; ng];
        let mut rope = vec![vec![0.0; ng]; ng];
        let mut fope = vec![vec![0.0; ng]; ng];
        let mut filt = vec![vec![0.0; ng]; ng];
        let mut diss = vec![vec![0.0; ng]; ng];
        let mut opak = vec![vec![0.0; ng]; ng];
        let mut rdis = vec![vec![0.0; ng]; ng];
        let mut etdv = vec![vec![vec![0.0; nz]; ng]; ng];
        let mut htdv = vec![vec![vec![0.0; nz]; ng]; ng];
        let mut ap = vec![vec![0.0; ng]; ng];
        let mut etd1 = vec![0.0; nz];
        let mut htd1 = vec![0.0; nz];
        let mut theta = vec![0.0; nz + 1];
        let mut weight = vec![0.0; nz + 1];
        let mut hrkx = vec![0.0; ng];
        let mut hrky = vec![0.0; ng];
        let mut rk = vec![0.0; ng];
        let mut xtrig = vec![0.0; 2 * ng];
        let mut ytrig = vec![0.0; 2 * ng];
        let mut xfactors = [0; 5];
        let mut yfactors = [0; 5];
        let mut spmf = vec![0.0; ng + 1];
        let mut alk = vec![0.0; ng];
        let mut kmag = vec![vec![0; ng]; ng];
        let kmax;
        let kmaxred;

        let mut a0 = vec![vec![0.0; ng]; ng];
        let mut a0b = vec![vec![0.0; ng]; ng];
        let mut apb = vec![vec![0.0; ng]; ng];

        let rkmax: f64;
        let mut rks: f64;
        let snorm: f64;
        let mut rrsq: f64;

        let anu: f64;
        let rkfsq: f64;

        init2dfft(
            ng,
            ng,
            2.0 * PI,
            2.0 * PI,
            &mut xfactors,
            &mut yfactors,
            &mut xtrig,
            &mut ytrig,
            &mut hrkx,
            &mut hrky,
        );

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
                kmag[kx][ky] = k;
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
                hlap[kx][ky] = -rks;
                //Spectral c^2*grad^2 - f^2 operator (G in paper):
                opak[kx][ky] = -(FSQ + CSQ * rks);
                //Hyperviscous operator:
                diss[kx][ky] = anu * rks.powf(NNU);
                //De-aliasing filter:
                if rks > rkfsq {
                    filt[kx][ky] = 0.0;
                    glap[kx][ky] = 0.0;
                    c2g2[kx][ky] = 0.0;
                    rlap[kx][ky] = 0.0;
                    helm[kx][ky] = 0.0;
                    rope[kx][ky] = 0.0;
                    rdis[kx][ky] = 0.0;
                } else {
                    filt[kx][ky] = 1.0;
                    //-g*grad^2:
                    glap[kx][ky] = GRAVITY * rks;
                    //c^2*grad^2:
                    c2g2[kx][ky] = -CSQ * rks;
                    //grad^{-2} (inverse Laplacian):
                    rlap[kx][ky] = -1.0 / (rks + 1.0E-20);
                    //(c^2*grad^2 - f^2)^{-1} (G^{-1} in paper):
                    helm[kx][ky] = 1.0 / opak[kx][ky];
                    //c^2*grad^2/(c^2*grad^2 - f^2) (used in layer thickness inversion):
                    rope[kx][ky] = c2g2[kx][ky] * helm[kx][ky];
                    rdis[kx][ky] = dt2i + diss[kx][ky];
                }
                //Operators needed for semi-implicit time stepping:
                rrsq = (dt2i + diss[kx][ky]).powf(2.0);
                fope[kx][ky] = -c2g2[kx][ky] / (rrsq - opak[kx][ky]);
                //Semi-implicit operator for inverting divergence:
                simp[kx][ky] = 1.0 / (rrsq + FSQ);
                //Re-define damping operator for use in qd evolution:
                diss[kx][ky] = 2.0 / (1.0 + dt2 * diss[kx][ky]);
            }
        }

        //Ensure area averages remain zero:
        rlap[0][0] = 0.0;

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
                a0[kx][ky] = -2.0 * dzisq - (5.0 / 6.0) * rks;
                a0b[kx][ky] = -dzisq - (1.0 / 3.0) * rks;
                ap[kx][ky] = dzisq - (1.0 / 12.0) * rks;
                apb[kx][ky] = dzisq - (1.0 / 6.0) * rks;
            }
        }
        //Tridiagonal arrays for the pressure:

        for i in 0..ng {
            for j in 0..ng {
                htdv[i][j][0] = filt[i][j] / a0b[i][j];
                etdv[i][j][0] = -apb[i][j] * htdv[i][j][0];
                for iz in 1..=nz - 2 {
                    htdv[i][j][iz] = filt[i][j] / (a0[i][j] + ap[i][j] * etdv[i][j][iz - 1]);
                    etdv[i][j][iz] = -ap[i][j] * htdv[i][j][iz];
                }
                htdv[i][j][nz - 1] = filt[i][j] / (a0[i][j] + ap[i][j] * etdv[i][j][nz - 2]);
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
            xtrig,
            ytrig,
            xfactors,
            yfactors,
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

        let mut wkh_matrix = slice_to_2d(&wkh, self.ng, self.ng);
        let es_matrix = slice_to_3d(&es, self.ng, self.ng, self.nz + 1);

        for iz in 0..=self.nz {
            for j in 0..self.ng {
                for i in 0..self.ng {
                    wkh_matrix[i][j] += self.weight[iz] * es_matrix[i][j][iz];
                }
            }
        }

        //Multiply by F = c^2*k^2/(f^2+c^2k^2) in spectral space:
        for i in 0..self.ng {
            for j in 0..self.ng {
                wkh_matrix[i][j] *= self.rope[i][j];
            }
        }
        wkh = _2d_to_vec(&wkh_matrix);

        //Initialise mean flow:
        uio = 0.0;
        vio = 0.0;

        //Complete inversion:
        for iz in 0..=self.nz {
            //Obtain layer thickness anomaly (spectral, in wka):
            let es_matrix = slice_to_3d(&es, self.ng, self.ng, self.nz + 1);
            let wkh_matrix = slice_to_2d(&wkh, self.ng, self.ng);
            let mut wka_matrix = slice_to_2d(&wka, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wka_matrix[i][j] = es_matrix[i][j][iz] - wkh_matrix[i][j];
                }
            }
            wka = _2d_to_vec(&wka_matrix);

            //Obtain relative vorticity (spectral, in wkb):
            //wkb=qs(:,:,iz)+COF*wka;
            let qs_matrix = slice_to_3d(&qs, self.ng, self.ng, self.nz + 1);
            let wka_matrix = slice_to_2d(&wka, self.ng, self.ng);
            let mut wkb_matrix = slice_to_2d(&wkb, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkb_matrix[i][j] = qs_matrix[i][j][iz] + COF * wka_matrix[i][j];
                }
            }
            wkb = _2d_to_vec(&wkb_matrix);

            //Invert Laplace operator on zeta & delta to define velocity:
            let wkb_matrix = slice_to_2d(&wkb, self.ng, self.ng);
            let ds_matrix = slice_to_3d(&ds, self.ng, self.ng, self.nz + 1);
            let mut wkc_matrix = slice_to_2d(&wkc, self.ng, self.ng);
            let mut wkd_matrix = slice_to_2d(&wkd, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkc_matrix[i][j] = self.rlap[i][j] * wkb_matrix[i][j];
                    wkd_matrix[i][j] = self.rlap[i][j] * ds_matrix[i][j][iz];
                }
            }
            wkc = _2d_to_vec(&wkc_matrix);
            wkd = _2d_to_vec(&wkd_matrix);

            //Calculate derivatives spectrally:
            xderiv(self.ng, self.ng, &self.hrkx, &wkd, &mut wke);
            yderiv(self.ng, self.ng, &self.hrky, &wkd, &mut wkf);
            xderiv(self.ng, self.ng, &self.hrkx, &wkc, &mut wkd);
            yderiv(self.ng, self.ng, &self.hrky, &wkc, &mut wkg);

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
            spctop(
                self.ng,
                self.ng,
                &mut wka,
                &mut wkc,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wkc_matrix = slice_to_2d(&wkc, self.ng, self.ng);
            let mut r_matrix = slice_to_3d(&r, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    r_matrix[i][j][iz] = wkc_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&r_matrix).iter().enumerate() {
                r[i] = *e;
            }

            spctop(
                self.ng,
                self.ng,
                &mut wkb,
                &mut wkd,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wkd_matrix = slice_to_2d(&wkd, self.ng, self.ng);
            let mut zeta_matrix = slice_to_3d(&zeta, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    zeta_matrix[i][j][iz] = wkd_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&zeta_matrix).iter().enumerate() {
                zeta[i] = *e;
            }

            spctop(
                self.ng,
                self.ng,
                &mut wke,
                &mut wka,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wka_matrix = slice_to_2d(&wka, self.ng, self.ng);
            let mut u_matrix = slice_to_3d(&u, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    u_matrix[i][j][iz] = wka_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&u_matrix).iter().enumerate() {
                u[i] = *e;
            }

            spctop(
                self.ng,
                self.ng,
                &mut wkf,
                &mut wkb,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wkb_matrix = slice_to_2d(&wkb, self.ng, self.ng);
            let mut v_matrix = slice_to_3d(&v, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    v_matrix[i][j][iz] = wkb_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&v_matrix).iter().enumerate() {
                v[i] = *e;
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

        ptospc(
            self.ng,
            self.ng,
            &mut wkb,
            &mut wka,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        //Get derivatives of aa:
        xderiv(self.ng, self.ng, &self.hrkx, &wka, &mut wkb);
        spctop(
            self.ng,
            self.ng,
            &mut wkb,
            &mut ax,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        yderiv(self.ng, self.ng, &self.hrky, &wka, &mut wkb);
        spctop(
            self.ng,
            self.ng,
            &mut wkb,
            &mut ay,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = bb[i];
        }

        ptospc(
            self.ng,
            self.ng,
            &mut wkb,
            &mut wka,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        //Get derivatives of bb:
        xderiv(self.ng, self.ng, &self.hrkx, &wka, &mut wkb);
        spctop(
            self.ng,
            self.ng,
            &mut wkb,
            &mut bx,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        yderiv(self.ng, self.ng, &self.hrky, &wka, &mut wkb);
        spctop(
            self.ng,
            self.ng,
            &mut wkb,
            &mut by,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = ax[i] * by[i] - ay[i] * bx[i];
        }
        ptospc(
            self.ng,
            self.ng,
            &mut wkb,
            cs,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
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

        ptospc(
            self.ng,
            self.ng,
            &mut wkp,
            &mut wka,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        xderiv(self.ng, self.ng, &self.hrkx, &wka, &mut wkb);

        for (i, e) in bb.iter().enumerate() {
            wkp[i] = *e;
        }

        ptospc(
            self.ng,
            self.ng,
            &mut wkp,
            &mut wka,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
        yderiv(self.ng, self.ng, &self.hrky, &wka, cs);

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
            let mut wkp_matrix = slice_to_2d(&wkp, self.ng, self.ng);
            let fp_matrix = slice_to_3d(&fp, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkp_matrix[i][j] = fp_matrix[i][j][iz];
                }
            }
            wkp = _2d_to_vec(&wkp_matrix);

            ptospc(
                self.ng,
                self.ng,
                &mut wkp,
                &mut wks,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wks_matrix = slice_to_2d(&wks, self.ng, self.ng);
            let mut fs_matrix = slice_to_3d(&fs, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    fs_matrix[i][j][iz] = wks_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&fs_matrix).iter().enumerate() {
                fs[i] = *e;
            }
        }
    }

    /// Transforms a spectral 3d field fs to physical space (horizontally)
    /// as the array fp.
    pub fn spctop3d(&self, fs: &[f64], fp: &mut [f64], izbeg: usize, izend: usize) {
        let mut wks = vec![0.0; self.ng * self.ng];
        let mut wkp = vec![0.0; self.ng * self.ng];

        for iz in izbeg..=izend {
            let mut wks_matrix = slice_to_2d(&wks, self.ng, self.ng);
            let fs_matrix = slice_to_3d(&fs, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wks_matrix[i][j] = fs_matrix[i][j][iz];
                }
            }
            wks = _2d_to_vec(&wks_matrix);

            spctop(
                self.ng,
                self.ng,
                &mut wks,
                &mut wkp,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            let wkp_matrix = slice_to_2d(&wkp, self.ng, self.ng);
            let mut fp_matrix = slice_to_3d(&fp, self.ng, self.ng, self.nz + 1);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    fp_matrix[i][j][iz] = wkp_matrix[i][j];
                }
            }
            for (i, e) in _3d_to_vec(&fp_matrix).iter().enumerate() {
                fp[i] = *e;
            }
        }
    }

    /// Filters (horizontally) a physical 3d field fp (overwrites fp)
    pub fn deal3d(&self, fp: &mut [f64]) {
        let mut fp_matrix = slice_to_3d(fp, self.ng, self.ng, self.nz + 1);

        let mut wkp_matrix = vec![vec![0.0; self.ng]; self.ng];
        let mut wks_matrix = vec![vec![0.0; self.ng]; self.ng];

        for iz in 0..=self.nz {
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wkp_matrix[i][j] = fp_matrix[i][j][iz];
                }
            }

            let mut wkp = _2d_to_vec(wkp_matrix.as_slice());
            let mut wks = _2d_to_vec(wks_matrix.as_slice());

            ptospc(
                self.ng,
                self.ng,
                &mut wkp,
                &mut wks,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            wks_matrix = slice_to_2d(&wks, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    wks_matrix[i][j] *= self.filt[i][j];
                }
            }
            wks = _2d_to_vec(wks_matrix.as_slice());

            spctop(
                self.ng,
                self.ng,
                &mut wks,
                &mut wkp,
                &self.xfactors,
                &self.yfactors,
                &self.xtrig,
                &self.ytrig,
            );

            wkp_matrix = slice_to_2d(&wkp, self.ng, self.ng);
            for i in 0..self.ng {
                for j in 0..self.ng {
                    fp_matrix[i][j][iz] = wkp_matrix[i][j];
                }
            }
        }

        for (i, e) in _3d_to_vec(&fp_matrix).iter().enumerate() {
            fp[i] = *e;
        }
    }

    /// Filters (horizontally) a physical 2d field fp (overwrites fp)
    pub fn deal2d(&self, fp: &mut [f64]) {
        let mut fs = vec![0.0; self.ng * self.ng];

        ptospc(
            self.ng,
            self.ng,
            fp,
            fs.as_mut_slice(),
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );

        let mut fs_matrix = slice_to_2d(&fs, self.ng, self.ng);
        for i in 0..self.ng {
            for j in 0..self.ng {
                fs_matrix[i][j] *= self.filt[i][j]
            }
        }

        for (i, e) in _2d_to_vec(&fs_matrix).iter().enumerate() {
            fs[i] = *e;
        }

        spctop(
            self.ng,
            self.ng,
            fs.as_mut_slice(),
            fp,
            &self.xfactors,
            &self.yfactors,
            &self.xtrig,
            &self.ytrig,
        );
    }

    /// Computes the 1d spectrum of a spectral field ss and returns the
    /// result in spec
    pub fn spec1d(&self, ss: &[f64], spec: &mut [f64]) {
        assert_eq!(self.ng * self.ng, ss.len());
        assert_eq!(self.ng + 1, spec.len());

        let ss = slice_to_2d(ss, self.ng, self.ng);

        for k in 0..=self.kmax {
            spec[k] = 0.0;
        }

        // x and y-independent mode:
        let k = self.kmag[0][0];
        spec[k] += (1.0 / 4.0) * ss[0][0].powf(2.0);

        // y-independent mode:
        for kx in 1..self.ng {
            let k = self.kmag[kx][0];
            spec[k] += (1.0 / 2.0) * ss[kx][0].powf(2.0)
        }

        // x-independent mode:
        for ky in 1..self.ng {
            let k = self.kmag[0][ky];
            spec[k] += (1.0 / 2.0) * ss[0][ky].powf(2.0)
        }

        // All other modes:
        for ky in 1..self.ng {
            for kx in 1..self.ng {
                let k = self.kmag[kx][ky];
                spec[k] += ss[kx][ky].powf(2.0)
            }
        }
    }
}
