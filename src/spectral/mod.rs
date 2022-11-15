//! Module containing subroutines for spectral operations, inversion, etc

#[cfg(test)]
mod test;

use {
    crate::{constants::*, sta2dfft::D2FFT, utils::*},
    core::f64::consts::PI,
    ndarray::{
        Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis, Slice, Zip,
    },
    rayon::prelude::*,
    serde::{Deserialize, Serialize},
    std::sync::{Arc, Mutex},
};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
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

        let mut hlap = arr2zero(ng);
        let mut glap = arr2zero(ng);
        let mut rlap = arr2zero(ng);
        let mut helm = arr2zero(ng);
        let mut c2g2 = arr2zero(ng);
        let mut simp = arr2zero(ng);
        let mut rope = arr2zero(ng);
        let mut fope = arr2zero(ng);
        let mut filt = arr2zero(ng);
        let mut diss = arr2zero(ng);
        let mut opak = arr2zero(ng);
        let mut rdis = arr2zero(ng);

        let mut etdv = Array3::<f64>::zeros((ng, ng, nz));
        let mut htdv = Array3::<f64>::zeros((ng, ng, nz));
        let mut ap = arr2zero(ng);
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
        let kmax: usize;
        let kmaxred: usize;

        let mut a0 = arr2zero(ng);
        let mut a0b = arr2zero(ng);
        let mut apb = arr2zero(ng);

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

        Zip::from(&mut htdv.index_axis_mut(Axis(2), 0))
            .and(&filt)
            .and(&a0b)
            .for_each(|htdv, filt, a0b| *htdv = filt / a0b);
        Zip::from(&mut etdv.index_axis_mut(Axis(2), 0))
            .and(&apb)
            .and(&htdv.index_axis(Axis(2), 0))
            .for_each(|etdv, apb, htdv| *etdv = -apb * htdv);

        for iz in 1..=nz - 2 {
            Zip::from(htdv.index_axis_mut(Axis(2), iz))
                .and(&filt)
                .and(&a0)
                .and(&ap)
                .and(etdv.index_axis(Axis(2), iz - 1))
                .for_each(|htdv, filt, a0, ap, etdv| *htdv = filt / (a0 + ap * etdv));

            Zip::from(etdv.index_axis_mut(Axis(2), iz))
                .and(&ap)
                .and(htdv.index_axis(Axis(2), iz))
                .for_each(|etdv, ap, htdv| *etdv = -ap * htdv);
        }

        Zip::from(htdv.index_axis_mut(Axis(2), nz - 1))
            .and(&filt)
            .and(&a0)
            .and(&ap)
            .and(etdv.index_axis(Axis(2), nz - 2))
            .for_each(|htdv, filt, a0, ap, etdv| *htdv = filt / (a0 + ap * etdv));

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
    pub fn main_invert(
        &self,
        qs: ArrayView3<f64>,
        ds: ArrayView3<f64>,
        gs: ArrayView3<f64>,
        mut r: ArrayViewMut3<f64>,
        mut u: ArrayViewMut3<f64>,
        mut v: ArrayViewMut3<f64>,
        mut zeta: ArrayViewMut3<f64>,
    ) {
        let ng = self.ng;
        let nz = self.nz;
        let dsumi = 1.0 / (ng * ng) as f64;

        let mut es = arr3zero(ng, nz);
        let mut wkh = arr2zero(ng);

        //Define eta = gamma_l/f^2 - q_l/f (spectral):
        Zip::from(&mut es)
            .and(gs)
            .and(qs)
            .for_each(|es, gs, qs| *es = COFI * (COFI * gs - qs));

        //Compute vertical average of eta (store in wkh):
        wkh.fill(0.0);

        for iz in 0..=nz {
            Zip::from(&mut wkh)
                .and(&es.index_axis(Axis(2), iz))
                .for_each(|wkh, es| *wkh += self.weight[iz] * es);
        }

        //Multiply by F = c^2*k^2/(f^2+c^2k^2) in spectral space:
        Zip::from(&mut wkh)
            .and(&self.rope)
            .for_each(|wkh, rope| *wkh *= rope);

        //Initialise mean flow:
        let uio = Arc::new(Mutex::new(0.0));
        let vio = Arc::new(Mutex::new(0.0));

        //Complete inversion:
        (0..=nz as u16)
            .into_par_iter()
            .zip(r.axis_iter_mut(Axis(2)).into_par_iter())
            .zip(zeta.axis_iter_mut(Axis(2)).into_par_iter())
            .zip(u.axis_iter_mut(Axis(2)).into_par_iter())
            .zip(v.axis_iter_mut(Axis(2)).into_par_iter())
            .for_each(|((((iz, mut r), mut zeta), mut u), mut v)| {
                let iz = iz as usize;

                let mut wka = arr2zero(ng);
                let mut wkb = arr2zero(ng);
                let mut wkc = arr2zero(ng);
                let mut wkd = arr2zero(ng);
                let mut wke = arr2zero(ng);
                let mut wkf = arr2zero(ng);
                let mut wkg = arr2zero(ng);

                //Obtain layer thickness anomaly (spectral, in wka):
                Zip::from(&mut wka)
                    .and(&es.index_axis(Axis(2), iz))
                    .and(&wkh)
                    .for_each(|wka, es, wkh| *wka = es - wkh);

                //Obtain relative vorticity (spectral, in wkb):
                //wkb=qs(:,:,iz)+COF*wka;
                Zip::from(&mut wkb)
                    .and(&qs.index_axis(Axis(2), iz))
                    .and(&wka)
                    .for_each(|wkb, qs, wka| *wkb = qs + COF * wka);

                //Invert Laplace operator on zeta & delta to define velocity:
                Zip::from(&mut wkc)
                    .and(&self.rlap)
                    .and(&wkb)
                    .for_each(|wkc, rlap, wkb| *wkc = rlap * wkb);
                Zip::from(&mut wkd)
                    .and(&self.rlap)
                    .and(&ds.index_axis(Axis(2), iz))
                    .for_each(|wkd, rlap, ds| *wkd = rlap * ds);

                //Calculate derivatives spectrally:
                self.d2fft.xderiv(&self.hrkx, wkd.view(), wke.view_mut());
                self.d2fft.yderiv(&self.hrky, wkd.view(), wkf.view_mut());
                self.d2fft.xderiv(&self.hrkx, wkc.view(), wkd.view_mut());
                self.d2fft.yderiv(&self.hrky, wkc.view(), wkg.view_mut());

                //Define velocity components:
                //wke=wke-wkg;
                Zip::from(&mut wke)
                    .and(&wkg)
                    .for_each(|wke, wkg| *wke -= wkg);
                //wkf=wkf+wkd;
                Zip::from(&mut wkf)
                    .and(&wkd)
                    .for_each(|wkf, wkd| *wkf += wkd);

                //Bring quantities back to physical space and store:
                self.d2fft.spctop(wka.view_mut(), r.view_mut());
                self.d2fft.spctop(wkb.view_mut(), zeta.view_mut());
                self.d2fft.spctop(wke.view_mut(), u.view_mut());
                self.d2fft.spctop(wkf.view_mut(), v.view_mut());

                //Accumulate mean flow (uio,vio):
                *uio.lock().unwrap() -= self.weight[iz] * (&u * &r).sum() * dsumi;
                *vio.lock().unwrap() -= self.weight[iz] * (&v * &r).sum() * dsumi;
            });

        //Add mean flow:
        u += *uio.lock().unwrap();
        v += *vio.lock().unwrap();
    }

    /// Computes the (xy) Jacobian of aa and bb and returns it in cs.
    /// aa and bb are in physical space while cs is in spectral space
    ///
    /// NOTE: aa and bb are assumed to be spectrally truncated (de-aliased).
    pub fn jacob(&self, aa: ArrayView2<f64>, bb: ArrayView2<f64>, mut cs: ArrayViewMut2<f64>) {
        let ng = self.ng;

        let mut ax = arr2zero(ng);
        let mut ay = arr2zero(ng);
        let mut bx = arr2zero(ng);
        let mut by = arr2zero(ng);
        let mut wka = arr2zero(ng);
        let mut wkb = aa.to_owned();

        self.d2fft.ptospc(wkb.view_mut(), wka.view_mut());
        //Get derivatives of aa:
        self.d2fft.xderiv(&self.hrkx, wka.view(), wkb.view_mut());
        self.d2fft.spctop(wkb.view_mut(), ax.view_mut());
        self.d2fft.yderiv(&self.hrky, wka.view(), wkb.view_mut());
        self.d2fft.spctop(wkb.view_mut(), ay.view_mut());

        let mut wkb = bb.to_owned();

        self.d2fft.ptospc(wkb.view_mut(), wka.view_mut());
        //Get derivatives of bb:
        self.d2fft.xderiv(&self.hrkx, wka.view(), wkb.view_mut());
        self.d2fft.spctop(wkb.view_mut(), bx.view_mut());
        self.d2fft.yderiv(&self.hrky, wka.view(), wkb.view_mut());
        self.d2fft.spctop(wkb.view_mut(), by.view_mut());

        Zip::from(&mut wkb)
            .and(&ax)
            .and(&ay)
            .and(&bx)
            .and(&by)
            .for_each(|wkb, ax, ay, bx, by| *wkb = ax * by - ay * bx);

        self.d2fft.ptospc(wkb.view_mut(), cs.view_mut());
    }

    /// Computes the divergence of (aa,bb) and returns it in cs.
    /// Both aa and bb in physical space but cs is in spectral space.
    pub fn divs(&self, aa: ArrayView2<f64>, bb: ArrayView2<f64>, mut cs: ArrayViewMut2<f64>) {
        let mut wkp = aa.to_owned();
        let mut wka = arr2zero(self.ng);
        let mut wkb = arr2zero(self.ng);

        self.d2fft.ptospc(wkp.view_mut(), wka.view_mut());
        self.d2fft.xderiv(&self.hrkx, wka.view(), wkb.view_mut());

        let mut wkp = bb.to_owned();

        self.d2fft.ptospc(wkp.view_mut(), wka.view_mut());
        self.d2fft.yderiv(&self.hrky, wka.view(), cs.view_mut());

        cs += &wkb;
    }

    /// Transforms a physical 3d field fp to spectral space (horizontally)
    /// as the array fs.
    pub fn ptospc3d(
        &self,
        fp: ArrayView3<f64>,
        mut fs: ArrayViewMut3<f64>,
        izbeg: usize,
        izend: usize,
    ) {
        (izbeg as u16..=izend as u16)
            .into_par_iter()
            .zip(
                fs.slice_axis_mut(Axis(2), Slice::from(izbeg..=izend))
                    .axis_iter_mut(Axis(2))
                    .into_par_iter(),
            )
            .for_each(|(iz, mut fs)| {
                let mut wkp = fp.index_axis(Axis(2), iz as usize).to_owned();

                self.d2fft.ptospc(wkp.view_mut(), fs.view_mut());
            });
    }

    /// Transforms a spectral 3d field fs to physical space (horizontally)
    /// as the array fp.
    pub fn spctop3d(
        &self,
        fs: ArrayView3<f64>,
        mut fp: ArrayViewMut3<f64>,
        izbeg: usize,
        izend: usize,
    ) {
        (izbeg as u16..=izend as u16)
            .into_par_iter()
            .zip(
                fp.slice_axis_mut(Axis(2), Slice::from(izbeg..=izend))
                    .axis_iter_mut(Axis(2))
                    .into_par_iter(),
            )
            .for_each(|(iz, mut fp)| {
                let mut wks = fs.index_axis(Axis(2), iz as usize).to_owned();

                self.d2fft.spctop(wks.view_mut(), fp.view_mut());
            });
    }

    /// Filters (horizontally) a physical 3d field fp (overwrites fp)
    pub fn deal3d(&self, mut fp: ArrayViewMut3<f64>) {
        let ng = self.ng;

        fp.axis_iter_mut(Axis(2))
            .into_par_iter()
            .for_each(|mut fp| {
                let mut wkp = fp.to_owned();
                let mut wks = arr2zero(ng);

                self.d2fft.ptospc(wkp.view_mut(), wks.view_mut());

                Zip::from(&mut wks)
                    .and(&self.filt)
                    .for_each(|wks, filt| *wks *= filt);

                self.d2fft.spctop(wks.view_mut(), fp.view_mut());
            });
    }

    /// Filters (horizontally) a physical 2d field fp (overwrites fp)
    pub fn deal2d(&self, mut fp: ArrayViewMut2<f64>) {
        let mut fs = arr2zero(self.ng);

        self.d2fft.ptospc(fp.view_mut(), fs.view_mut());

        Zip::from(&mut fs)
            .and(&self.filt)
            .for_each(|fs, filt| *fs *= filt);

        self.d2fft.spctop(fs.view_mut(), fp.view_mut());
    }

    /// Computes the 1d spectrum of a spectral field ss and returns the
    /// result in spec
    pub fn spec1d(&self, ss: &[f64], spec: &mut [f64]) {
        assert_eq!(self.ng * self.ng, ss.len());
        assert_eq!(self.ng + 1, spec.len());

        let ss = view2d(ss, self.ng, self.ng);

        for e in spec.iter_mut().take(self.kmax + 1) {
            *e = 0.0;
        }

        // x and y-independent mode:
        let k = self.kmag[[0, 0]];
        spec[k] += (1.0 / 4.0) * ss[[0, 0]].powf(2.0);

        // y-independent mode:
        for kx in 1..self.ng {
            let k = self.kmag[[kx, 0]];
            spec[k] += (1.0 / 2.0) * ss[[kx, 0]].powf(2.0);
        }

        // x-independent mode:
        for ky in 1..self.ng {
            let k = self.kmag[[0, ky]];
            spec[k] += (1.0 / 2.0) * ss[[0, ky]].powf(2.0);
        }

        // All other modes:
        for ky in 1..self.ng {
            for kx in 1..self.ng {
                let k = self.kmag[[kx, ky]];
                spec[k] += ss[[kx, ky]].powf(2.0);
            }
        }
    }
}
