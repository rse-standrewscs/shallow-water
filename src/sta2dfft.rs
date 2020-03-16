//! This module performs FFTs in two directions on two dimensional arrays using
//! the stafft library module to actually compute the FFTs. If FFTs in one
//! direction only are required use the stafft module directly. The module can
//! compute any combination of sine, cosine and full FFTs in each direction.
//! Along with the usual forwards (physical -> Fourier space) and reverse
//! (Fourier space -> physical) routines there are also routines for computing
//! the first derivatives in either direction.
//!
//! The convention is that for each direction the array is dimensioned 1:nx or
//! 1:ny for either the sine or full transforms. While the cosine transforms
//! require the additional endpoint so 0:nx or 0:ny.

use {
    crate::{
        stafft::{forfft, initfft, revfft},
        utils::*,
    },
    core::f64::consts::PI,
    log::error,
};

#[derive(Debug, PartialEq, Clone)]
pub struct D2FFT {
    pub nx: usize,
    pub ny: usize,
    pub xfactors: [usize; 5],
    pub yfactors: [usize; 5],
    pub xtrig: Vec<f64>,
    pub ytrig: Vec<f64>,
}

impl D2FFT {
    /// This subroutine performs the initialisation work for all subsequent
    /// transform and derivative routines.
    /// It calls the initfft() routine from the supproting 1d FFT module for
    /// transforms in both x and y directions.
    /// The routine then defines the two wavenumber arrays, one in each direction.
    pub fn new(nx: usize, ny: usize, lx: f64, ly: f64, kx: &mut [f64], ky: &mut [f64]) -> Self {
        let mut new = Self {
            nx,
            ny,
            xfactors: [0; 5],
            yfactors: [0; 5],
            xtrig: vec![0.0; nx * 2],
            ytrig: vec![0.0; ny * 2],
        };

        // The routines contained in this module are:
        //
        // init2dfft(nx,ny,lx,ly,xfactors,yfactors,xtrig,ytrig,kx,ky)
        //          This routine initialises all the arrays needed for further
        //          transforms. The integers nx and ny are the array dimensions. Then
        //          lx and ly are the domain lengths - these are needed for the correct
        //          scaling when computing derivatives. The arrays xfactors, yfactors,
        //          xtrig and ytrig are needed to perform the various FFTs by the stafft
        //          module (see there for further details. kx and ky are arrays to hold
        //          the wavenumbers associated with each mode in the domain, and are
        //          used in computing derivatives.
        //
        //          If it is known at initialisation that no derivatives are required
        //          it is possible just to pass 1.d0 for each of lx and ly, along with
        //          dummy arrays for kx and ky since these are only needed for
        //          computing the derviatives.

        initfft(nx, &mut new.xfactors, &mut new.xtrig);
        initfft(ny, &mut new.yfactors, &mut new.ytrig);

        if lx != 0.0 {
            let sc = PI / lx;
            for k in 1..=nx {
                kx[k - 1] = sc * k as f64;
            }
        } else {
            error!("Wavenumber array definition not possible, domain length in x equal to zero not allowed!");
            quit::with_code(1);
        }

        if ly != 0.0 {
            let sc = PI / ly;
            for k in 1..=ny {
                ky[k - 1] = sc * k as f64;
            }
        }

        new
    }

    /// Performs a physical -> spectral transform of a variable
    /// rvar(ny,nx) periodic in x and y, and returns the result
    /// (transposed) in svar(nx,ny).
    /// *** Note rvar is destroyed on return. ***
    pub fn ptospc(&self, rvar: &mut [f64], svar: &mut [f64]) {
        let nx = self.nx;
        let ny = self.ny;

        forfft(self.ny, self.nx, rvar, &self.xtrig, &self.xfactors);

        let rvar_nd = viewmut2d(rvar, ny, nx);
        let mut svar_nd = viewmut2d(svar, nx, ny);

        for kx in 0..nx {
            for iy in 0..ny {
                svar_nd[[kx, iy]] = rvar_nd[[iy, kx]];
            }
        }

        forfft(nx, ny, svar, &self.ytrig, &self.yfactors);
    }

    /// Performs a spectral -> physical transform of a variable
    /// svar(nx,ny) periodic in x and y and returns the result
    /// (transposed) in rvar(ny,nx).
    /// *** Note svar is destroyed on return. ***
    pub fn spctop(&self, svar: &mut [f64], rvar: &mut [f64]) {
        let nx = self.nx;
        let ny = self.ny;

        revfft(nx, ny, svar, &self.ytrig, &self.yfactors);

        let svar_nd = viewmut2d(svar, nx, ny);
        let mut rvar_nd = viewmut2d(rvar, ny, nx);

        for kx in 0..nx {
            for iy in 0..ny {
                rvar_nd[[iy, kx]] = svar_nd[[kx, iy]];
            }
        }

        revfft(ny, nx, rvar, &self.xtrig, &self.xfactors);
    }

    /// Computes der = d(var)/dx, spectrally, for a variable
    /// var(nx,ny) periodic in x and y.
    /// *** both var and der are spectral ***
    pub fn xderiv(&self, rkx: &[f64], var: &[f64], der: &mut [f64]) {
        let nx = self.nx;
        let ny = self.ny;
        let var = view2d(var, nx, ny);
        let mut der = viewmut2d(der, nx, ny);

        let mut dkx: usize;
        let mut kxc: usize;

        let nwx = nx / 2;
        let nxp2 = nx + 2;

        // Carry out differentiation by wavenumber multiplication:
        for ky in 0..ny {
            der[[0, ky]] = 0.0;
            for kx in 2..=nx - nwx {
                dkx = 2 * (kx - 1);
                kxc = nxp2 - kx;
                der[[kx - 1, ky]] = -rkx[dkx - 1] * var[[kxc - 1, ky]];
                der[[kxc - 1, ky]] = rkx[dkx - 1] * var[[kx - 1, ky]];
            }
        }

        if nx % 2 == 0 {
            kxc = nwx + 1;
            for ky in 0..ny {
                der[[kxc - 1, ky]] = 0.0;
            }
        }
    }

    /// Computes der = d(var)/dy, spectrally, for a variable
    /// var(nx,ny) periodic in x and y.
    /// *** both var and der are spectral ***
    pub fn yderiv(&self, rky: &[f64], var: &[f64], der: &mut [f64]) {
        let nx = self.nx;
        let ny = self.ny;
        let var = view2d(var, nx, ny);
        let mut der = viewmut2d(der, nx, ny);

        let mut kyc: usize;
        let mut fac: f64;

        let nwy = ny / 2;
        let nyp2 = ny + 2;
        // Carry out differentiation by wavenumber multiplication:
        for kx in 0..nx {
            der[[kx, 0]] = 0.0;
        }

        for ky in 2..=ny - nwy {
            kyc = nyp2 - ky;
            fac = rky[2 * (ky - 1) - 1];
            for kx in 0..nx {
                der[[kx, ky - 1]] = -fac * var[[kx, kyc - 1]];
                der[[kx, kyc - 1]] = fac * var[[kx, ky - 1]];
            }
        }

        if ny % 2 == 0 {
            kyc = nwy + 1;
            for kx in 0..nx {
                der[[kx, kyc - 1]] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        approx::assert_abs_diff_eq,
        byteorder::{ByteOrder, NetworkEndian},
        insta::assert_debug_snapshot,
    };

    #[test]
    fn init2dfft_ng30() {
        let nx = 30;
        let ny = 30;
        let lx = 6.283_185_307_179_586;
        let ly = 6.283_185_307_179_586;

        let mut kx = [0.0; 30];
        let mut ky = [0.0; 30];

        let d2fft = D2FFT::new(nx, ny, lx, ly, &mut kx, &mut ky);

        for (i, e) in include_bytes!("testdata/init2dfft/ng30_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .enumerate()
        {
            assert_abs_diff_eq!(e, d2fft.xtrig[i]);
            assert_abs_diff_eq!(e, d2fft.ytrig[i]);
        }

        assert_debug_snapshot!(d2fft.xfactors);
        assert_debug_snapshot!(d2fft.yfactors);
        assert_debug_snapshot!(kx);
        assert_debug_snapshot!(ky);
    }

    #[test]
    fn init2dfft_ng120() {
        let nx = 120;
        let ny = 120;
        let lx = 6.283_185_307_179_586;
        let ly = 6.283_185_307_179_586;

        let mut kx = [0.0; 120];
        let mut ky = [0.0; 120];

        let d2fft = D2FFT::new(nx, ny, lx, ly, &mut kx, &mut ky);

        for (i, e) in include_bytes!("testdata/init2dfft/ng120_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .enumerate()
        {
            assert_abs_diff_eq!(e, d2fft.xtrig[i]);
            assert_abs_diff_eq!(e, d2fft.ytrig[i]);
        }

        assert_debug_snapshot!(d2fft.xfactors);
        assert_debug_snapshot!(d2fft.yfactors);
        assert_debug_snapshot!(&kx[..]);
        assert_debug_snapshot!(&ky[..]);
    }

    #[test]
    fn ptospc_ng32() {
        let nx = 32;
        let ny = 32;
        let mut rvar = include_bytes!("testdata/ptospc/ptospc_ng32_rvar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut svar = include_bytes!("testdata/ptospc/ptospc_ng32_svar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let rvar2 = include_bytes!("testdata/ptospc/ptospc_ng32_rvar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let svar2 = include_bytes!("testdata/ptospc/ptospc_ng32_svar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let xfactors = [0, 2, 1, 0, 0];
        let yfactors = [0, 2, 1, 0, 0];
        let xtrig = include_bytes!("testdata/ptospc/ptospc_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let ytrig = include_bytes!("testdata/ptospc/ptospc_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2fft = D2FFT {
            nx,
            ny,
            xfactors,
            yfactors,
            xtrig,
            ytrig,
        };
        d2fft.ptospc(&mut rvar, &mut svar);

        assert_eq!(rvar2, rvar);
        assert_eq!(svar2, svar);
    }

    #[test]
    fn spctop_ng32() {
        let nx = 32;
        let ny = 32;
        let mut rvar = include_bytes!("testdata/spctop/spctop_ng32_rvar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut svar = include_bytes!("testdata/spctop/spctop_ng32_svar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let rvar2 = include_bytes!("testdata/spctop/spctop_ng32_rvar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let svar2 = include_bytes!("testdata/spctop/spctop_ng32_svar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let xfactors = [0, 2, 1, 0, 0];
        let yfactors = [0, 2, 1, 0, 0];
        let xtrig = include_bytes!("testdata/spctop/spctop_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let ytrig = include_bytes!("testdata/spctop/spctop_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2fft = D2FFT {
            nx,
            ny,
            xfactors,
            yfactors,
            xtrig,
            ytrig,
        };
        d2fft.spctop(&mut svar, &mut rvar);

        assert_eq!(rvar2, rvar);
        assert_eq!(svar2, svar);
    }

    #[test]
    fn xderiv_1() {
        let nx = 32;
        let ny = 32;
        let rkx = include_bytes!("testdata/deriv/xderiv_1_rkx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("testdata/deriv/xderiv_1_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("testdata/deriv/xderiv_1_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("testdata/deriv/xderiv_1_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.xderiv(&rkx, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn xderiv_2() {
        let nx = 24;
        let ny = 24;
        let rkx = include_bytes!("testdata/deriv/xderiv_2_rkx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("testdata/deriv/xderiv_2_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("testdata/deriv/xderiv_2_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("testdata/deriv/xderiv_2_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.xderiv(&rkx, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn yderiv_1() {
        let nx = 32;
        let ny = 32;
        let rky = include_bytes!("testdata/deriv/yderiv_1_rky.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("testdata/deriv/yderiv_1_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("testdata/deriv/yderiv_1_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("testdata/deriv/yderiv_1_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.yderiv(&rky, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn yderiv_2() {
        let nx = 24;
        let ny = 24;
        let rky = include_bytes!("testdata/deriv/yderiv_2_rky.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("testdata/deriv/yderiv_2_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("testdata/deriv/yderiv_2_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("testdata/deriv/yderiv_2_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.yderiv(&rky, &var, &mut der);

        assert_eq!(der2, der);
    }
}
