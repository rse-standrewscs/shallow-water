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
    crate::stafft::{forfft, initfft, revfft},
    core::f64::consts::PI,
    log::error,
    ndarray::{ArrayView2, ArrayViewMut2},
    serde::{Deserialize, Serialize},
};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
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
    pub fn ptospc(&self, mut rvar: ArrayViewMut2<f64>, mut svar: ArrayViewMut2<f64>) {
        let nx = self.nx;
        let ny = self.ny;

        forfft(
            self.ny,
            self.nx,
            rvar.as_slice_memory_order_mut().unwrap(),
            &self.xtrig,
            &self.xfactors,
        );

        rvar.swap_axes(0, 1);
        svar.assign(&rvar);

        forfft(
            nx,
            ny,
            svar.as_slice_memory_order_mut().unwrap(),
            &self.ytrig,
            &self.yfactors,
        );
    }

    /// Performs a spectral -> physical transform of a variable
    /// svar(nx,ny) periodic in x and y and returns the result
    /// (transposed) in rvar(ny,nx).
    /// *** Note svar is destroyed on return. ***
    pub fn spctop(&self, mut svar: ArrayViewMut2<f64>, mut rvar: ArrayViewMut2<f64>) {
        let nx = self.nx;
        let ny = self.ny;

        revfft(
            nx,
            ny,
            svar.as_slice_memory_order_mut().unwrap(),
            &self.ytrig,
            &self.yfactors,
        );

        svar.swap_axes(0, 1);
        rvar.assign(&svar);

        revfft(
            ny,
            nx,
            rvar.as_slice_memory_order_mut().unwrap(),
            &self.xtrig,
            &self.xfactors,
        );
    }

    /// Computes der = d(var)/dx, spectrally, for a variable
    /// var(nx,ny) periodic in x and y.
    /// *** both var and der are spectral ***
    pub fn xderiv(&self, rkx: &[f64], var: ArrayView2<f64>, mut der: ArrayViewMut2<f64>) {
        let nx = self.nx;
        let ny = self.ny;

        let nwx = nx / 2;
        let nxp2 = nx + 2;

        // Carry out differentiation by wavenumber multiplication:
        for ky in 0..ny {
            der[[0, ky]] = 0.0;
            for kx in 2..=nx - nwx {
                let dkx = 2 * (kx - 1);
                let kxc = nxp2 - kx;
                der[[kx - 1, ky]] = -rkx[dkx - 1] * var[[kxc - 1, ky]];
                der[[kxc - 1, ky]] = rkx[dkx - 1] * var[[kx - 1, ky]];
            }
        }

        if nx % 2 == 0 {
            let kxc = nwx + 1;
            for ky in 0..ny {
                der[[kxc - 1, ky]] = 0.0;
            }
        }
    }

    /// Computes der = d(var)/dy, spectrally, for a variable
    /// var(nx,ny) periodic in x and y.
    /// *** both var and der are spectral ***
    pub fn yderiv(&self, rky: &[f64], var: ArrayView2<f64>, mut der: ArrayViewMut2<f64>) {
        let nx = self.nx;
        let ny = self.ny;

        let nwy = ny / 2;
        let nyp2 = ny + 2;

        // Carry out differentiation by wavenumber multiplication:
        for kx in 0..nx {
            der[[kx, 0]] = 0.0;
        }

        for ky in 2..=ny - nwy {
            let kyc = nyp2 - ky;
            let fac = rky[2 * (ky - 1) - 1];
            for kx in 0..nx {
                der[[kx, ky - 1]] = -fac * var[[kx, kyc - 1]];
                der[[kx, kyc - 1]] = fac * var[[kx, ky - 1]];
            }
        }

        if ny % 2 == 0 {
            let kyc = nwy + 1;
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
        crate::array2_from_file,
        approx::assert_abs_diff_eq,
        byteorder::{ByteOrder, NetworkEndian},
        insta::assert_debug_snapshot,
        ndarray::{Array2, ShapeBuilder},
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
        let mut rvar = array2_from_file!(32, 32, "testdata/ptospc/ptospc_ng32_rvar.bin");
        let mut svar = array2_from_file!(32, 32, "testdata/ptospc/ptospc_ng32_svar.bin");
        let rvar2 = array2_from_file!(32, 32, "testdata/ptospc/ptospc_ng32_rvar2.bin");
        let svar2 = array2_from_file!(32, 32, "testdata/ptospc/ptospc_ng32_svar2.bin");
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
        d2fft.ptospc(rvar.view_mut(), svar.view_mut());

        assert_eq!(rvar2, rvar);
        assert_eq!(svar2, svar);
    }

    #[test]
    fn spctop_ng32() {
        let nx = 32;
        let ny = 32;
        let mut rvar = array2_from_file!(32, 32, "testdata/spctop/spctop_ng32_rvar.bin");
        let mut svar = array2_from_file!(32, 32, "testdata/spctop/spctop_ng32_svar.bin");
        let rvar2 = array2_from_file!(32, 32, "testdata/spctop/spctop_ng32_rvar2.bin");
        let svar2 = array2_from_file!(32, 32, "testdata/spctop/spctop_ng32_svar2.bin");
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
        d2fft.spctop(svar.view_mut(), rvar.view_mut());

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
        let var = array2_from_file!(32, 32, "testdata/deriv/xderiv_1_var.bin");
        let mut der = array2_from_file!(32, 32, "testdata/deriv/xderiv_1_der.bin");
        let der2 = array2_from_file!(32, 32, "testdata/deriv/xderiv_1_der2.bin");

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);

        d2dfft.xderiv(&rkx, var.view(), der.view_mut());

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
        let var = array2_from_file!(24, 24, "testdata/deriv/xderiv_2_var.bin");
        let mut der = array2_from_file!(24, 24, "testdata/deriv/xderiv_2_der.bin");
        let der2 = array2_from_file!(24, 24, "testdata/deriv/xderiv_2_der2.bin");

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);

        d2dfft.xderiv(&rkx, var.view(), der.view_mut());

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
        let var = array2_from_file!(32, 32, "testdata/deriv/yderiv_1_var.bin");
        let mut der = array2_from_file!(32, 32, "testdata/deriv/yderiv_1_der.bin");
        let der2 = array2_from_file!(32, 32, "testdata/deriv/yderiv_1_der2.bin");

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.yderiv(&rky, var.view(), der.view_mut());

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
        let var = array2_from_file!(24, 24, "testdata/deriv/yderiv_2_var.bin");
        let mut der = array2_from_file!(24, 24, "testdata/deriv/yderiv_2_der.bin");
        let der2 = array2_from_file!(24, 24, "testdata/deriv/yderiv_2_der2.bin");

        let d2dfft = D2FFT::new(nx, ny, 2.0 * PI, 2.0 * PI, &mut [0.0; 32], &mut [0.0; 32]);
        d2dfft.yderiv(&rky, var.view(), der.view_mut());

        assert_eq!(der2, der);
    }
}
