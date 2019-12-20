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
        utils::{_2d_to_vec, slice_to_2d},
    },
    core::f64::consts::PI,
};

/// This subroutine performs the initialisation work for all subsequent
/// transform and derivative routines.
/// It calls the initfft() routine from the supproting 1d FFT module for
/// transforms in both x and y directions.
/// The routine then defines the two wavenumber arrays, one in each direction.
pub fn init2dfft(
    nx: usize,
    ny: usize,
    lx: f64,
    ly: f64,
    xfactors: &mut [u8; 5],
    yfactors: &mut [u8; 5],
    xtrig: &mut [f64],
    ytrig: &mut [f64],
    kx: &mut [f64],
    ky: &mut [f64],
) {
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

    initfft(nx, xfactors, xtrig);
    initfft(ny, yfactors, ytrig);

    if lx != 0.0 {
        let sc = PI / lx;
        for k in 1..=nx {
            kx[k - 1] = sc * k as f64;
        }
    } else {
        panic!("Wavenumber array definition not possible, domain length in x equal to zero not allowed!");
    }

    if ly != 0.0 {
        let sc = PI / ly;
        for k in 1..=ny {
            ky[k - 1] = sc * k as f64;
        }
    }
}

/// Performs a physical -> spectral transform of a variable
/// rvar(ny,nx) periodic in x and y, and returns the result
/// (transposed) in svar(nx,ny).
/// *** Note rvar is destroyed on return. ***
pub fn ptospc(
    nx: usize,
    ny: usize,
    rvar: &mut [f64],
    svar: &mut [f64],
    xfactors: &[usize; 5],
    yfactors: &[usize; 5],
    xtrig: &[f64],
    ytrig: &[f64],
) {
    forfft(ny, nx, rvar, xtrig, xfactors);

    let rvar_matrix = slice_to_2d(rvar, ny, nx);
    let mut svar_matrix = slice_to_2d(svar, nx, ny);

    for kx in 0..nx {
        for iy in 0..ny {
            svar_matrix[kx][iy] = rvar_matrix[iy][kx];
        }
    }

    for (i, e) in _2d_to_vec(&svar_matrix).iter().enumerate() {
        svar[i] = *e;
    }

    forfft(nx, ny, svar, ytrig, yfactors);
}

/// Performs a spectral -> physical transform of a variable
/// svar(nx,ny) periodic in x and y and returns the result
/// (transposed) in rvar(ny,nx).
/// *** Note svar is destroyed on return. ***
pub fn spctop(
    nx: usize,
    ny: usize,
    svar: &mut [f64],
    rvar: &mut [f64],
    xfactors: &[usize; 5],
    yfactors: &[usize; 5],
    xtrig: &[f64],
    ytrig: &[f64],
) {
    revfft(nx, ny, svar, ytrig, yfactors);

    let mut rvar_matrix = slice_to_2d(rvar, ny, nx);
    let svar_matrix = slice_to_2d(svar, nx, ny);

    for kx in 0..nx {
        for iy in 0..ny {
            rvar_matrix[iy][kx] = svar_matrix[kx][iy];
        }
    }

    for (i, e) in _2d_to_vec(&rvar_matrix).iter().enumerate() {
        rvar[i] = *e;
    }

    revfft(ny, nx, rvar, xtrig, xfactors);
}

/// Computes der = d(var)/dx, spectrally, for a variable
/// var(nx,ny) periodic in x and y.
/// *** both var and der are spectral ***
pub fn xderiv(nx: usize, ny: usize, rkx: &[f64], var: &[f64], der: &mut [f64]) {
    let var_matrix = slice_to_2d(var, nx, ny);
    let mut der_matrix = slice_to_2d(der, nx, ny);

    let mut dkx: usize;
    let mut kxc: usize;

    let nwx = nx / 2;
    let nxp2 = nx + 2;
    // Carry out differentiation by wavenumber multiplication:
    for ky in 1..=ny {
        der_matrix[0][ky - 1] = 0.0;
        for kx in 2..=nx - nwx {
            dkx = 2 * (kx - 1);
            kxc = nxp2 - kx;
            der_matrix[kx - 1][ky - 1] = -rkx[dkx - 1] * var_matrix[kxc - 1][ky - 1];
            der_matrix[kxc - 1][ky - 1] = rkx[dkx - 1] * var_matrix[kx - 1][ky - 1];
        }
    }

    if nx % 2 == 0 {
        kxc = nwx + 1;
        for ky in 1..=ny {
            der_matrix[kxc - 1][ky - 1] = 0.0;
        }
    }

    for (i, e) in _2d_to_vec(&der_matrix).iter().enumerate() {
        der[i] = *e;
    }
}

/// Computes der = d(var)/dy, spectrally, for a variable
/// var(nx,ny) periodic in x and y.
/// *** both var and der are spectral ***
pub fn yderiv(nx: usize, ny: usize, rky: &[f64], var: &[f64], der: &mut [f64]) {
    let var_matrix = slice_to_2d(var, nx, ny);
    let mut der_matrix = slice_to_2d(der, nx, ny);

    let mut kyc: usize;
    let mut fac: f64;

    let nwy = ny / 2;
    let nyp2 = ny + 2;
    // Carry out differentiation by wavenumber multiplication:
    for kx in 1..=nx {
        der_matrix[kx - 1][0] = 0.0;
    }

    for ky in 2..=ny - nwy {
        kyc = nyp2 - ky;
        fac = rky[2 * (ky - 1) - 1];
        for kx in 1..=nx {
            der_matrix[kx - 1][ky - 1] = -fac * var_matrix[kx - 1][kyc - 1];
            der_matrix[kx - 1][kyc - 1] = fac * var_matrix[kx - 1][ky - 1];
        }
    }

    if ny % 2 == 0 {
        kyc = nwy + 1;
        for kx in 1..=nx {
            der_matrix[kx - 1][kyc - 1] = 0.0;
        }
    }

    for (i, e) in _2d_to_vec(&der_matrix).iter().enumerate() {
        der[i] = *e;
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
    fn init2dfft_snapshot() {
        let nx = 32;
        let ny = 32;
        let lx = 6.283_185_307_179_586;
        let ly = 6.283_185_307_179_586;

        let mut xfactors = [0; 5];
        let mut yfactors = [0; 5];
        let mut xtrig = [0.0; 64];
        let mut ytrig = [0.0; 64];
        let mut kx = [0.0; 32];
        let mut ky = [0.0; 32];

        init2dfft(
            nx,
            ny,
            lx,
            ly,
            &mut xfactors,
            &mut yfactors,
            &mut xtrig,
            &mut ytrig,
            &mut kx,
            &mut ky,
        );

        for (i, e) in include_bytes!("../../testdata/init2dfft_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .enumerate()
        {
            assert_abs_diff_eq!(e, xtrig[i]);
            assert_abs_diff_eq!(e, ytrig[i]);
        }

        assert_debug_snapshot!(xfactors);
        assert_debug_snapshot!(yfactors);
        assert_debug_snapshot!(kx);
        assert_debug_snapshot!(ky);
    }

    #[test]
    fn ptospc_ng32() {
        let nx = 32;
        let ny = 32;
        let mut rvar = include_bytes!("../../testdata/ptospc_ng32_rvar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut svar = include_bytes!("../../testdata/ptospc_ng32_svar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let rvar2 = include_bytes!("../../testdata/ptospc_ng32_rvar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let svar2 = include_bytes!("../../testdata/ptospc_ng32_svar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let xfactors = [0, 2, 1, 0, 0];
        let yfactors = [0, 2, 1, 0, 0];
        let xtrig = include_bytes!("../../testdata/ptospc_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let ytrig = include_bytes!("../../testdata/ptospc_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        ptospc(
            nx, ny, &mut rvar, &mut svar, &xfactors, &yfactors, &xtrig, &ytrig,
        );

        assert_eq!(rvar2, rvar);
        assert_eq!(svar2, svar);
    }

    #[test]
    fn spctop_ng32() {
        let nx = 32;
        let ny = 32;
        let mut rvar = include_bytes!("../../testdata/spctop_ng32_rvar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut svar = include_bytes!("../../testdata/spctop_ng32_svar.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let rvar2 = include_bytes!("../../testdata/spctop_ng32_rvar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let svar2 = include_bytes!("../../testdata/spctop_ng32_svar2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let xfactors = [0, 2, 1, 0, 0];
        let yfactors = [0, 2, 1, 0, 0];
        let xtrig = include_bytes!("../../testdata/spctop_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let ytrig = include_bytes!("../../testdata/spctop_ng32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        spctop(
            nx, ny, &mut svar, &mut rvar, &xfactors, &yfactors, &xtrig, &ytrig,
        );

        assert_eq!(rvar2, rvar);
        assert_eq!(svar2, svar);
    }

    #[test]
    fn xderiv_1() {
        let nx = 32;
        let ny = 32;
        let rkx = include_bytes!("../../testdata/xderiv_1_rkx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("../../testdata/xderiv_1_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("../../testdata/xderiv_1_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("../../testdata/xderiv_1_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        xderiv(nx, ny, &rkx, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn xderiv_2() {
        let nx = 24;
        let ny = 24;
        let rkx = include_bytes!("../../testdata/xderiv_2_rkx.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("../../testdata/xderiv_2_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("../../testdata/xderiv_2_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("../../testdata/xderiv_2_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        xderiv(nx, ny, &rkx, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn yderiv_1() {
        let nx = 32;
        let ny = 32;
        let rky = include_bytes!("../../testdata/yderiv_1_rky.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("../../testdata/yderiv_1_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("../../testdata/yderiv_1_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("../../testdata/yderiv_1_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        yderiv(nx, ny, &rky, &var, &mut der);

        assert_eq!(der2, der);
    }

    #[test]
    fn yderiv_2() {
        let nx = 24;
        let ny = 24;
        let rky = include_bytes!("../../testdata/yderiv_2_rky.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let var = include_bytes!("../../testdata/yderiv_2_var.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut der = include_bytes!("../../testdata/yderiv_2_der.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let der2 = include_bytes!("../../testdata/yderiv_2_der2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        yderiv(nx, ny, &rky, &var, &mut der);

        assert_eq!(der2, der);
    }
}
