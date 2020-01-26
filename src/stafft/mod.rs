//! Fourier transform module.
//! This is not a general purpose transform package but is designed to be
//! quick for arrays of length 2^n. It will work if the array length is of
//! the form 2^i * 3^j * 4^k * 5^l * 6^m (integer powers obviously).
//!
//! Minimal error-checking is performed by the code below. The only check is that
//! the initial factorisation can be performed.
//! Therefore if the transforms are called with an array of length <2, or a trig array
//! not matching the length of the array to be transformed the code will fail in a
//! spectacular way (eg. Seg. fault or nonsense returned).
//! It is up to the calling code to ensure everything is called sensibly.
//! The reason for stripping error checking is to speed up the backend by performing
//! less if() evaluations - as errors in practice seem to occur very rarely.
//! So the good news is this should be a fast library - the bad is that you may have to pick
//! around in it if there are failures.
//!
//! To initialise the routines call init(n,factors,trig,ierr).
//! This fills a factorisation array (factors), and a sin/cos array (trig).
//! These must be kept in memory by the calling program.
//! The init routine can be called multiple times with different arrays if more than
//! one length of array is to be transformed.
//! If a factorisation of the array length n cannot be found (as specified above)
//! then the init routine will exit immediately and the integer ierr will be set to 1.
//! If the init returns with ierr=0 then the call was successful.
//!
//! Top-level subroutines contained in this module are:
//! 1) initfft(n,factors,trig)        :
//!      Performs intialisation of the module, by working out the factors of n (the FFT length).
//!      This will fail if n is not factorised completely by 2,3,4,5,6.
//!      The trig array contains the necessary cosine and sine values.
//!      Both arrays passed to init **must** be kept between calls to routines in this module.
//! 2) forfft(m,n,x,trig,factors)  :
//!      This performs a FFT of an array x containing m vectors of length n.
//!      The transform length is n.
//!      This inverse of this transform is obtained by revfft.
//! 3) revfft(m,n,x,trig,factors)  :
//!      This performs an inverse FFT of an array x containing m vectors of length n.
//!      The transform length is n.
//!      This inverse of this transform is forfft.
//! 4) dct(m,n,x,trig,factors)     :
//!      This performs a discrete cosine transform of an array x containing m vectors of length n.
//!      The transform length is n.
//!      This routine calls forfft and performs pre- and post- processing to obtain the transform.
//!      This transform is it's own inverse.
//! 5) dst(m,n,x,trig,factors)     :
//!      This performs a discrete sine transform of an array x containing m vectors of length n.
//!      The transform length is n.
//!      This routine calls forfft and performs pre- and post- processing to obtain the transform.
//!      This transform is it's own inverse.
//!
//! The storage of the transformed array is in 'Hermitian form'. This means that, for the jth vector
//! the values x(j,1:nw) contain the cosine modes of the transform, while the values x(j,nw+1:n) contain
//! the sine modes (in reverse order ie. wave number increasing from n back to nw+1).
//! [Here, for even n, nw=n/2, and for odd n, nw=(n-1)/2].

use {crate::utils::*, core::f64::consts::PI, log::error};

mod forward;
mod reverse;

pub use forward::*;
pub use reverse::*;

/// Subroutine performs initialisation work for all the transforms.
/// It calls routines to factorise the array length n and then sets up
/// a trig array full of sin/cos values used in the transform backend.
pub fn initfft(n: usize, factors: &mut [usize; 5], trig: &mut [f64]) {
    assert_eq!(2 * n, trig.len());

    let fac = [6, 4, 2, 3, 5];
    // First factorise n
    factorisen(n, factors);

    //Define constants needed in trig array definition
    let mut ftwopin = 2.0 * PI / (n as f64);
    let mut rem = n;
    let mut m = 1;
    for (i, element) in factors.iter().enumerate() {
        for _ in 1..=*element {
            rem /= fac[i];
            for k in 1..fac[i] {
                for l in 0..rem {
                    trig[m - 1] = ftwopin * ((k * l) as f64);
                    m += 1;
                }
            }
            ftwopin *= fac[i] as f64;
        }
    }

    for i in 1..n {
        trig[i + n - 1] = -(trig[i - 1].sin());
        trig[i - 1] = trig[i - 1].cos();
    }
}

pub fn factorisen(n: usize, factors: &mut [usize; 5]) {
    let mut rem = n;

    for elem in factors.iter_mut() {
        *elem = 0;
    }

    //Find factors of 6:
    while rem % 6 == 0 {
        factors[0] += 1;
        rem /= 6;
        if rem == 1 {
            return;
        }
    }

    //Find factors of 4:
    while rem % 4 == 0 {
        factors[1] += 1;
        rem /= 4;
        if rem == 1 {
            return;
        }
    }

    //Find factors of 2:
    while rem % 2 == 0 {
        factors[2] += 1;
        rem /= 2;
        if rem == 1 {
            return;
        }
    }
    //Find factors of 3:
    while rem % 3 == 0 {
        factors[3] += 1;
        rem /= 3;
        if rem == 1 {
            return;
        }
    }

    //Find factors of 5:
    while rem % 5 == 0 {
        factors[4] += 1;
        rem /= 5;
        if rem == 1 {
            return;
        }
    }

    error!("Factorization failed");
    std::process::exit(1);
}

/// Main physical to spectral (forward) FFT routine.
/// Performs m transforms of length n in the array x which is dimensioned x(m,n).
/// The arrays trig and factors are filled by the init routine and
/// should be kept from call to call.
/// Backend consists of mixed-radix routines, with 'decimation in time'.
/// Transform is stored in Hermitian form.
pub fn forfft(m: usize, n: usize, xs: &mut [f64], trig: &[f64], factors: &[usize; 5]) {
    assert_eq!(m * n, xs.len());
    assert_eq!(2 * n, trig.len());

    let normfac: f64;
    let mut wk = vec![0.0; m * n];

    let mut rem = n;
    let mut cum = 1;
    let mut iloc: usize;

    let mut orig = true;

    //Use factors of 5:
    for _ in 0..factors[4] {
        rem /= 5;
        iloc = (rem - 1) * 5 * cum;

        let cosine = view2d(&trig[iloc..], cum, 4);
        let sine = view2d(&trig[n + iloc..], cum, 4);

        if orig {
            let a = view3d(xs, m * rem, 5, cum);
            let b = viewmut3d(&mut wk, m * rem, cum, 5);
            forrdx5(a, b, m * rem, cum, cosine, sine);
        } else {
            let a = view3d(&wk, m * rem, 5, cum);
            let b = viewmut3d(xs, m * rem, cum, 5);
            forrdx5(a, b, m * rem, cum, cosine, sine);
        }

        orig = !orig;
        cum *= 5;
    }

    //Use factors of 3:
    for _ in 0..factors[3] {
        rem /= 3;
        iloc = (rem - 1) * 3 * cum;

        let cosine = view2d(&trig[iloc..], cum, 2);
        let sine = view2d(&trig[n + iloc..], cum, 2);

        if orig {
            let a = view3d(xs, m * rem, 3, cum);
            let b = viewmut3d(&mut wk, m * rem, cum, 3);
            forrdx3(a, b, m * rem, cum, cosine, sine);
        } else {
            let a = view3d(&wk, m * rem, 3, cum);
            let b = viewmut3d(xs, m * rem, cum, 3);
            forrdx3(a, b, m * rem, cum, cosine, sine);
        }

        orig = !orig;
        cum *= 3;
    }

    //Use factors of 2:
    for _ in 0..factors[2] {
        rem /= 2;
        iloc = (rem - 1) * 2 * cum;

        let cosine = view2d(&trig[iloc..], cum, 1);
        let sine = view2d(&trig[n + iloc..], cum, 1);

        if orig {
            let a = view3d(xs, m * rem, 2, cum);
            let b = viewmut3d(&mut wk, m * rem, cum, 2);
            forrdx2(a, b, m * rem, cum, cosine, sine);
        } else {
            let a = view3d(&wk, m * rem, 2, cum);
            let b = viewmut3d(xs, m * rem, cum, 2);
            forrdx2(a, b, m * rem, cum, cosine, sine);
        }

        orig = !orig;
        cum *= 2;
    }

    //Use factors of 4:
    for _ in 0..factors[1] {
        rem /= 4;
        iloc = (rem - 1) * 4 * cum;

        let cosine = view2d(&trig[iloc..], cum, 3);
        let sine = view2d(&trig[n + iloc..], cum, 3);

        if orig {
            let a = view3d(xs, m * rem, 4, cum);
            let b = viewmut3d(&mut wk, m * rem, cum, 4);
            forrdx4(a, b, m * rem, cum, cosine, sine);
        } else {
            let a = view3d(&wk, m * rem, 4, cum);
            let b = viewmut3d(xs, m * rem, cum, 4);
            forrdx4(a, b, m * rem, cum, cosine, sine);
        }

        orig = !orig;
        cum *= 4;
    }

    //Use factors of 6:
    for _ in 0..factors[0] {
        rem /= 6;
        iloc = (rem - 1) * 6 * cum;

        let cosine = view2d(&trig[iloc..], cum, 5);
        let sine = view2d(&trig[n + iloc..], cum, 5);

        if orig {
            let a = view3d(xs, m * rem, 6, cum);
            let b = viewmut3d(&mut wk, m * rem, cum, 6);
            forrdx6(a, b, m * rem, cum, cosine, sine);
        } else {
            let a = view3d(&wk, m * rem, 6, cum);
            let b = viewmut3d(xs, m * rem, cum, 6);
            forrdx6(a, b, m * rem, cum, cosine, sine);
        }

        orig = !orig;
        cum *= 6;
    }

    //Multiply by the normalisation constant and put
    //transformed array in the right location:
    normfac = 1.0 / (n as f64).sqrt();
    for (i, x) in xs.iter_mut().enumerate() {
        if orig {
            *x *= normfac;
        } else {
            *x = wk[i] * normfac;
        }
    }
}

/// Main spectral to physical (reverse) FFT routine.
/// Performs m reverse transforms of length n in the array x which is dimensioned x(m,n).
/// The arrays trig and factors are filled by the init routine and
/// should be kept from call to call.
/// Backend consists of mixed-radix routines, with 'decimation in frequency'.
/// Reverse transform starts in Hermitian form.
pub fn revfft(m: usize, n: usize, xs: &mut [f64], trig: &[f64], factors: &[usize; 5]) {
    assert_eq!(m * n, xs.len());
    assert_eq!(2 * n, trig.len());

    let normfac: f64;
    let mut wk = vec![0.0; m * n];

    let mut rem = n;
    let mut cum = 1;
    let mut iloc: usize;

    let mut orig = true;

    for elem in xs.iter_mut().skip((n / 2 + 1) * m) {
        *elem = -*elem;
    }

    //Scale 0 and Nyquist frequencies:
    for elem in xs.iter_mut().take(m) {
        *elem *= 0.5;
    }

    if n % 2 == 0 {
        let k = m * n / 2;
        for i in 0..m {
            xs[k + i] *= 0.5;
        }
    }

    //Use factors of 6:
    for _ in 0..factors[0] {
        rem /= 6;
        iloc = (cum - 1) * 6 * rem;

        let cosine = view2d(&trig[iloc..], rem, 5);
        let sine = view2d(&trig[n + iloc..], rem, 5);

        if orig {
            let a = view3d(xs, m * cum, rem, 6);
            let b = viewmut3d(&mut wk, m * cum, 6, rem);
            revrdx6(a, b, m * cum, rem, cosine, sine);
        } else {
            let a = view3d(&wk, m * cum, rem, 6);
            let b = viewmut3d(xs, m * cum, 6, rem);
            revrdx6(a, b, m * cum, rem, cosine, sine);
        }

        orig = !orig;
        cum *= 6;
    }

    //Use factors of 4:
    for _ in 0..factors[1] {
        rem /= 4;
        iloc = (cum - 1) * 4 * rem;

        let cosine = view2d(&trig[iloc..], rem, 3);
        let sine = view2d(&trig[n + iloc..], rem, 3);

        if orig {
            let a = view3d(xs, m * cum, rem, 4);
            let b = viewmut3d(&mut wk, m * cum, 4, rem);
            revrdx4(a, b, m * cum, rem, cosine, sine);
        } else {
            let a = view3d(&wk, m * cum, rem, 4);
            let b = viewmut3d(xs, m * cum, 4, rem);
            revrdx4(a, b, m * cum, rem, cosine, sine);
        }

        orig = !orig;
        cum *= 4;
    }

    //Use factors of 2:
    for _ in 0..factors[2] {
        rem /= 2;
        iloc = (cum - 1) * 2 * rem;

        let cosine = view2d(&trig[iloc..], rem, 1);
        let sine = view2d(&trig[n + iloc..], rem, 1);

        if orig {
            let a = view3d(xs, m * cum, rem, 2);
            let b = viewmut3d(&mut wk, m * cum, 2, rem);
            revrdx2(a, b, m * cum, rem, cosine, sine);
        } else {
            let a = view3d(&wk, m * cum, rem, 2);
            let b = viewmut3d(xs, m * cum, 2, rem);
            revrdx2(a, b, m * cum, rem, cosine, sine);
        }

        orig = !orig;
        cum *= 2;
    }

    //Use factors of 3:
    for _ in 0..factors[3] {
        rem /= 3;
        iloc = (cum - 1) * 3 * rem;

        let cosine = view2d(&trig[iloc..], rem, 2);
        let sine = view2d(&trig[n + iloc..], rem, 2);

        if orig {
            let a = view3d(xs, m * cum, rem, 3);
            let b = viewmut3d(&mut wk, m * cum, 3, rem);
            revrdx3(a, b, m * cum, rem, cosine, sine);
        } else {
            let a = view3d(&wk, m * cum, rem, 3);
            let b = viewmut3d(xs, m * cum, 3, rem);
            revrdx3(a, b, m * cum, rem, cosine, sine);
        }

        orig = !orig;
        cum *= 3;
    }

    //Use factors of 5:
    for _ in 0..factors[4] {
        rem /= 5;
        iloc = (cum - 1) * 5 * rem;

        let cosine = view2d(&trig[iloc..], rem, 4);
        let sine = view2d(&trig[n + iloc..], rem, 4);

        if orig {
            let a = view3d(xs, m * cum, rem, 5);
            let b = viewmut3d(&mut wk, m * cum, 5, rem);
            revrdx5(a, b, m * cum, rem, cosine, sine);
        } else {
            let a = view3d(&wk, m * cum, rem, 5);
            let b = viewmut3d(xs, m * cum, 5, rem);
            revrdx5(a, b, m * cum, rem, cosine, sine);
        }

        orig = !orig;
        cum *= 5;
    }

    //Multiply by the normalisation constant and put
    //transformed array in the right location:
    normfac = 2.0 / (n as f64).sqrt();
    for (i, x) in xs.iter_mut().enumerate() {
        if orig {
            *x *= normfac;
        } else {
            *x = wk[i] * normfac;
        }
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        byteorder::{ByteOrder, NetworkEndian},
        insta::assert_debug_snapshot,
    };

    #[test]
    fn factorisen_snapshot_1() {
        let n = 16;
        let mut factors = [0; 5];

        factorisen(n, &mut factors);

        assert_debug_snapshot!(factors);
    }

    #[test]
    fn factorisen_snapshot_2() {
        let n = 8640;
        let mut factors = [0; 5];

        factorisen(n, &mut factors);

        assert_debug_snapshot!(factors);
    }

    #[test]
    fn initfft_snapshot_1() {
        let n = 30;
        let mut factors = [0; 5];
        let mut trig = [0.0; 60];
        let trig2 = include_bytes!("testdata/initfft/30_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        initfft(n, &mut factors, &mut trig);

        assert_approx_eq_slice(&trig2, &trig);
    }

    #[test]
    fn initfft_snapshot_2() {
        let n = 32;
        let mut factors = [0; 5];
        let mut trig = [0.0; 64];
        let trig2 = include_bytes!("testdata/initfft/32_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        initfft(n, &mut factors, &mut trig);

        assert_approx_eq_slice(&trig2, &trig);
    }

    #[test]
    fn initfft_snapshot_3() {
        let n = 18;
        let mut factors = [0; 5];
        let mut trig = [0.0; 36];
        let trig2 = include_bytes!("testdata/initfft/18_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        initfft(n, &mut factors, &mut trig);

        assert_approx_eq_slice(&trig2, &trig);
    }

    #[test]
    fn forfft_ng12_1() {
        let m = 12;
        let n = 12;
        let mut x = include_bytes!("testdata/forfft/forfft_ng12_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng12_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng12_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 1, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng12_2() {
        let m = 12;
        let n = 12;
        let mut x = include_bytes!("testdata/forfft/forfft_ng12_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng12_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng12_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 1, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng15_1() {
        let m = 15;
        let n = 15;
        let mut x = include_bytes!("testdata/forfft/forfft_ng15_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng15_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng15_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [0, 0, 0, 1, 1];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng15_2() {
        let m = 15;
        let n = 15;
        let mut x = include_bytes!("testdata/forfft/forfft_ng15_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng15_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng15_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [0, 0, 0, 1, 1];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng16_1() {
        let m = 16;
        let n = 16;
        let mut x = include_bytes!("testdata/forfft/forfft_ng16_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng16_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng16_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [0, 2, 0, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng16_2() {
        let m = 16;
        let n = 16;
        let mut x = include_bytes!("testdata/forfft/forfft_ng16_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng16_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng16_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [0, 2, 0, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng18_1() {
        let m = 18;
        let n = 18;
        let mut x = include_bytes!("testdata/forfft/forfft_ng18_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng18_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng18_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 0, 1, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng18_2() {
        let m = 18;
        let n = 18;
        let mut x = include_bytes!("testdata/forfft/forfft_ng18_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng18_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng18_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 0, 1, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng24_1() {
        let m = 24;
        let n = 24;
        let mut x = include_bytes!("testdata/forfft/forfft_ng24_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng24_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng24_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 1, 0, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn forfft_ng24_2() {
        let m = 24;
        let n = 24;
        let mut x = include_bytes!("testdata/forfft/forfft_ng24_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/forfft/forfft_ng24_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/forfft/forfft_ng24_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 1, 0, 0, 0];

        forfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng12_1() {
        let m = 12;
        let n = 12;
        let mut x = include_bytes!("testdata/revfft/revfft_ng12_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng12_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng12_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 1, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng12_2() {
        let m = 12;
        let n = 12;
        let mut x = include_bytes!("testdata/revfft/revfft_ng12_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng12_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng12_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let factors = [1, 0, 1, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng15_1() {
        let m = 15;
        let n = 15;
        let mut x = include_bytes!("testdata/revfft/revfft_ng15_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng15_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng15_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [0, 0, 0, 1, 1];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng15_2() {
        let m = 15;
        let n = 15;
        let mut x = include_bytes!("testdata/revfft/revfft_ng15_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng15_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng15_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [0, 0, 0, 1, 1];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng16_1() {
        let m = 16;
        let n = 16;
        let mut x = include_bytes!("testdata/revfft/revfft_ng16_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng16_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng16_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [0, 2, 0, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng16_2() {
        let m = 16;
        let n = 16;
        let mut x = include_bytes!("testdata/revfft/revfft_ng16_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng16_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng16_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [0, 2, 0, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng18_1() {
        let m = 18;
        let n = 18;
        let mut x = include_bytes!("testdata/revfft/revfft_ng18_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng18_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng18_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [1, 0, 0, 1, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng18_2() {
        let m = 18;
        let n = 18;
        let mut x = include_bytes!("testdata/revfft/revfft_ng18_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng18_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng18_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [1, 0, 0, 1, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng24_1() {
        let m = 24;
        let n = 24;
        let mut x = include_bytes!("testdata/revfft/revfft_ng24_1_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng24_1_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng24_1_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [1, 1, 0, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }

    #[test]
    fn revfft_ng24_2() {
        let m = 24;
        let n = 24;
        let mut x = include_bytes!("testdata/revfft/revfft_ng24_2_x.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let x2 = include_bytes!("testdata/revfft/revfft_ng24_2_x2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let trig = include_bytes!("testdata/revfft/revfft_ng24_2_trig.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let factors = [1, 1, 0, 0, 0];

        revfft(m, n, &mut x, &trig, &factors);

        assert_eq!(x2, x);
    }
}
