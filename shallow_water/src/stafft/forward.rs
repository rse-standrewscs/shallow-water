//! Physical to spectral (forward) routines
use {
    crate::constants::*,
    core::f64::consts::FRAC_1_SQRT_2,
    ndarray::{ArrayView2, ArrayView3, ArrayViewMut3, Axis},
};

/// Radix six physical to Hermitian FFT with 'decimation in time'.
pub fn forrdx6(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(6, a.len_of(Axis(1)));
    assert_eq!(lv, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(lv, b.len_of(Axis(1)));
    assert_eq!(6, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(5, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(5, sine.len_of(Axis(1)));

    let mut x1p: f64;
    let mut x2p: f64;
    let mut x3p: f64;
    let mut x4p: f64;
    let mut x5p: f64;
    let mut y1p: f64;
    let mut y2p: f64;
    let mut y3p: f64;
    let mut y4p: f64;
    let mut y5p: f64;
    let mut s1k: f64;
    let mut s2k: f64;
    let mut s3k: f64;
    let mut s4k: f64;
    let mut s5k: f64;
    let mut c1k: f64;
    let mut c2k: f64;
    let mut c3k: f64;
    let mut c4k: f64;
    let mut c5k: f64;
    let mut t1i: f64;
    let mut t1r: f64;
    let mut t2i: f64;
    let mut t2r: f64;
    let mut t3i: f64;
    let mut t3r: f64;
    let mut u0i: f64;
    let mut u0r: f64;
    let mut u1i: f64;
    let mut u1r: f64;
    let mut u2i: f64;
    let mut u2r: f64;
    let mut v0i: f64;
    let mut v0r: f64;
    let mut v1i: f64;
    let mut v1r: f64;
    let mut v2i: f64;
    let mut v2r: f64;
    let mut q1: f64;
    let mut q2: f64;
    let mut q3: f64;
    let mut q4: f64;
    let mut q5: f64;
    let mut q6: f64;

    let mut kc: usize;

    // Do k=0 first:
    for i in 0..nv {
        t1r = a[[i, 2, 0]] + a[[i, 4, 0]];
        t2r = a[[i, 0, 0]] - 0.5 * t1r;
        t3r = SINFPI3 * (a[[i, 4, 0]] - a[[i, 2, 0]]);
        u0r = a[[i, 0, 0]] + t1r;
        t1i = a[[i, 5, 0]] + a[[i, 1, 0]];
        t2i = a[[i, 3, 0]] - 0.5 * t1i;
        t3i = SINFPI3 * (a[[i, 5, 0]] - a[[i, 1, 0]]);
        v0r = a[[i, 3, 0]] + t1i;
        b[[i, 0, 0]] = u0r + v0r;
        b[[i, 0, 1]] = t2r - t2i;
        b[[i, 0, 2]] = t2r + t2i;
        b[[i, 0, 3]] = u0r - v0r;
        b[[i, 0, 4]] = t3i - t3r;
        b[[i, 0, 5]] = t3r + t3i;
    }
    // Next do remaining k:
    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                x1p = cosine[[k, 0]] * a[[i, 1, k]] - sine[[k, 0]] * a[[i, 1, kc]];
                y1p = cosine[[k, 0]] * a[[i, 1, kc]] + sine[[k, 0]] * a[[i, 1, k]];
                x2p = cosine[[k, 1]] * a[[i, 2, k]] - sine[[k, 1]] * a[[i, 2, kc]];
                y2p = cosine[[k, 1]] * a[[i, 2, kc]] + sine[[k, 1]] * a[[i, 2, k]];
                x3p = cosine[[k, 2]] * a[[i, 3, k]] - sine[[k, 2]] * a[[i, 3, kc]];
                y3p = cosine[[k, 2]] * a[[i, 3, kc]] + sine[[k, 2]] * a[[i, 3, k]];
                x4p = cosine[[k, 3]] * a[[i, 4, k]] - sine[[k, 3]] * a[[i, 4, kc]];
                y4p = cosine[[k, 3]] * a[[i, 4, kc]] + sine[[k, 3]] * a[[i, 4, k]];
                x5p = cosine[[k, 3]] * a[[i, 5, k]] - sine[[k, 4]] * a[[i, 5, kc]];
                y5p = cosine[[k, 3]] * a[[i, 5, kc]] + sine[[k, 4]] * a[[i, 5, k]];
                t1r = x2p + x4p;
                t1i = y2p + y4p;
                t2r = a[[i, 0, k]] - 0.5 * t1r;
                t2i = a[[i, 0, kc]] - 0.5 * t1i;
                t3r = SINFPI3 * (x2p - x4p);
                t3i = SINFPI3 * (y2p - y4p);
                u0r = a[[i, 0, k]] + t1r;
                u0i = a[[i, 0, kc]] + t1i;
                u1r = t2r + t3i;
                u1i = t2i - t3r;
                u2r = t2r - t3i;
                u2i = t2i + t3r;
                t1r = x5p + x1p;
                t1i = y5p + y1p;
                t2r = x3p - 0.5 * t1r;
                t2i = y3p - 0.5 * t1i;
                t3r = SINFPI3 * (x5p - x1p);
                t3i = SINFPI3 * (y5p - y1p);
                v0r = x3p + t1r;
                v0i = y3p + t1i;
                v1r = t2r + t3i;
                v1i = t3r - t2i;
                v2r = t2r - t3i;
                v2i = t2i + t3r;
                b[[i, k, 0]] = u0r + v0r;
                b[[i, kc, 0]] = u2r - v2r;
                b[[i, k, 1]] = u1r - v1r;
                b[[i, kc, 1]] = u1r + v1r;
                b[[i, k, 2]] = u2r + v2r;
                b[[i, kc, 2]] = u0r - v0r;
                b[[i, k, 3]] = v0i - u0i;
                b[[i, kc, 3]] = u2i + v2i;
                b[[i, k, 4]] = v1i - u1i;
                b[[i, kc, 4]] = u1i + v1i;
                b[[i, k, 5]] = v2i - u2i;
                b[[i, kc, 5]] = u0i + v0i;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[k, 0]];
            s1k = sine[[k, 0]];
            c2k = cosine[[k, 1]];
            s2k = sine[[k, 1]];
            c3k = cosine[[k, 2]];
            s3k = sine[[k, 2]];
            c4k = cosine[[k, 3]];
            s4k = sine[[k, 3]];
            c5k = cosine[[k, 4]];
            s5k = sine[[k, 4]];
            for i in 0..nv {
                x1p = c1k * a[[i, 1, k]] - s1k * a[[i, 1, kc]];
                y1p = c1k * a[[i, 1, kc]] + s1k * a[[i, 1, k]];
                x2p = c2k * a[[i, 2, k]] - s2k * a[[i, 2, kc]];
                y2p = c2k * a[[i, 2, kc]] + s2k * a[[i, 2, k]];
                x3p = c3k * a[[i, 3, k]] - s3k * a[[i, 3, kc]];
                y3p = c3k * a[[i, 3, kc]] + s3k * a[[i, 3, k]];
                x4p = c4k * a[[i, 4, k]] - s4k * a[[i, 4, kc]];
                y4p = c4k * a[[i, 4, kc]] + s4k * a[[i, 4, k]];
                x5p = c5k * a[[i, 5, k]] - s5k * a[[i, 5, kc]];
                y5p = c5k * a[[i, 5, kc]] + s5k * a[[i, 5, k]];
                t1r = x2p + x4p;
                t1i = y2p + y4p;
                t2r = a[[i, 0, k]] - 0.5 * t1r;
                t2i = a[[i, 0, kc]] - 0.5 * t1i;
                t3r = SINFPI3 * (x2p - x4p);
                t3i = SINFPI3 * (y2p - y4p);
                u0r = a[[i, 0, k]] + t1r;
                u0i = a[[i, 0, kc]] + t1i;
                u1r = t2r + t3i;
                u1i = t2i - t3r;
                u2r = t2r - t3i;
                u2i = t2i + t3r;
                t1r = x5p + x1p;
                t1i = y5p + y1p;
                t2r = x3p - 0.5 * t1r;
                t2i = y3p - 0.5 * t1i;
                t3r = SINFPI3 * (x5p - x1p);
                t3i = SINFPI3 * (y5p - y1p);
                v0r = x3p + t1r;
                v0i = y3p + t1i;
                v1r = t2r + t3i;
                v1i = t3r - t2i;
                v2r = t2r - t3i;
                v2i = t2i + t3r;
                b[[i, k, 0]] = u0r + v0r;
                b[[i, kc, 0]] = u2r - v2r;
                b[[i, k, 1]] = u1r - v1r;
                b[[i, kc, 1]] = u1r + v1r;
                b[[i, k, 2]] = u2r + v2r;
                b[[i, kc, 2]] = u0r - v0r;
                b[[i, k, 3]] = v0i - u0i;
                b[[i, kc, 3]] = u2i + v2i;
                b[[i, k, 4]] = v1i - u1i;
                b[[i, kc, 4]] = u1i + v1i;
                b[[i, k, 5]] = v2i - u2i;
                b[[i, kc, 5]] = u0i + v0i;
            }
        }
    }

    //Catch the case k=lv/2 when lv even:
    if lv % 2 == 0 {
        let lvd2 = lv / 2;
        for i in 0..nv {
            q1 = a[[i, 2, lvd2]] - a[[i, 4, lvd2]];
            q2 = a[[i, 0, lvd2]] + 0.5 * q1;
            q3 = SINFPI3 * (a[[i, 2, lvd2]] + a[[i, 4, lvd2]]);
            q4 = a[[i, 1, lvd2]] + a[[i, 5, lvd2]];
            q5 = -a[[i, 3, lvd2]] - 0.5 * q4;
            q6 = SINFPI3 * (a[[i, 1, lvd2]] - a[[i, 5, lvd2]]);
            b[[i, lvd2, 0]] = q2 + q6;
            b[[i, lvd2, 1]] = a[[i, 0, lvd2]] - q1;
            b[[i, lvd2, 2]] = q2 - q6;
            b[[i, lvd2, 3]] = q5 + q3;
            b[[i, lvd2, 4]] = a[[i, 3, lvd2]] - q4;
            b[[i, lvd2, 5]] = q5 - q3;
        }
    }
}

/// Radix five physical to Hermitian FFT with 'decimation in time'.
pub fn forrdx5(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(5, a.len_of(Axis(1)));
    assert_eq!(lv, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(lv, b.len_of(Axis(1)));
    assert_eq!(5, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(4, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(4, sine.len_of(Axis(1)));

    let mut x1p: f64;
    let mut x2p: f64;
    let mut x3p: f64;
    let mut x4p: f64;
    let mut y1p: f64;
    let mut y2p: f64;
    let mut y3p: f64;
    let mut y4p: f64;
    let mut s1k: f64;
    let mut s2k: f64;
    let mut s3k: f64;
    let mut s4k: f64;
    let mut c1k: f64;
    let mut c2k: f64;
    let mut c3k: f64;
    let mut c4k: f64;
    let mut t1i: f64;
    let mut t1r: f64;
    let mut t2i: f64;
    let mut t2r: f64;
    let mut t3i: f64;
    let mut t3r: f64;
    let mut t4i: f64;
    let mut t4r: f64;
    let mut t5i: f64;
    let mut t5r: f64;
    let mut t6i: f64;
    let mut t6r: f64;
    let mut t7i: f64;
    let mut t7r: f64;
    let mut t8i: f64;
    let mut t8r: f64;
    let mut t9i: f64;
    let mut t9r: f64;
    let mut t10i: f64;
    let mut t10r: f64;
    let mut t11i: f64;
    let mut t11r: f64;

    let mut kc: usize;

    //Do k=0 first:
    for i in 0..nv {
        t1r = a[[i, 1, 0]] + a[[i, 4, 0]];
        t2r = a[[i, 2, 0]] + a[[i, 3, 0]];
        t3r = SINF2PI5 * (a[[i, 4, 0]] - a[[i, 1, 0]]);
        t4r = SINF2PI5 * (a[[i, 2, 0]] - a[[i, 3, 0]]);
        t5r = t1r + t2r;
        t6r = RTF516 * (t1r - t2r);
        t7r = a[[i, 0, 0]] - 0.25 * t5r;
        b[[i, 0, 0]] = a[[i, 0, 0]] + t5r;
        b[[i, 0, 1]] = t7r + t6r;
        b[[i, 0, 2]] = t7r - t6r;
        b[[i, 0, 3]] = t4r + SINRAT * t3r;
        b[[i, 0, 4]] = t3r - SINRAT * t4r;
    }
    //Next do remaining k:
    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                x1p = cosine[[k, 0]] * a[[i, 1, k]] - sine[[k, 0]] * a[[i, 1, kc]];
                y1p = cosine[[k, 0]] * a[[i, 1, kc]] + sine[[k, 0]] * a[[i, 1, k]];
                x2p = cosine[[k, 1]] * a[[i, 2, k]] - sine[[k, 1]] * a[[i, 2, kc]];
                y2p = cosine[[k, 1]] * a[[i, 2, kc]] + sine[[k, 1]] * a[[i, 2, k]];
                x3p = cosine[[k, 2]] * a[[i, 3, k]] - sine[[k, 2]] * a[[i, 3, kc]];
                y3p = cosine[[k, 2]] * a[[i, 3, kc]] + sine[[k, 2]] * a[[i, 3, k]];
                x4p = cosine[[k, 3]] * a[[i, 4, k]] - sine[[k, 3]] * a[[i, 4, kc]];
                y4p = cosine[[k, 3]] * a[[i, 4, kc]] + sine[[k, 3]] * a[[i, 4, k]];
                t1r = x1p + x4p;
                t1i = y1p + y4p;
                t2r = x2p + x3p;
                t2i = y2p + y3p;
                t3r = SINF2PI5 * (x1p - x4p);
                t3i = SINF2PI5 * (y1p - y4p);
                t4r = SINF2PI5 * (x2p - x3p);
                t4i = SINF2PI5 * (y2p - y3p);
                t5r = t1r + t2r;
                t5i = t1i + t2i;
                t6r = RTF516 * (t1r - t2r);
                t6i = RTF516 * (t1i - t2i);
                t7r = a[[i, 0, k]] - 0.25 * t5r;
                t7i = a[[i, 0, kc]] - 0.25 * t5i;
                t8r = t7r + t6r;
                t8i = t7i + t6i;
                t9r = t7r - t6r;
                t9i = t7i - t6i;
                t10r = t3r + SINRAT * t4r;
                t10i = t3i + SINRAT * t4i;
                t11r = t4r - SINRAT * t3r;
                t11i = SINRAT * t3i - t4i;
                b[[i, k, 0]] = a[[i, 0, k]] + t5r;
                b[[i, kc, 0]] = t8r - t10i;
                b[[i, k, 1]] = t8r + t10i;
                b[[i, kc, 1]] = t9r - t11i;
                b[[i, k, 2]] = t9r + t11i;
                b[[i, kc, 2]] = t9i + t11r;
                b[[i, k, 3]] = t11r - t9i;
                b[[i, kc, 3]] = t8i - t10r;
                b[[i, k, 4]] = -t8i - t10r;
                b[[i, kc, 4]] = a[[i, 0, kc]] + t5i;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[k, 0]];
            s1k = sine[[k, 0]];
            c2k = cosine[[k, 1]];
            s2k = sine[[k, 1]];
            c3k = cosine[[k, 2]];
            s3k = sine[[k, 2]];
            c4k = cosine[[k, 3]];
            s4k = sine[[k, 3]];
            for i in 0..nv {
                x1p = c1k * a[[i, 1, k]] - s1k * a[[i, 1, kc]];
                y1p = c1k * a[[i, 1, kc]] + s1k * a[[i, 1, k]];
                x2p = c2k * a[[i, 2, k]] - s2k * a[[i, 2, kc]];
                y2p = c2k * a[[i, 2, kc]] + s2k * a[[i, 2, k]];
                x3p = c3k * a[[i, 3, k]] - s3k * a[[i, 3, kc]];
                y3p = c3k * a[[i, 3, kc]] + s3k * a[[i, 3, k]];
                x4p = c4k * a[[i, 4, k]] - s4k * a[[i, 4, kc]];
                y4p = c4k * a[[i, 4, kc]] + s4k * a[[i, 4, k]];
                t1r = x1p + x4p;
                t1i = y1p + y4p;
                t2r = x2p + x3p;
                t2i = y2p + y3p;
                t3r = SINF2PI5 * (x1p - x4p);
                t3i = SINF2PI5 * (y1p - y4p);
                t4r = SINF2PI5 * (x2p - x3p);
                t4i = SINF2PI5 * (y2p - y3p);
                t5r = t1r + t2r;
                t5i = t1i + t2i;
                t6r = RTF516 * (t1r - t2r);
                t6i = RTF516 * (t1i - t2i);
                t7r = a[[i, 0, k]] - 0.25 * t5r;
                t7i = a[[i, 0, kc]] - 0.25 * t5i;
                t8r = t7r + t6r;
                t8i = t7i + t6i;
                t9r = t7r - t6r;
                t9i = t7i - t6i;
                t10r = t3r + SINRAT * t4r;
                t10i = t3i + SINRAT * t4i;
                t11r = t4r - SINRAT * t3r;
                t11i = SINRAT * t3i - t4i;
                b[[i, k, 0]] = a[[i, 0, k]] + t5r;
                b[[i, kc, 0]] = t8r - t10i;
                b[[i, k, 1]] = t8r + t10i;
                b[[i, kc, 1]] = t9r - t11i;
                b[[i, k, 2]] = t9r + t11i;
                b[[i, kc, 2]] = t9i + t11r;
                b[[i, k, 3]] = t11r - t9i;
                b[[i, kc, 3]] = t8i - t10r;
                b[[i, k, 4]] = -t8i - t10r;
                b[[i, kc, 4]] = a[[i, 0, kc]] + t5i;
            }
        }
    }
}

/// Radix four physical to Hermitian FFT with 'decimation in time'.
pub fn forrdx4(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(4, a.len_of(Axis(1)));
    assert_eq!(lv, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(lv, b.len_of(Axis(1)));
    assert_eq!(4, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(3, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(3, sine.len_of(Axis(1)));

    let mut x1p: f64;
    let mut x2p: f64;
    let mut x3p: f64;
    let mut y1p: f64;
    let mut y2p: f64;
    let mut y3p: f64;
    let mut s1k: f64;
    let mut s2k: f64;
    let mut s3k: f64;
    let mut c1k: f64;
    let mut c2k: f64;
    let mut c3k: f64;
    let mut t1i: f64;
    let mut t1r: f64;
    let mut t2i: f64;
    let mut t2r: f64;
    let mut t3i: f64;
    let mut t3r: f64;
    let mut t4i: f64;
    let mut t4r: f64;
    let mut q1: f64;
    let mut q2: f64;

    let mut kc: usize;

    //Do k=0 first:
    for i in 0..nv {
        t1r = a[[i, 0, 0]] + a[[i, 2, 0]];
        t2r = a[[i, 1, 0]] + a[[i, 3, 0]];
        b[[i, 0, 0]] = t1r + t2r;
        b[[i, 0, 1]] = a[[i, 0, 0]] - a[[i, 2, 0]];
        b[[i, 0, 2]] = t1r - t2r;
        b[[i, 0, 3]] = a[[i, 3, 0]] - a[[i, 1, 0]];
    }
    //Next do remaining k:
    if nv < (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                x1p = cosine[[k, 0]] * a[[i, 1, k]] - sine[[k, 0]] * a[[i, 1, kc]];
                y1p = cosine[[k, 0]] * a[[i, 1, kc]] + sine[[k, 0]] * a[[i, 1, k]];
                x2p = cosine[[k, 1]] * a[[i, 2, k]] - sine[[k, 1]] * a[[i, 2, kc]];
                y2p = cosine[[k, 1]] * a[[i, 2, kc]] + sine[[k, 1]] * a[[i, 2, k]];
                x3p = cosine[[k, 2]] * a[[i, 3, k]] - sine[[k, 2]] * a[[i, 3, kc]];
                y3p = cosine[[k, 2]] * a[[i, 3, kc]] + sine[[k, 2]] * a[[i, 3, k]];
                t1r = a[[i, 0, k]] + x2p;
                t1i = a[[i, 0, kc]] + y2p;
                t2r = x1p + x3p;
                t2i = y1p + y3p;
                t3r = a[[i, 0, k]] - x2p;
                t3i = a[[i, 0, kc]] - y2p;
                t4r = x3p - x1p;
                t4i = y1p - y3p;
                b[[i, k, 0]] = t1r + t2r;
                b[[i, kc, 0]] = t3r - t4i;
                b[[i, k, 1]] = t3r + t4i;
                b[[i, kc, 1]] = t1r - t2r;
                b[[i, k, 2]] = t2i - t1i;
                b[[i, kc, 2]] = t3i + t4r;
                b[[i, k, 3]] = t4r - t3i;
                b[[i, kc, 3]] = t1i + t2i;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[k, 0]];
            s1k = sine[[k, 0]];
            c2k = cosine[[k, 1]];
            s2k = sine[[k, 1]];
            c3k = cosine[[k, 2]];
            s3k = sine[[k, 2]];
            for i in 0..nv {
                x1p = c1k * a[[i, 1, k]] - s1k * a[[i, 1, kc]];
                y1p = c1k * a[[i, 1, kc]] + s1k * a[[i, 1, k]];
                x2p = c2k * a[[i, 2, k]] - s2k * a[[i, 2, kc]];
                y2p = c2k * a[[i, 2, kc]] + s2k * a[[i, 2, k]];
                x3p = c3k * a[[i, 3, k]] - s3k * a[[i, 3, kc]];
                y3p = c3k * a[[i, 3, kc]] + s3k * a[[i, 3, k]];
                t1r = a[[i, 0, k]] + x2p;
                t1i = a[[i, 0, kc]] + y2p;
                t2r = x1p + x3p;
                t2i = y1p + y3p;
                t3r = a[[i, 0, k]] - x2p;
                t3i = a[[i, 0, kc]] - y2p;
                t4r = x3p - x1p;
                t4i = y1p - y3p;
                b[[i, k, 0]] = t1r + t2r;
                b[[i, kc, 0]] = t3r - t4i;
                b[[i, k, 1]] = t3r + t4i;
                b[[i, kc, 1]] = t1r - t2r;
                b[[i, k, 2]] = t2i - t1i;
                b[[i, kc, 2]] = t3i + t4r;
                b[[i, k, 3]] = t4r - t3i;
                b[[i, kc, 3]] = t1i + t2i;
            }
        }
    }

    //Catch the case k=lv/2 when lv even:
    if lv % 2 == 0 {
        let lvd2 = lv / 2;
        for i in 0..nv {
            q1 = FRAC_1_SQRT_2 * (a[[i, 1, lvd2]] - a[[i, 3, lvd2]]);
            q2 = FRAC_1_SQRT_2 * (a[[i, 1, lvd2]] + a[[i, 3, lvd2]]);
            b[[i, lvd2, 0]] = a[[i, 0, lvd2]] + q1;
            b[[i, lvd2, 1]] = a[[i, 0, lvd2]] - q1;
            b[[i, lvd2, 2]] = a[[i, 2, lvd2]] - q2;
            b[[i, lvd2, 3]] = -a[[i, 2, lvd2]] - q2;
        }
    }
}

/// Radix three physical to Hermitian FFT with 'decimation in time'.
pub fn forrdx3(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(3, a.len_of(Axis(1)));
    assert_eq!(lv, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(lv, b.len_of(Axis(1)));
    assert_eq!(3, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(2, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(2, sine.len_of(Axis(1)));

    let mut x1p: f64;
    let mut x2p: f64;
    let mut y1p: f64;
    let mut y2p: f64;

    let mut c2k: f64;
    let mut c1k: f64;
    let mut s2k: f64;
    let mut s1k: f64;

    let mut t1i: f64;
    let mut t1r: f64;
    let mut t2i: f64;
    let mut t2r: f64;
    let mut t3i: f64;
    let mut t3r: f64;

    let mut kc: usize;

    //Do k=0 first:
    for i in 0..nv {
        t1r = a[[i, 1, 0]] + a[[i, 2, 0]];
        b[[i, 0, 0]] = a[[i, 0, 0]] + t1r;
        b[[i, 0, 1]] = a[[i, 0, 0]] - 0.5 * t1r;
        b[[i, 0, 2]] = SINFPI3 * (a[[i, 2, 0]] - a[[i, 1, 0]]);
    }
    //Next do remaining k:
    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                x1p = cosine[[k, 0]] * a[[i, 1, k]] - sine[[k, 0]] * a[[i, 1, kc]];
                y1p = cosine[[k, 0]] * a[[i, 1, kc]] + sine[[k, 0]] * a[[i, 1, k]];
                x2p = cosine[[k, 1]] * a[[i, 2, k]] - sine[[k, 1]] * a[[i, 2, kc]];
                y2p = cosine[[k, 1]] * a[[i, 2, kc]] + sine[[k, 1]] * a[[i, 2, k]];
                t1r = x1p + x2p;
                t1i = y1p + y2p;
                t2r = a[[i, 0, k]] - 0.5 * t1r;
                t2i = 0.5 * t1i - a[[i, 0, kc]];
                t3r = SINFPI3 * (x2p - x1p);
                t3i = SINFPI3 * (y1p - y2p);
                b[[i, k, 0]] = a[[i, 0, k]] + t1r;
                b[[i, kc, 0]] = t2r - t3i;
                b[[i, k, 1]] = t2r + t3i;
                b[[i, kc, 1]] = t3r - t2i;
                b[[i, k, 2]] = t2i + t3r;
                b[[i, kc, 2]] = a[[i, 0, kc]] + t1i;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[k, 0]];
            s1k = sine[[k, 0]];
            c2k = cosine[[k, 1]];
            s2k = sine[[k, 1]];
            for i in 0..nv {
                x1p = c1k * a[[i, 1, k]] - s1k * a[[i, 1, kc]];
                y1p = c1k * a[[i, 1, kc]] + s1k * a[[i, 1, k]];
                x2p = c2k * a[[i, 2, k]] - s2k * a[[i, 2, kc]];
                y2p = c2k * a[[i, 2, kc]] + s2k * a[[i, 2, k]];
                t1r = x1p + x2p;
                t1i = y1p + y2p;
                t2r = a[[i, 0, k]] - 0.5 * t1r;
                t2i = 0.5 * t1i - a[[i, 0, kc]];
                t3r = SINFPI3 * (x2p - x1p);
                t3i = SINFPI3 * (y1p - y2p);
                b[[i, k, 0]] = a[[i, 0, k]] + t1r;
                b[[i, kc, 0]] = t2r - t3i;
                b[[i, k, 1]] = t2r + t3i;
                b[[i, kc, 1]] = t3r - t2i;
                b[[i, k, 2]] = t2i + t3r;
                b[[i, kc, 2]] = a[[i, 0, kc]] + t1i;
            }
        }
    }
}

/// Radix two physical to Hermitian FFT with 'decimation in time'.
pub fn forrdx2(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(2, a.len_of(Axis(1)));
    assert_eq!(lv, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(lv, b.len_of(Axis(1)));
    assert_eq!(2, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(1, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(1, sine.len_of(Axis(1)));

    let mut x1: f64;
    let mut y1: f64;
    let mut c1k: f64;
    let mut s1k: f64;

    let mut kc: usize;

    //Do k=0 first:
    for i in 0..nv {
        b[[i, 0, 0]] = a[[i, 0, 0]] + a[[i, 1, 0]];
        b[[i, 0, 1]] = a[[i, 0, 0]] - a[[i, 1, 0]];
    }
    //Next do remaining k:
    if nv < (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                x1 = cosine[[0, k - 1]] * a[[i, 1, k]] - sine[[0, k - 1]] * a[[i, 1, kc]];
                y1 = cosine[[0, k - 1]] * a[[i, 1, kc]] + sine[[0, k - 1]] * a[[i, 1, k]];
                b[[i, k, 0]] = a[[i, 0, k]] + x1;
                b[[i, kc, 0]] = a[[i, 0, k]] - x1;
                b[[i, k, 1]] = y1 - a[[i, 0, kc]];
                b[[i, kc, 1]] = a[[i, 0, kc]] + y1;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[0, k - 1]];
            s1k = sine[[0, k - 1]];
            for i in 0..nv {
                x1 = c1k * a[[i, 1, k]] - s1k * a[[i, 1, kc]];
                y1 = c1k * a[[i, 1, kc]] + s1k * a[[i, 1, k]];
                b[[i, k, 0]] = a[[i, 0, k]] + x1;
                b[[i, kc, 0]] = a[[i, 0, k]] - x1;
                b[[i, k, 1]] = y1 - a[[i, 0, kc]];
                b[[i, kc, 1]] = a[[i, 0, kc]] + y1;
            }
        }
    }
}
