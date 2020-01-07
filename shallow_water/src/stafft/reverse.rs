//! Spectral to physical (reverse) routines
use {
    crate::constants::*,
    core::f64::consts::FRAC_1_SQRT_2,
    ndarray::{ArrayView2, ArrayView3, ArrayViewMut3, Axis},
};

/// Radix six Hermitian to physical FFT with 'decimation in frequency'.
pub fn revrdx6(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(lv, a.len_of(Axis(1)));
    assert_eq!(6, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(6, b.len_of(Axis(1)));
    assert_eq!(lv, b.len_of(Axis(2)));

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

    //Do k=0 first:
    for i in 0..nv {
        t2r = a[[i, 0, 0]] - 0.5 * a[[i, 0, 2]];
        t3r = SINFPI3 * a[[i, 0, 4]];
        u0r = a[[i, 0, 0]] + a[[i, 0, 2]];
        u1r = t2r + t3r;
        u2r = t2r - t3r;
        t2i = a[[i, 0, 3]] - 0.5 * a[[i, 0, 1]];
        t3i = -SINFPI3 * a[[i, 0, 5]];
        v0r = a[[i, 0, 3]] + a[[i, 0, 1]];
        v1r = t2i + t3i;
        v2r = t2i - t3i;
        b[[i, 0, 0]] = u0r + v0r;
        b[[i, 1, 0]] = u1r - v1r;
        b[[i, 2, 0]] = u2r + v2r;
        b[[i, 3, 0]] = u0r - v0r;
        b[[i, 4, 0]] = u1r + v1r;
        b[[i, 5, 0]] = u2r - v2r;
    }
    //Next do remaining k:
    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                t1r = a[[i, k, 2]] + a[[i, kc, 1]];
                t1i = a[[i, kc, 3]] - a[[i, k, 4]];
                t2r = a[[i, k, 0]] - 0.5 * t1r;
                t2i = a[[i, kc, 5]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, k, 2]] - a[[i, kc, 1]]);
                t3i = SINFPI3 * (a[[i, kc, 3]] + a[[i, k, 4]]);
                u0r = a[[i, k, 0]] + t1r;
                u0i = a[[i, kc, 5]] + t1i;
                u1r = t2r + t3i;
                u1i = t2i - t3r;
                u2r = t2r - t3i;
                u2i = t2i + t3r;
                t1r = a[[i, kc, 0]] + a[[i, k, 1]];
                t1i = a[[i, kc, 4]] - a[[i, k, 5]];
                t2r = a[[i, kc, 2]] - 0.5 * t1r;
                t2i = -a[[i, k, 3]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, kc, 0]] - a[[i, k, 1]]);
                t3i = SINFPI3 * (-a[[i, k, 5]] - a[[i, kc, 4]]);
                v0r = a[[i, kc, 2]] + t1r;
                v0i = t1i - a[[i, k, 3]];
                v1r = t2r + t3i;
                v1i = t2i - t3r;
                v2r = t2r - t3i;
                v2i = t2i + t3r;
                x1p = u1r - v1r;
                y1p = u1i - v1i;
                x2p = u2r + v2r;
                y2p = u2i + v2i;
                x3p = u0r - v0r;
                y3p = u0i - v0i;
                x4p = u1r + v1r;
                y4p = u1i + v1i;
                x5p = u2r - v2r;
                y5p = u2i - v2i;
                b[[i, 0, k]] = u0r + v0r;
                b[[i, 0, kc]] = u0i + v0i;
                b[[i, 1, k]] = cosine[[k, 0]] * x1p - sine[[k, 0]] * y1p;
                b[[i, 1, kc]] = cosine[[k, 0]] * y1p + sine[[k, 0]] * x1p;
                b[[i, 2, k]] = cosine[[k, 1]] * x2p - sine[[k, 1]] * y2p;
                b[[i, 2, kc]] = cosine[[k, 1]] * y2p + sine[[k, 1]] * x2p;
                b[[i, 3, k]] = cosine[[k, 2]] * x3p - sine[[k, 2]] * y3p;
                b[[i, 3, kc]] = cosine[[k, 2]] * y3p + sine[[k, 2]] * x3p;
                b[[i, 4, k]] = cosine[[k, 4]] * x4p - sine[[k, 4]] * y4p;
                b[[i, 4, kc]] = cosine[[k, 4]] * y4p + sine[[k, 4]] * x4p;
                b[[i, 5, k]] = cosine[[k, 5]] * x5p - sine[[k, 5]] * y5p;
                b[[i, 5, kc]] = cosine[[k, 5]] * y5p + sine[[k, 5]] * x5p;
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
                t1r = a[[i, k, 2]] + a[[i, kc, 1]];
                t1i = a[[i, kc, 3]] - a[[i, k, 4]];
                t2r = a[[i, k, 0]] - 0.5 * t1r;
                t2i = a[[i, kc, 5]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, k, 2]] - a[[i, kc, 1]]);
                t3i = SINFPI3 * (a[[i, kc, 3]] + a[[i, k, 4]]);
                u0r = a[[i, k, 0]] + t1r;
                u0i = a[[i, kc, 5]] + t1i;
                u1r = t2r + t3i;
                u1i = t2i - t3r;
                u2r = t2r - t3i;
                u2i = t2i + t3r;
                t1r = a[[i, kc, 0]] + a[[i, k, 1]];
                t1i = a[[i, kc, 4]] - a[[i, k, 5]];
                t2r = a[[i, kc, 2]] - 0.5 * t1r;
                t2i = -a[[i, k, 3]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, kc, 0]] - a[[i, k, 1]]);
                t3i = SINFPI3 * (-a[[i, k, 5]] - a[[i, kc, 4]]);
                v0r = a[[i, kc, 2]] + t1r;
                v0i = t1i - a[[i, k, 3]];
                v1r = t2r + t3i;
                v1i = t2i - t3r;
                v2r = t2r - t3i;
                v2i = t2i + t3r;
                x1p = u1r - v1r;
                y1p = u1i - v1i;
                x2p = u2r + v2r;
                y2p = u2i + v2i;
                x3p = u0r - v0r;
                y3p = u0i - v0i;
                x4p = u1r + v1r;
                y4p = u1i + v1i;
                x5p = u2r - v2r;
                y5p = u2i - v2i;
                b[[i, 0, k]] = u0r + v0r;
                b[[i, 0, kc]] = u0i + v0i;
                b[[i, 1, k]] = c1k * x1p - s1k * y1p;
                b[[i, 1, kc]] = c1k * y1p + s1k * x1p;
                b[[i, 2, k]] = c2k * x2p - s2k * y2p;
                b[[i, 2, kc]] = c2k * y2p + s2k * x2p;
                b[[i, 3, k]] = c3k * x3p - s3k * y3p;
                b[[i, 3, kc]] = c3k * y3p + s3k * x3p;
                b[[i, 4, k]] = c4k * x4p - s4k * y4p;
                b[[i, 4, kc]] = c4k * y4p + s4k * x4p;
                b[[i, 5, k]] = c5k * x5p - s5k * y5p;
                b[[i, 5, kc]] = c5k * y5p + s5k * x5p;
            }
        }
    }

    //Catch the case k=lv/2 when lv even:
    if lv % 2 == 0 {
        let lvd2 = lv / 2;
        for i in 0..nv {
            q1 = a[[i, lvd2, 0]] + a[[i, lvd2, 2]];
            q2 = a[[i, lvd2, 5]] + a[[i, lvd2, 3]];
            q3 = a[[i, lvd2, 1]] - 0.5 * q1;
            q4 = a[[i, lvd2, 4]] + 0.5 * q2;
            q5 = SINFPI3 * (a[[i, lvd2, 0]] - a[[i, lvd2, 2]]);
            q6 = SINFPI3 * (a[[i, lvd2, 5]] - a[[i, lvd2, 3]]);
            b[[i, 0, lvd2]] = a[[i, lvd2, 1]] + q1;
            b[[i, 1, lvd2]] = q4 + q5;
            b[[i, 2, lvd2]] = q6 - q3;
            b[[i, 3, lvd2]] = q2 - a[[i, lvd2, 4]];
            b[[i, 4, lvd2]] = q3 + q6;
            b[[i, 5, lvd2]] = q4 - q5;
        }
    }
}

/// Radix five Hermitian to physical FFT with 'decimation in frequency'.
pub fn revrdx5(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(lv, a.len_of(Axis(1)));
    assert_eq!(5, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(5, b.len_of(Axis(1)));
    assert_eq!(lv, b.len_of(Axis(2)));

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
        t3r = SINF2PI5 * a[[i, 0, 4]];
        t4r = SINF2PI5 * a[[i, 0, 3]];
        t5r = a[[i, 0, 1]] + a[[i, 0, 2]];
        t6r = RTF516 * (a[[i, 0, 1]] - a[[i, 0, 2]]);
        t7r = a[[i, 0, 0]] - 0.25 * t5r;
        t8r = t7r + t6r;
        t9r = t7r - t6r;
        t10r = t3r + SINRAT * t4r;
        t11r = SINRAT * t3r - t4r;
        b[[i, 0, 0]] = a[[i, 0, 0]] + t5r;
        b[[i, 1, 0]] = t8r + t10r;
        b[[i, 2, 0]] = t9r + t11r;
        b[[i, 3, 0]] = t9r - t11r;
        b[[i, 4, 0]] = t8r - t10r;
    }
    //Next do remaining k:
    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                t1r = a[[i, k, 1]] + a[[i, kc, 0]];
                t1i = a[[i, kc, 3]] - a[[i, k, 4]];
                t2r = a[[i, k, 2]] + a[[i, kc, 1]];
                t2i = a[[i, kc, 2]] - a[[i, k, 3]];
                t3r = SINF2PI5 * (a[[i, k, 1]] - a[[i, kc, 0]]);
                t3i = SINF2PI5 * (a[[i, kc, 3]] + a[[i, k, 4]]);
                t4r = SINF2PI5 * (a[[i, k, 2]] - a[[i, kc, 1]]);
                t4i = SINF2PI5 * (a[[i, kc, 2]] + a[[i, k, 3]]);
                t5r = t1r + t2r;
                t5i = t1i + t2i;
                t6r = RTF516 * (t1r - t2r);
                t6i = RTF516 * (t1i - t2i);
                t7r = a[[i, k, 0]] - 0.25 * t5r;
                t7i = a[[i, kc, 4]] - 0.25 * t5i;
                t8r = t7r + t6r;
                t8i = t7i + t6i;
                t9r = t7r - t6r;
                t9i = t7i - t6i;
                t10r = t3r + SINRAT * t4r;
                t10i = t3i + SINRAT * t4i;
                t11r = SINRAT * t3r - t4r;
                t11i = SINRAT * t3i - t4i;
                x1p = t8r + t10i;
                y1p = t8i - t10r;
                x2p = t9r + t11i;
                y2p = t9i - t11r;
                x3p = t9r - t11i;
                y3p = t9i + t11r;
                x4p = t8r - t10i;
                y4p = t8i + t10r;
                b[[i, 0, k]] = a[[i, k, 0]] + t5r;
                b[[i, 0, kc]] = a[[i, kc, 4]] + t5i;
                b[[i, 1, k]] = cosine[[k, 0]] * x1p - sine[[k, 0]] * y1p;
                b[[i, 1, kc]] = cosine[[k, 0]] * y1p + sine[[k, 0]] * x1p;
                b[[i, 2, k]] = cosine[[k, 1]] * x2p - sine[[k, 1]] * y2p;
                b[[i, 2, kc]] = cosine[[k, 1]] * y2p + sine[[k, 1]] * x2p;
                b[[i, 3, k]] = cosine[[k, 2]] * x3p - sine[[k, 2]] * y3p;
                b[[i, 3, kc]] = cosine[[k, 2]] * y3p + sine[[k, 2]] * x3p;
                b[[i, 4, k]] = cosine[[k, 4]] * x4p - sine[[k, 4]] * y4p;
                b[[i, 4, kc]] = cosine[[k, 4]] * y4p + sine[[k, 4]] * x4p;
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
            c4k = cosine[[k, 4]];
            s4k = sine[[k, 4]];
            for i in 0..nv {
                t1r = a[[i, k, 1]] + a[[i, kc, 0]];
                t1i = a[[i, kc, 3]] - a[[i, k, 4]];
                t2r = a[[i, k, 2]] + a[[i, kc, 1]];
                t2i = a[[i, kc, 2]] - a[[i, k, 3]];
                t3r = SINF2PI5 * (a[[i, k, 1]] - a[[i, kc, 0]]);
                t3i = SINF2PI5 * (a[[i, kc, 3]] + a[[i, k, 4]]);
                t4r = SINF2PI5 * (a[[i, k, 2]] - a[[i, kc, 1]]);
                t4i = SINF2PI5 * (a[[i, kc, 2]] + a[[i, k, 3]]);
                t5r = t1r + t2r;
                t5i = t1i + t2i;
                t6r = RTF516 * (t1r - t2r);
                t6i = RTF516 * (t1i - t2i);
                t7r = a[[i, k, 0]] - 0.25 * t5r;
                t7i = a[[i, kc, 4]] - 0.25 * t5i;
                t8r = t7r + t6r;
                t8i = t7i + t6i;
                t9r = t7r - t6r;
                t9i = t7i - t6i;
                t10r = t3r + SINRAT * t4r;
                t10i = t3i + SINRAT * t4i;
                t11r = SINRAT * t3r - t4r;
                t11i = SINRAT * t3i - t4i;
                x1p = t8r + t10i;
                y1p = t8i - t10r;
                x2p = t9r + t11i;
                y2p = t9i - t11r;
                x3p = t9r - t11i;
                y3p = t9i + t11r;
                x4p = t8r - t10i;
                y4p = t8i + t10r;
                b[[i, 0, k]] = a[[i, k, 0]] + t5r;
                b[[i, 0, kc]] = a[[i, kc, 4]] + t5i;
                b[[i, 1, k]] = c1k * x1p - s1k * y1p;
                b[[i, 1, kc]] = c1k * y1p + s1k * x1p;
                b[[i, 2, k]] = c2k * x2p - s2k * y2p;
                b[[i, 2, kc]] = c2k * y2p + s2k * x2p;
                b[[i, 3, k]] = c3k * x3p - s3k * y3p;
                b[[i, 3, kc]] = c3k * y3p + s3k * x3p;
                b[[i, 4, k]] = c4k * x4p - s4k * y4p;
                b[[i, 4, kc]] = c4k * y4p + s4k * x4p;
            }
        }
    }
}

/// Radix four Hermitian to physical FFT with 'decimation in frequency'.
pub fn revrdx4(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(lv, a.len_of(Axis(1)));
    assert_eq!(4, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(4, b.len_of(Axis(1)));
    assert_eq!(lv, b.len_of(Axis(2)));

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

    let mut kc: usize;

    //Do k=0 first:
    for i in 0..nv {
        t1r = a[[i, 0, 0]] + a[[i, 0, 2]];
        t2r = a[[i, 0, 1]];
        t3r = a[[i, 0, 0]] - a[[i, 0, 2]];
        t4r = a[[i, 0, 3]];
        b[[i, 0, 0]] = t1r + t2r;
        b[[i, 1, 0]] = t3r + t4r;
        b[[i, 2, 0]] = t1r - t2r;
        b[[i, 3, 0]] = t3r - t4r;
    }
    //Next do remaining k:
    if nv < (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                t1r = a[[i, k, 0]] + a[[i, kc, 1]];
                t1i = a[[i, kc, 3]] - a[[i, k, 2]];
                t2r = a[[i, k, 1]] + a[[i, kc, 0]];
                t2i = a[[i, kc, 2]] - a[[i, k, 3]];
                t3r = a[[i, k, 0]] - a[[i, kc, 1]];
                t3i = a[[i, kc, 3]] + a[[i, k, 2]];
                t4r = a[[i, k, 1]] - a[[i, kc, 0]];
                t4i = a[[i, kc, 2]] + a[[i, k, 3]];
                x1p = t3r + t4i;
                y1p = t3i - t4r;
                x2p = t1r - t2r;
                y2p = t1i - t2i;
                x3p = t3r - t4i;
                y3p = t3i + t4r;
                b[[i, 0, k]] = t1r + t2r;
                b[[i, 0, kc]] = t1i + t2i;
                b[[i, 1, k]] = cosine[[k, 0]] * x1p - sine[[k, 0]] * y1p;
                b[[i, 1, kc]] = cosine[[k, 0]] * y1p + sine[[k, 0]] * x1p;
                b[[i, 2, k]] = cosine[[k, 1]] * x2p - sine[[k, 1]] * y2p;
                b[[i, 2, kc]] = cosine[[k, 1]] * y2p + sine[[k, 1]] * x2p;
                b[[i, 3, k]] = cosine[[k, 2]] * x3p - sine[[k, 2]] * y3p;
                b[[i, 3, kc]] = cosine[[k, 2]] * y3p + sine[[k, 2]] * x3p;
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
                t1r = a[[i, k, 0]] + a[[i, kc, 1]];
                t1i = a[[i, kc, 3]] - a[[i, k, 2]];
                t2r = a[[i, k, 1]] + a[[i, kc, 0]];
                t2i = a[[i, kc, 2]] - a[[i, k, 3]];
                t3r = a[[i, k, 0]] - a[[i, kc, 1]];
                t3i = a[[i, kc, 3]] + a[[i, k, 2]];
                t4r = a[[i, k, 1]] - a[[i, kc, 0]];
                t4i = a[[i, kc, 2]] + a[[i, k, 3]];
                x1p = t3r + t4i;
                y1p = t3i - t4r;
                x2p = t1r - t2r;
                y2p = t1i - t2i;
                x3p = t3r - t4i;
                y3p = t3i + t4r;
                b[[i, 0, k]] = t1r + t2r;
                b[[i, 0, kc]] = t1i + t2i;
                b[[i, 1, k]] = c1k * x1p - s1k * y1p;
                b[[i, 1, kc]] = c1k * y1p + s1k * x1p;
                b[[i, 2, k]] = c2k * x2p - s2k * y2p;
                b[[i, 2, kc]] = c2k * y2p + s2k * x2p;
                b[[i, 3, k]] = c3k * x3p - s3k * y3p;
                b[[i, 3, kc]] = c3k * y3p + s3k * x3p;
            }
        }
    }

    //Catch the case k=lv/2 when lv even:
    if lv % 2 == 0 {
        let lvd2 = lv / 2;
        for i in 0..nv {
            b[[i, 0, lvd2]] = a[[i, lvd2, 0]] + a[[i, lvd2, 1]];
            b[[i, 2, lvd2]] = a[[i, lvd2, 3]] - a[[i, lvd2, 2]];
            t3r = a[[i, lvd2, 0]] - a[[i, lvd2, 1]];
            t4r = a[[i, lvd2, 3]] + a[[i, lvd2, 2]];
            b[[i, 1, lvd2]] = FRAC_1_SQRT_2 * (t3r + t4r);
            b[[i, 3, lvd2]] = FRAC_1_SQRT_2 * (t4r - t3r);
        }
    }
}

/// Radix three Hermitian to physical FFT with 'decimation in frequency'.
pub fn revrdx3(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(lv, a.len_of(Axis(1)));
    assert_eq!(3, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(3, b.len_of(Axis(1)));
    assert_eq!(lv, b.len_of(Axis(2)));

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

    for i in 0..nv {
        t1r = a[[i, 0, 1]];
        t2r = a[[i, 0, 0]] - 0.5 * t1r;
        t3r = SINFPI3 * a[[i, 0, 2]];
        b[[i, 0, 0]] = a[[i, 0, 0]] + t1r;
        b[[i, 1, 0]] = t2r + t3r;
        b[[i, 2, 0]] = t2r - t3r;
    }

    if nv <= (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;
                t1r = a[[i, k, 1]] + a[[i, kc, 0]];
                t1i = a[[i, kc, 1]] - a[[i, k, 2]];
                t2r = a[[i, k, 0]] - 0.5 * t1r;
                t2i = a[[i, kc, 2]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, k, 1]] - a[[i, kc, 0]]);
                t3i = SINFPI3 * (a[[i, kc, 1]] + a[[i, k, 2]]);
                x1p = t2r + t3i;
                y1p = t2i - t3r;
                x2p = t2r - t3i;
                y2p = t2i + t3r;
                b[[i, 0, k]] = a[[i, k, 0]] + t1r;
                b[[i, 0, kc]] = a[[i, kc, 2]] + t1i;
                b[[i, 1, k]] = cosine[[k, 0]] * x1p - sine[[k, 0]] * y1p;
                b[[i, 1, kc]] = sine[[k, 0]] * x1p + cosine[[k, 0]] * y1p;
                b[[i, 2, k]] = cosine[[k, 1]] * x2p - sine[[k, 1]] * y2p;
                b[[i, 2, kc]] = sine[[k, 1]] * x2p + cosine[[k, 1]] * y2p;
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
                t1r = a[[i, k, 1]] + a[[i, kc, 0]];
                t1i = a[[i, kc, 1]] - a[[i, k, 2]];
                t2r = a[[i, k, 0]] - 0.5 * t1r;
                t2i = a[[i, kc, 2]] - 0.5 * t1i;
                t3r = SINFPI3 * (a[[i, k, 1]] - a[[i, kc, 0]]);
                t3i = SINFPI3 * (a[[i, kc, 1]] + a[[i, k, 2]]);
                x1p = t2r + t3i;
                y1p = t2i - t3r;
                x2p = t2r - t3i;
                y2p = t2i + t3r;
                b[[i, 0, k]] = a[[i, k, 0]] + t1r;
                b[[i, 0, kc]] = a[[i, kc, 2]] + t1i;
                b[[i, 1, k]] = c1k * x1p - s1k * y1p;
                b[[i, 1, kc]] = s1k * x1p + c1k * y1p;
                b[[i, 2, k]] = c2k * x2p - s2k * y2p;
                b[[i, 2, kc]] = s2k * x2p + c2k * y2p;
            }
        }
    }
}

/// Radix two Hermitian to physical FFT with 'decimation in frequency'.
pub fn revrdx2(
    a: ArrayView3<f64>,
    mut b: ArrayViewMut3<f64>,
    nv: usize,
    lv: usize,
    cosine: ArrayView2<f64>,
    sine: ArrayView2<f64>,
) {
    assert_eq!(nv, a.len_of(Axis(0)));
    assert_eq!(lv, a.len_of(Axis(1)));
    assert_eq!(2, a.len_of(Axis(2)));

    assert_eq!(nv, b.len_of(Axis(0)));
    assert_eq!(2, b.len_of(Axis(1)));
    assert_eq!(lv, b.len_of(Axis(2)));

    assert_eq!(lv, cosine.len_of(Axis(0)));
    assert_eq!(1, cosine.len_of(Axis(1)));

    assert_eq!(lv, sine.len_of(Axis(0)));
    assert_eq!(1, sine.len_of(Axis(1)));

    let mut x1p: f64;
    let mut y1p: f64;
    let mut c1k: f64;
    let mut s1k: f64;

    let mut kc: usize;

    // Do k=0 first:
    for i in 0..nv {
        b[[i, 0, 0]] = a[[i, 0, 0]] + a[[i, 0, 1]];
        b[[i, 1, 0]] = a[[i, 0, 0]] - a[[i, 0, 1]];
    }

    // !Next do remaining k:
    if nv < (lv - 1) / 2 {
        for i in 0..nv {
            for k in 1..=(lv - 1) / 2 {
                kc = lv - k;

                x1p = a[[i, k, 0]] - a[[i, kc, 0]];
                y1p = a[[i, kc, 1]] + a[[i, k, 1]];

                b[[i, 0, k]] = a[[i, k, 0]] + a[[i, kc, 0]];
                b[[i, 0, kc]] = a[[i, kc, 1]] - a[[i, k, 1]];
                b[[i, 1, k]] = cosine[[k, 0]] * x1p - sine[[k, 0]] * y1p;
                b[[i, 1, kc]] = cosine[[k, 0]] * y1p + sine[[k, 0]] * x1p;
            }
        }
    } else {
        for k in 1..=(lv - 1) / 2 {
            kc = lv - k;
            c1k = cosine[[k, 0]];
            s1k = sine[[k, 0]];

            for i in 0..nv {
                x1p = a[[i, k, 0]] - a[[i, kc, 0]];
                y1p = a[[i, kc, 1]] + a[[i, k, 1]];
                b[[i, 0, k]] = a[[i, k, 0]] + a[[i, kc, 0]];
                b[[i, 0, kc]] = a[[i, kc, 1]] - a[[i, k, 1]];
                b[[i, 1, k]] = c1k * x1p - s1k * y1p;
                b[[i, 1, kc]] = c1k * y1p + s1k * x1p;
            }
        }
    }
}
