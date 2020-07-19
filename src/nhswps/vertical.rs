use {
    crate::{constants::*, nhswps::State, utils::*},
    ndarray::{Axis, Zip},
};

/// Calculates layer heights (z), as well as dz/dx & dz/dy (zx & zy),
/// the vertical velocity (w), and the A = grad{u*rho'_theta} (aa).
pub fn vertical(state: &mut State) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let dz2 = (HBAR / nz as f64) / 2.0;

    let mut rsrc = arr3zero(ng, nz);
    let mut wkq = arr2zero(ng);
    let mut wka = arr2zero(ng);
    let mut wkb = arr2zero(ng);
    let mut wkc = arr2zero(ng);

    // Only need to consider iz > 0 as z = w = 0 for iz = 0:

    // Find z by trapezoidal integration of rho_theta (integrate over
    // rho'_theta then add theta to the result):

    Zip::from(state.z.index_axis_mut(Axis(2), 1))
        .and(state.r.index_axis(Axis(2), 0))
        .and(state.r.index_axis(Axis(2), 1))
        .apply(|z, r0, r1| *z = dz2 * (r0 + r1));

    let mut last_layer = state.z.index_axis(Axis(2), 1).into_owned();
    for iz in 1..nz {
        Zip::from(state.z.index_axis_mut(Axis(2), iz + 1))
            .and(&last_layer)
            .and(state.r.index_axis(Axis(2), iz))
            .and(state.r.index_axis(Axis(2), iz + 1))
            .apply(|z1, z0, r0, r1| *z1 = z0 + dz2 * (r0 + r1));

        last_layer.assign(&state.z.index_axis(Axis(2), iz + 1));
    }

    for iz in 1..=nz {
        // Add on theta (a linear function) to complete definition of z:
        let theta = state.spectral.theta[iz];
        Zip::from(state.z.index_axis_mut(Axis(2), iz)).apply(|z| *z += theta);

        // Calculate z_x & z_y:
        let mut wkq = state.z.index_axis(Axis(2), iz).to_owned();

        state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, wka.view(), wkb.view_mut());
        state.spectral.d2fft.spctop(wkb.view_mut(), wkq.view_mut());

        state.zx.index_axis_mut(Axis(2), iz).assign(&wkq);

        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, wka.view(), wkb.view_mut());
        state.spectral.d2fft.spctop(wkb.view_mut(), wkq.view_mut());

        state.zy.index_axis_mut(Axis(2), iz).assign(&wkq);
    }

    // Calculate A = grad{u*rho'_theta} (spectral):
    for iz in 0..=nz {
        // Calculate (u*rho'_theta)_x:
        Zip::from(&mut wkq)
            .and(state.u.index_axis(Axis(2), iz))
            .and(state.r.index_axis(Axis(2), iz))
            .apply(|wkq, u, r| *wkq = u * r);

        state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
        state
            .spectral
            .d2fft
            .xderiv(&state.spectral.hrkx, wka.view(), wkb.view_mut());

        // Calculate (v*rho'_theta)_y:
        Zip::from(&mut wkq)
            .and(state.v.index_axis(Axis(2), iz))
            .and(state.r.index_axis(Axis(2), iz))
            .apply(|wkq, v, r| *wkq = v * r);

        state.spectral.d2fft.ptospc(wkq.view_mut(), wka.view_mut());
        state
            .spectral
            .d2fft
            .yderiv(&state.spectral.hrky, wka.view(), wkc.view_mut());

        // Apply de-aliasing filter and complete definition of A:
        Zip::from(state.aa.index_axis_mut(Axis(2), iz))
            .and(&state.spectral.filt)
            .and(&wkb)
            .and(&wkc)
            .apply(|aa, filt, wkb, wkc| *aa = filt * (wkb + wkc));

        // Need -(A + delta) in physical space for computing w just below:
        Zip::from(&mut wka)
            .and(state.aa.index_axis(Axis(2), iz))
            .and(state.ds.index_axis(Axis(2), iz))
            .apply(|wka, aa, ds| *wka = aa + ds);

        state.spectral.d2fft.spctop(wka.view_mut(), wkq.view_mut());

        Zip::from(rsrc.index_axis_mut(Axis(2), iz))
            .and(&wkq)
            .apply(|rsrc, wkq| *rsrc = -wkq);
    }

    // Calculate vertical velocity (0 at iz = 0):
    Zip::from(state.w.index_axis_mut(Axis(2), 1))
        .and(rsrc.index_axis(Axis(2), 0))
        .and(rsrc.index_axis(Axis(2), 1))
        .apply(|w, rsrc0, rsrc1| *w = dz2 * (rsrc0 + rsrc1));

    for iz in 1..nz {
        let w0 = state.w.index_axis(Axis(2), iz).to_owned();
        Zip::from(state.w.index_axis_mut(Axis(2), iz + 1))
            .and(&w0)
            .and(rsrc.index_axis(Axis(2), iz))
            .and(rsrc.index_axis(Axis(2), iz + 1))
            .apply(|w1, w0, rsrc0, rsrc1| *w1 = w0 + dz2 * (rsrc0 + rsrc1));
    }

    // Complete definition of w by adding u*z_x + v*z_y after de-aliasing:
    for iz in 1..=nz {
        Zip::from(&mut wkq)
            .and(state.u.index_axis(Axis(2), iz))
            .and(state.zx.index_axis(Axis(2), iz))
            .and(state.v.index_axis(Axis(2), iz))
            .and(state.zy.index_axis(Axis(2), iz))
            .apply(|wkq, u, zx, v, zy| *wkq = u * zx + v * zy);

        state.spectral.deal2d(wkq.view_mut());

        Zip::from(state.w.index_axis_mut(Axis(2), iz))
            .and(&wkq)
            .apply(|w, wkq| *w += wkq);
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{array3_from_file, nhswps::Spectral},
        approx::assert_abs_diff_eq,
        byteorder::ByteOrder,
        lazy_static::lazy_static,
        ndarray::{Array3, ShapeBuilder},
    };

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let z = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_zy.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_r.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_w.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_aa.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_v.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/18_2_ds.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w,
                z,
                zx,
                zy,
                r,
                ri: Array3::<f64>::zeros((ng, ng, nz + 1)),
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ps: Array3::<f64>::zeros((ng, ng, nz + 1)),
                qs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ds,
                gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                t: 0.0,
                ngsave: 0,
                itime: 0,
                jtime: 0,
                ggen: false,
            }
        };
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let z = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_zy.bin");
            let r = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_r.bin");
            let w = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_w.bin");
            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_aa.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_v.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/vertical/32_4_ds.bin");
            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w,
                z,
                zx,
                zy,
                r,
                ri: Array3::<f64>::zeros((ng, ng, nz + 1)),
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ps: Array3::<f64>::zeros((ng, ng, nz + 1)),
                qs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ds,
                gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                t: 0.0,
                ngsave: 0,
                itime: 0,
                jtime: 0,
                ggen: false,
            }
        };
    }

    #[test]
    fn _18_2_z() {
        let z2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_z2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(z2, state.z, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_zx() {
        let zx2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_zx2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(zx2, state.zx, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_zy() {
        let zy2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_zy2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(zy2, state.zy, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_w() {
        let w2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_w2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(w2, state.w, epsilon = 1.0E-10);
    }

    #[test]
    fn _18_2_aa() {
        let aa2 = array3_from_file!(18, 18, 3, "testdata/vertical/18_2_aa2.bin");

        let mut state = STATE_18_2.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(&aa2, &state.aa, epsilon = 1.0E-10);
    }

    #[test]
    fn _32_4() {
        let z2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_z2.bin");
        let zx2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_zx2.bin");
        let zy2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_zy2.bin");
        let w2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_w2.bin");
        let aa2 = array3_from_file!(32, 32, 5, "testdata/vertical/32_4_aa2.bin");

        let mut state = STATE_32_4.clone();

        vertical(&mut state);

        assert_abs_diff_eq!(z2, state.z, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zx2, state.zx, epsilon = 1.0E-10);
        assert_abs_diff_eq!(zy2, state.zy, epsilon = 1.0E-10);
        assert_abs_diff_eq!(w2, state.w, epsilon = 1.0E-10);
        assert_abs_diff_eq!(aa2, state.aa, epsilon = 1.0E-10);
    }
}
