use {
    crate::{constants::*, nhswps::State, utils::*},
    ndarray::{ArrayViewMut3, Axis, Zip},
    rayon::prelude::*,
};

/// Gets the nonlinear source terms for linearised PV, divergence and
/// acceleration divergence  --- all in spectral space.  These are
/// returned in sqs, sds and sgs respectively.
///
/// Note that (sds,sgs) only include the nonlinear terms for a
/// semi-implicit treatment, closely analogous to that described in
/// the appendix of Mohebalhojeh & Dritschel (2004).
///
/// The spectral fields qs, ds and gs are all spectrally truncated.
/// Note: u, v & zeta obtained by main_invert, and z obtained by psolve
/// (which calls vertical) before calling this routine are all
/// spectrally truncated.
pub fn source(
    state: &State,
    mut sqs: ArrayViewMut3<f64>,
    mut sds: ArrayViewMut3<f64>,
    mut sgs: ArrayViewMut3<f64>,
) {
    let ng = state.spectral.ng;
    let nz = state.spectral.nz;

    let mut wkd = arr2zero(ng);

    //Calculate vertically-independent part of gs source (wkd):;
    for iz in 0..=nz {
        Zip::from(&mut wkd)
            .and(state.aa.index_axis(Axis(2), iz))
            .apply(|wkd, aa| *wkd += state.spectral.weight[iz] * aa);
    }

    //Note: aa contains div(u*rho_theta) in spectral space
    wkd *= &state.spectral.c2g2;

    sqs.axis_iter_mut(Axis(2))
        .into_par_iter()
        .zip(sds.axis_iter_mut(Axis(2)).into_par_iter())
        .zip(sgs.axis_iter_mut(Axis(2)).into_par_iter())
        .zip(0..=(nz as u16))
        .for_each(|(((mut sqs_slice, sds_slice), sgs_slice), iz)| {
            let iz = iz as usize;
            let mut dd = arr2zero(ng);
            let mut ff = arr2zero(ng);

            let mut wkp = arr2zero(ng);
            let mut wkq = arr2zero(ng);

            let mut wka = arr2zero(ng);
            let mut wkb = arr2zero(ng);
            let mut wkc = arr2zero(ng);

            // qs source:

            // Compute div(ql*u,ql*v) (wka in spectral space):
            wka.assign(&state.qs.index_axis(Axis(2), iz));

            state.spectral.d2fft.spctop(wka.view_mut(), wkq.view_mut());
            // wkq contains the linearised PV in physical space

            wkp.assign(&(&wkq * &state.u.index_axis(Axis(2), iz)));
            wkq *= &state.v.index_axis(Axis(2), iz);

            // Compute spectral divergence from physical fields:
            state.spectral.divs(wkp.view(), wkq.view(), wka.view_mut());

            // Compute Jacobian of F = (1/rho_theta)*dP'/dtheta & z (wkb, spectral):
            ff.assign(&(&state.ri.index_axis(Axis(2), iz) * &state.dpn.index_axis(Axis(2), iz)));

            state.spectral.deal2d(ff.view_mut());

            wkq.assign(&state.z.index_axis(Axis(2), iz));

            state.spectral.jacob(ff.view(), wkq.view(), wkb.view_mut());

            // Sum to get qs source:
            {
                Zip::from(&mut sqs_slice)
                    .and(&state.spectral.filt)
                    .and(&wkb)
                    .and(&wka)
                    .apply(|sqs, filt, wkb, wka| *sqs = filt * (wkb - wka));
            }

            // Nonlinear part of ds source:

            // Compute J(u,v) (wkc in spectral space):
            state.spectral.jacob(
                state.u.index_axis(Axis(2), iz),
                state.v.index_axis(Axis(2), iz),
                wkc.view_mut(),
            );

            // Convert ds to physical space as dd:
            wka.assign(&state.ds.index_axis(Axis(2), iz));

            state.spectral.d2fft.spctop(wka.view_mut(), dd.view_mut());

            // Compute div(F*grad{z}-delta*{u,v}) (wkb in spectral space):
            Zip::from(&mut wkp)
                .and(&ff)
                .and(state.zx.index_axis(Axis(2), iz))
                .and(&dd)
                .and(state.u.index_axis(Axis(2), iz))
                .apply(|wkp, ff, zx, dd, u| *wkp = ff * zx - dd * u);

            Zip::from(&mut wkq)
                .and(&ff)
                .and(state.zy.index_axis(Axis(2), iz))
                .and(&dd)
                .and(state.v.index_axis(Axis(2), iz))
                .apply(|wkq, ff, zy, dd, v| *wkq = ff * zy - dd * v);

            state.spectral.divs(wkp.view(), wkq.view(), wkb.view_mut());

            // Add Lap(P') and complete definition of ds source:
            Zip::from(sds_slice)
                .and(&state.spectral.filt)
                .and(&wkc)
                .and(&wkb)
                .and(&state.spectral.hlap)
                .and(state.ps.index_axis(Axis(2), iz))
                .apply(|sds, filt, wkc, wkb, hlap, ps| *sds = filt * (2.0 * wkc + wkb - hlap * ps));

            // Nonlinear part of gs source:
            Zip::from(sgs_slice)
                .and(sqs_slice)
                .and(&wkd)
                .and(state.aa.index_axis(Axis(2), iz))
                .apply(|sgs, sqs, wkd, aa| *sgs = COF * *sqs + wkd - FSQ * aa);
        });
}

#[cfg(test)]
mod test {
    use {
        super::*,
        crate::{
            array3_from_file,
            nhswps::{Output, Spectral},
        },
        byteorder::{ByteOrder, NetworkEndian},
        lazy_static::lazy_static,
        ndarray::{Array3, ShapeBuilder},
    };

    lazy_static! {
        static ref STATE_18_2: State = {
            let ng = 18;
            let nz = 2;

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/18_2_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn,
                ps,
                qs,
                ds,
                gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                t: 0.0,
                ngsave: 0,
                itime: 0,
                jtime: 0,
                ggen: false,
                output: Output::default(),
            }
        };
        static ref STATE_32_4: State = {
            let ng = 32;
            let nz = 4;

            let aa = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_aa.bin");
            let qs = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_qs.bin");
            let ds = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ds.bin");
            let ps = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ps.bin");
            let u = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_u.bin");
            let v = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_v.bin");
            let ri = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_ri.bin");
            let dpn = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_dpn.bin");
            let z = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_z.bin");
            let zx = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zx.bin");
            let zy = array3_from_file!(ng, ng, nz + 1, "testdata/source/32_4_zy.bin");

            State {
                spectral: Spectral::new(ng, nz),
                u,
                v,
                w: Array3::<f64>::zeros((ng, ng, nz + 1)),
                z,
                zx,
                zy,
                r: Array3::<f64>::zeros((ng, ng, nz + 1)),
                ri,
                aa,
                zeta: Array3::<f64>::zeros((ng, ng, nz + 1)),
                pn: Array3::<f64>::zeros((ng, ng, nz + 1)),
                dpn,
                ps,
                qs,
                ds,
                gs: Array3::<f64>::zeros((ng, ng, nz + 1)),
                t: 0.0,
                ngsave: 0,
                itime: 0,
                jtime: 0,
                ggen: false,
                output: Output::default(),
            }
        };
    }

    #[test]
    fn _18_2_sqs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/18_2_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_18_2,
            viewmut3d(&mut sqs, 18, 18, 3),
            viewmut3d(&mut sds, 18, 18, 3),
            viewmut3d(&mut sgs, 18, 18, 3),
        );

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _18_2_sds() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/18_2_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_18_2,
            viewmut3d(&mut sqs, 18, 18, 3),
            viewmut3d(&mut sds, 18, 18, 3),
            viewmut3d(&mut sgs, 18, 18, 3),
        );

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _18_2_sgs() {
        let mut sqs = include_bytes!("testdata/source/18_2_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/18_2_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/18_2_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/18_2_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_18_2,
            viewmut3d(&mut sqs, 18, 18, 3),
            viewmut3d(&mut sds, 18, 18, 3),
            viewmut3d(&mut sgs, 18, 18, 3),
        );

        assert_approx_eq_slice(&sgs2, &sgs);
    }

    #[test]
    fn _32_4_sqs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sqs2 = include_bytes!("testdata/source/32_4_sqs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_32_4,
            viewmut3d(&mut sqs, 32, 32, 5),
            viewmut3d(&mut sds, 32, 32, 5),
            viewmut3d(&mut sgs, 32, 32, 5),
        );

        assert_approx_eq_slice(&sqs2, &sqs);
    }

    #[test]
    fn _32_4_sds() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sds2 = include_bytes!("testdata/source/32_4_sds2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_32_4,
            viewmut3d(&mut sqs, 32, 32, 5),
            viewmut3d(&mut sds, 32, 32, 5),
            viewmut3d(&mut sgs, 32, 32, 5),
        );

        assert_approx_eq_slice(&sds2, &sds);
    }

    #[test]
    fn _32_4_sgs() {
        let mut sqs = include_bytes!("testdata/source/32_4_sqs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sds = include_bytes!("testdata/source/32_4_sds.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let mut sgs = include_bytes!("testdata/source/32_4_sgs.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let sgs2 = include_bytes!("testdata/source/32_4_sgs2.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        source(
            &STATE_32_4,
            viewmut3d(&mut sqs, 32, 32, 5),
            viewmut3d(&mut sds, 32, 32, 5),
            viewmut3d(&mut sgs, 32, 32, 5),
        );
        assert_approx_eq_slice(&sgs2, &sgs);
    }
}
