//! Re-initialises a flow with balanced fields obtained from the conditions
//! delta_t=gamma_t=0 using data previously set up with a data generation
//! routine.  Assumes the previous data has delta = gamma = 0.
//!
//! Originally written 6/4/2018 by D G Dritschel @ St Andrews
//! Adapted for swnh to produce the linearised PV (q_l), the
//! divergence (delta) and the SW acceleration divergence (gamma).

use {
    crate::{constants::*, parameters::Parameters, spectral::Spectral, utils::*},
    log::info,
    ndarray::Zip,
};

pub fn balinit(zz: &[f64], parameters: &Parameters) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ng = parameters.numerical.grid_resolution;
    let nz = parameters.numerical.vertical_layers;

    let toler = 1.0E-9;

    let mut zz = zz.to_vec();
    let mut uu = vec![0.0; ng * ng];
    let mut vv = vec![0.0; ng * ng];

    let mut qq = zz.clone();
    let mut hh = vec![0.0; ng * ng];
    let mut dd = vec![0.0; ng * ng];
    let mut gg = vec![0.0; ng * ng];

    let mut wkp = vec![0.0; ng * ng];
    let mut wkq = vec![0.0; ng * ng];
    let mut htot = vec![0.0; ng * ng];

    let mut ds = vec![0.0; ng * ng];
    let mut gs = vec![0.0; ng * ng];

    let mut wka = vec![0.0; ng * ng];
    let mut wkb = vec![0.0; ng * ng];
    let mut wkc = vec![0.0; ng * ng];
    let mut wkd = vec![0.0; ng * ng];
    let mut wke = vec![0.0; ng * ng];

    let small: f64 = 1.0E-12;

    let mut qadd: f64;
    let mut qbar: f64;
    let mut fqbar: f64;
    let mut uio: f64;
    let mut vio: f64;
    let mut ddrmserr: f64;
    let mut ggrmserr: f64;
    let mut toterr: f64;
    let mut toterrpre: f64;

    let dsumi = 1.0 / (ng * ng) as f64;

    let spectral = Spectral::new(ng, nz);

    //Note: zz typically has zero domain average, whereas the actual
    //      PV anomaly may not since this is determined by the
    //      requirement that the mean relative vorticity is zero;
    //      qr is corrected upon calling main_invert in spectral.f90

    //Convert to spectral space (zz is overwritten; the PV is recovered below):
    /*ptospc(
        ng,
        ng,
        &mut zz,
        &mut qs,
        &spectral.xfactors,
        &spectral.yfactors,
        &spectral.xtrig,
        &spectral.ytrig,
    );*/

    //Find height anomaly field (hh):
    qadd = -dsumi * qq.iter().sum::<f64>();

    for e in qq.iter_mut() {
        *e += qadd;
    }

    qbar = dsumi * qq.iter().sum::<f64>();

    for (i, e) in qq.iter().enumerate() {
        wkp[i] = COF * e;
    }

    spectral.d2fft.ptospc(&mut wkp, &mut wkb);
    fqbar = COF * qbar;

    let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
    let wkb_matrix = view2d(&wkb, ng, ng);
    for i in 0..ng {
        for j in 0..ng {
            wka_matrix[[i, j]] =
                spectral.filt[[i, j]] * wkb_matrix[[i, j]] / (spectral.opak[[i, j]] - fqbar);
        }
    }

    spectral.d2fft.spctop(&mut wka, &mut hh);

    //wkp: corrected de-aliased height field (to be hh below)
    for (i, e) in hh.iter().enumerate() {
        htot[i] = 1.0 + e;
    }

    //Obtain relative vorticity field (zz):
    for (i, e) in qq.iter().enumerate() {
        wkp[i] = htot[i] * e;
    }
    qadd = -dsumi * wkp.iter().sum::<f64>();
    for e in qq.iter_mut() {
        *e += qadd;
    }
    for (i, e) in zz.iter_mut().enumerate() {
        *e = htot[i] * (COF + qq[i]) - COF;
    }

    //Obtain velocity field (uu,vv):
    spectral.d2fft.ptospc(&mut zz, &mut wkb);

    let mut wka_matrix = viewmut2d(&mut wka, ng, ng);
    let mut wkb_matrix = viewmut2d(&mut wkb, ng, ng);
    for i in 0..ng {
        for j in 0..ng {
            wka_matrix[[i, j]] = spectral.rlap[[i, j]] * wkb_matrix[[i, j]];
            wkb_matrix[[i, j]] *= spectral.filt[[i, j]];
        }
    }

    spectral.d2fft.spctop(&mut wkb, &mut zz);
    spectral.d2fft.xderiv(&spectral.hrkx, &wka, &mut wkd);
    spectral.d2fft.yderiv(&spectral.hrky, &wka, &mut wkb);
    for e in wkb.iter_mut() {
        *e = -*e;
    }
    spectral.d2fft.spctop(&mut wkb, &mut uu);
    spectral.d2fft.spctop(&mut wkd, &mut vv);
    //Add mean flow (uio,vio):
    uio = -hh.iter().zip(&uu).map(|(a, b)| a * b).sum::<f64>() * dsumi;
    for e in uu.iter_mut() {
        *e += uio;
    }

    //Iterate to find the balanced fields:

    //Energy norm error (must be > toler to start):
    toterrpre = 1.0 / small.powf(2.0);
    toterr = 1.0 / 2.0;
    let mut ddpre = dd.clone();
    let mut ggpre = gg.clone();

    while toterr > toler {
        //Obtain balanced estimate for gamma (gg):
        spectral.jacob(
            view2d(&uu, ng, ng),
            view2d(&vv, ng, ng),
            viewmut2d(&mut wkb, ng, ng),
        );

        for (i, e) in wkp.iter_mut().enumerate() {
            *e = dd[i] * uu[i];
        }
        for (i, e) in wkq.iter_mut().enumerate() {
            *e = dd[i] * vv[i];
        }
        spectral.divs(
            view2d(&wkp, ng, ng),
            view2d(&wkq, ng, ng),
            viewmut2d(&mut wka, ng, ng),
        );

        let wka_matrix = view2d(&wka, ng, ng);
        let wkb_matrix = view2d(&wkb, ng, ng);
        let gs_matrix = viewmut2d(&mut gs, ng, ng);

        Zip::from(gs_matrix)
            .and(&spectral.filt)
            .and(&wka_matrix)
            .and(&wkb_matrix)
            .apply(|gs, filt, wka, wkb| *gs = filt * (wka - 2.0 * wkb));

        wka = gs.clone();
        spectral.d2fft.spctop(&mut wka, &mut gg);

        gg.iter()
            .zip(&ggpre)
            .map(|(a, b)| (a - b).powf(2.0))
            .sum::<f64>();
        ggrmserr = gg
            .iter()
            .zip(&ggpre)
            .map(|(a, b)| (a - b).powf(2.0))
            .sum::<f64>()
            / (ggpre.iter().map(|x| x.powf(2.0)).sum::<f64>() + small);

        //Obtain balanced estimate for delta (dd):
        for (i, e) in wkp.iter_mut().enumerate() {
            *e = hh[i] * uu[i];
        }
        for (i, e) in wkq.iter_mut().enumerate() {
            *e = hh[i] * vv[i];
        }
        spectral.divs(
            view2d(&wkp, ng, ng),
            view2d(&wkq, ng, ng),
            viewmut2d(&mut wka, ng, ng),
        );
        for (i, e) in wkp.iter_mut().enumerate() {
            *e = zz[i] * uu[i];
        }
        for (i, e) in wkq.iter_mut().enumerate() {
            *e = zz[i] * vv[i];
        }
        spectral.divs(
            view2d(&wkp, ng, ng),
            view2d(&wkq, ng, ng),
            viewmut2d(&mut wkb, ng, ng),
        );

        let wka_matrix = view2d(&wka, ng, ng);
        let wkb_matrix = view2d(&wkb, ng, ng);
        let ds_matrix = viewmut2d(&mut ds, ng, ng);

        Zip::from(ds_matrix)
            .and(&spectral.helm)
            .and(&wkb_matrix)
            .and(&spectral.c2g2)
            .and(&wka_matrix)
            .apply(|ds, helm, wkb, c2g2, wka| *ds = helm * (COF * wkb - c2g2 * wka));

        wka = ds.clone();
        spectral.d2fft.spctop(&mut wka, &mut dd);
        ddrmserr = dd
            .iter()
            .zip(&ddpre)
            .map(|(a, b)| (a - b).powf(2.0))
            .sum::<f64>()
            / (ddpre.iter().map(|x| x.powf(2.0)).sum::<f64>() + small);

        //Find height anomaly field (hh):
        for (i, e) in htot.iter_mut().enumerate() {
            *e = 1.0 + hh[i];
        }
        qadd = -dsumi * qq.iter().zip(&htot).map(|(a, b)| a * b).sum::<f64>();
        for e in qq.iter_mut() {
            *e += qadd;
        }
        qbar = dsumi * qq.iter().sum::<f64>();
        for (i, e) in wkp.iter_mut().enumerate() {
            *e = COF * (qq[i] + hh[i] * (qq[i] - qbar)) - gg[i];
        }
        spectral.d2fft.ptospc(&mut wkp, &mut wkb);
        fqbar = COF * qbar;

        let wka_matrix = viewmut2d(&mut wka, ng, ng);
        let wkb_matrix = view2d(&wkb, ng, ng);

        Zip::from(wka_matrix)
            .and(&spectral.filt)
            .and(&wkb_matrix)
            .and(&spectral.opak)
            .apply(|wka, filt, wkb, opak| *wka = filt * wkb / (opak - fqbar));

        spectral.d2fft.spctop(&mut wka, &mut hh);
        //wkp: corrected de-aliased height field (to be hh below)
        for (i, e) in htot.iter_mut().enumerate() {
            *e = 1.0 + hh[i];
        }

        //Obtain relative vorticity field (zz):
        for (i, e) in wkp.iter_mut().enumerate() {
            *e = qq[i] * htot[i];
        }
        qadd = -dsumi * wkp.iter().sum::<f64>();
        for e in qq.iter_mut() {
            *e += qadd;
        }

        for (i, e) in zz.iter_mut().enumerate() {
            *e = htot[i] * (COF + qq[i]) - COF;
        }

        //Obtain velocity field (uu,vv):
        spectral.d2fft.ptospc(&mut zz, &mut wkb);

        let wka_matrix = viewmut2d(&mut wka, ng, ng);
        let wkb_matrix = viewmut2d(&mut wkb, ng, ng);
        Zip::from(wka_matrix)
            .and(&spectral.rlap)
            .and(&wkb_matrix)
            .apply(|wka, rlap, wkb| *wka = rlap * wkb);
        Zip::from(wkb_matrix)
            .and(&spectral.filt)
            .apply(|wkb, filt| *wkb *= filt);

        spectral.d2fft.spctop(&mut wkb, &mut zz);
        spectral.d2fft.xderiv(&spectral.hrkx, &wka, &mut wkd);
        spectral.d2fft.yderiv(&spectral.hrky, &wka, &mut wkb);

        let wke_matrix = viewmut2d(&mut wke, ng, ng);
        let ds_matrix = view2d(&ds, ng, ng);
        Zip::from(wke_matrix)
            .and(&spectral.rlap)
            .and(&ds_matrix)
            .apply(|wke, rlap, ds| *wke = rlap * ds);

        spectral.d2fft.xderiv(&spectral.hrkx, &wke, &mut wka);
        spectral.d2fft.yderiv(&spectral.hrky, &wke, &mut wkc);

        for (i, e) in wkb.iter_mut().enumerate() {
            *e = wka[i] - *e;
        }
        for (i, e) in wkd.iter_mut().enumerate() {
            *e += wkc[i];
        }

        spectral.d2fft.spctop(&mut wkb, &mut uu);
        spectral.d2fft.spctop(&mut wkd, &mut vv);

        //Add mean flow (uio,vio):
        uio = -hh.iter().zip(&uu).map(|(a, b)| a * b).sum::<f64>() * dsumi;
        vio = -hh.iter().zip(&vv).map(|(a, b)| a * b).sum::<f64>() * dsumi;
        for e in uu.iter_mut() {
            *e += uio;
        }
        for e in vv.iter_mut() {
            *e += vio;
        }

        //Compute overall error:
        toterr = (1.0 / 2.0) * (ddrmserr + ggrmserr);

        info!("relative delta error = {}", ddrmserr);
        info!("relative gamma error = {}", ggrmserr);

        //If error is going up again, stop and save fields:
        if toterrpre <= toterr {
            break;
        }

        //Otherwise continue with another iteration:
        ddpre = dd.clone();
        ggpre = gg.clone();
        toterrpre = toterr;
    }

    info!("Minimum error = {}", toterrpre);

    for (i, e) in qq.iter_mut().enumerate() {
        *e = zz[i] - COF * hh[i];
    }

    (qq, dd, gg)
}

#[cfg(test)]
mod test {
    use {
        super::*,
        approx::assert_abs_diff_eq,
        byteorder::{ByteOrder, LittleEndian, NetworkEndian},
    };

    fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
        for (i, e) in a.iter().enumerate() {
            assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-13);
        }
    }

    #[test]
    fn balinit_18_2_qq() {
        let zz = include_bytes!("testdata/balinit/18_2_zz.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let qq2 = include_bytes!("testdata/balinit/18_2_qq.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;

        let (qq, _, _) = balinit(&zz, &params);

        assert_approx_eq_slice(&qq2, &qq);
    }

    #[test]
    fn balinit_18_2_dd() {
        let zz = include_bytes!("testdata/balinit/18_2_zz.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let dd2 = include_bytes!("testdata/balinit/18_2_dd.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;

        let (_, dd, __) = balinit(&zz, &params);

        assert_approx_eq_slice(&dd2, &dd);
    }

    #[test]
    fn balinit_18_2_gg() {
        let zz = include_bytes!("testdata/balinit/18_2_zz.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();
        let gg2 = include_bytes!("testdata/balinit/18_2_gg.bin")
            .chunks(8)
            .map(NetworkEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;

        let (_, __, gg) = balinit(&zz, &params);

        assert_approx_eq_slice(&gg2, &gg);
    }

    #[test]
    fn end2end_18_2() {
        let qq_init = include_bytes!("testdata/balinit/18_2_qq_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let sw_init = include_bytes!("testdata/balinit/18_2_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;

        let (qq, dd, gg) = balinit(&qq_init, &params);

        assert_approx_eq_slice(
            &sw_init,
            &[vec![0.0], qq, vec![0.0], dd, vec![0.0], gg].concat(),
        );
    }

    #[test]
    fn end2end_32_4() {
        let qq_init = include_bytes!("testdata/balinit/32_4_qq_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();
        let sw_init = include_bytes!("testdata/balinit/32_4_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 32;
        params.numerical.vertical_layers = 4;

        let (qq, dd, gg) = balinit(&qq_init, &params);

        assert_approx_eq_slice(
            &sw_init,
            &[vec![0.0], qq, vec![0.0], dd, vec![0.0], gg].concat(),
        );
    }
}
