use {
    ndarray::{Array2, ShapeBuilder},
    std::f64::consts::PI,
};

pub fn init_pv_strip(ng: usize, width: f64, a2: f64, a3: f64) -> Array2<f64> {
    let ngu: usize = 16 * ng;
    let qmax: f64 = 4.0 * PI;

    let mut qod0 = vec![0f64; ngu / 2];
    let mut qod1 = vec![0f64; ngu / 2];
    let mut qod2 = vec![0f64; ngu / 2];

    let mut qev0 = vec![0f64; (ngu / 2) + 1];
    let mut qev1 = vec![0f64; (ngu / 2) + 1];
    let mut qev2 = vec![0f64; (ngu / 2) + 1];

    let mut qa = Array2::from_shape_vec((ngu, ngu), vec![0.0; ngu * ngu]).unwrap();
    let mut qq = Array2::from_shape_vec((ng, ng).strides((1, ng)), vec![0.0; ng * ng]).unwrap();

    let hwid = width / 2.0;

    let glu = 2.0 * PI / ngu as f64;

    for i in 0..ngu {
        let x = glu * i as f64 - PI;
        let y1 = -hwid;
        let y2 = hwid + a2 * (2.0 * x).sin() + a3 * (3.0 * x).sin();

        for j in 0..ngu {
            let y = glu * j as f64 - PI;

            qa[[j, i]] = if (y2 - y) * (y - y1) > 0.0 {
                4.0 * qmax * (y2 - y) * (y - y1) / (y2 - y1).powf(2.0)
            } else {
                0.0
            }
        }
    }

    // Average the PV field in qa to the coarser grid (ng,ng)
    let mut ngh = ngu;
    while ngh > ng {
        let nguf = ngh;
        ngh /= 2;

        // Perform nine-point averaging
        for iy in 0..ngh {
            let miy = 2 * (iy + 1);
            qod2[iy] = qa[[miy - 2, nguf - 1]];
            qev2[iy + 1] = qa[[miy - 1, nguf - 1]];
        }

        qev2[0] = qa[[nguf - 1, nguf - 1]];

        for ix in 0..ngh {
            let mix = 2 * (ix + 1);
            let mixm1 = mix - 1;

            for iy in 0..ngh {
                let miy = 2 * iy;

                qod1[iy] = qa[[miy, mixm1 - 1]];
                qod0[iy] = qa[[miy, mix - 1]];
                qev1[iy + 1] = qa[[miy + 1, mixm1 - 1]];
                qev0[iy + 1] = qa[[miy + 1, mix - 1]];
            }

            qev1[0] = qev1[ngh];
            qev0[0] = qev0[ngh];

            for iy in 0..ngh {
                qa[[iy, ix]] = 0.0625 * (qev0[iy + 1] + qev0[iy] + qev2[iy + 1] + qev2[iy])
                    + 0.125 * (qev1[iy + 1] + qev1[iy] + qod0[iy] + qod2[iy])
                    + 0.25 * qod1[iy]
            }

            qod2[..ngh].clone_from_slice(&qod0[..ngh]);
            qev2[1..=ngh].clone_from_slice(&qev0[1..=ngh]);

            qev2[0] = qev0[0];
        }
    }

    let mut qavg = 0f64;

    for ix in 0..ng {
        for iy in 0..ng {
            qavg += qa[[iy, ix]];
        }
    }

    qavg /= (ng * ng) as f64;

    for ix in 0..ng {
        for iy in 0..ng {
            qq[[iy, ix]] = qa[[iy, ix]] - qavg;
        }
    }

    qq
}

#[cfg(test)]
mod test {
    use {
        super::init_pv_strip,
        approx::assert_abs_diff_eq,
        byteorder::{ByteOrder, LittleEndian},
        ndarray::{Array2, ShapeBuilder},
    };

    /// Asserts that the generated .r8 file for ng=18 is close to the Fortran-created file.
    #[test]
    fn ng18_snapshot() {
        let qq2 = Array2::from_shape_vec(
            (18, 18).strides((1, 18)),
            include_bytes!("testdata/vstrip/qq_init_18.r8")
                .chunks(8)
                .skip(1)
                .map(LittleEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let qq = init_pv_strip(18, 0.4, 0.02, -0.01);

        assert_abs_diff_eq!(qq2, qq, epsilon = 1.0E-13);
    }

    /// Asserts that the generated .r8 file for ng=32 is close to the Fortran-created file.
    #[test]
    fn ng32_snapshot() {
        let qq2 = Array2::from_shape_vec(
            (32, 32).strides((1, 32)),
            include_bytes!("testdata/vstrip/qq_init_32.r8")
                .chunks(8)
                .skip(1)
                .map(LittleEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let qq = init_pv_strip(32, 0.4, 0.02, -0.01);

        assert_abs_diff_eq!(qq2, qq, epsilon = 1.0E-13);
    }

    /// Asserts that the generated .r8 file for ng=64 is close to the Fortran-created file.
    #[test]
    fn ng64_snapshot() {
        let qq2 = Array2::from_shape_vec(
            (64, 64).strides((1, 64)),
            include_bytes!("testdata/vstrip/qq_init_64.r8")
                .chunks(8)
                .skip(1)
                .map(LittleEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
        let qq = init_pv_strip(64, 0.4, 0.02, -0.01);

        assert_abs_diff_eq!(qq2, qq, epsilon = 1.0E-13);
    }
}
