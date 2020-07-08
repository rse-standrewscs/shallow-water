use {
    crate::{parameters::Parameters, utils::arr2zero},
    anyhow::Result,
    byteorder::{ByteOrder, LittleEndian},
    std::{f64::consts::PI, fs::File, io::Write},
};

pub fn init_pv_strip(parameters: &Parameters) -> Result<()> {
    let ng = parameters.numerical.grid_resolution;

    let ngu: usize = 16 * ng;
    let qmax: f64 = 4.0 * PI;

    let mut qod0 = vec![0f64; ngu / 2];
    let mut qod1 = vec![0f64; ngu / 2];
    let mut qod2 = vec![0f64; ngu / 2];

    let mut qev0 = vec![0f64; (ngu / 2) + 1];
    let mut qev1 = vec![0f64; (ngu / 2) + 1];
    let mut qev2 = vec![0f64; (ngu / 2) + 1];

    let mut qa = arr2zero(ngu);
    let mut qq = arr2zero(ng);

    let hwid = parameters.numerical.strip_width / 2.0;

    let glu = 2.0 * PI / ngu as f64;

    for i in 0..ngu {
        let x = glu * i as f64 - PI;
        let y1 = -hwid;
        let y2 = hwid
            + parameters.numerical.a2 * (2.0 * x).sin()
            + parameters.numerical.a3 * (3.0 * x).sin();

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

    let mut f = File::create(parameters.environment.output_directory.join("qq_init.r8"))?;
    let mut buf = [0u8; 8];
    f.write_all(&buf)?;
    for i in 0..ng {
        for j in 0..ng {
            LittleEndian::write_f64(&mut buf, qq[[j, i]]);
            f.write_all(&buf)?;
        }
    }

    let mut f = File::create(parameters.environment.output_directory.join("dd_init.r8"))?;
    f.write_all(&vec![0u8; ng * ng * 8])?;

    let mut f = File::create(parameters.environment.output_directory.join("gg_init.r8"))?;
    f.write_all(&vec![0u8; ng * ng * 8])?;

    Ok(())
}

#[cfg(test)]
mod test {
    use {super::*, std::io::Read, tempdir::TempDir};

    /// Asserts that the generated .r8 file for ng=18 is close to the Fortran-created file.
    #[test]
    fn ng18_snapshot() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.environment.output_directory = tempdir.path().to_owned();

        init_pv_strip(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("qq_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/vstrip/qq_init_18.r8")[i],
                byte.unwrap()
            );
        }
    }

    /// Asserts that the generated .r8 file for ng=32 is close to the Fortran-created file.
    #[test]
    fn ng32_snapshot() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 32;
        params.environment.output_directory = tempdir.path().to_owned();

        init_pv_strip(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("qq_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/vstrip/qq_init_32.r8")[i],
                byte.unwrap()
            );
        }
    }

    /// Asserts that the generated .r8 file for ng=64 is close to the Fortran-created file.
    #[test]
    fn ng64_snapshot() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 64;
        params.environment.output_directory = tempdir.path().to_owned();

        init_pv_strip(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("qq_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/vstrip/qq_init_64.r8")[i],
                byte.unwrap()
            );
        }
    }
}
