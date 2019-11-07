#[macro_use]
extern crate clap;

mod constants;

use {
    byteorder::{ByteOrder, LittleEndian},
    constants::*,
    std::f64::consts::PI,
    std::fs::File,
    std::io::prelude::*,
};

fn main() {
    let _ = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "Initialises a PV strip with zero fields of divergence and acceleration divergence.")
        //(@arg PARAMETERS_FILE: +required +takes_value "Path to file containing parameters to be used during PV strip initialization.")
    )
    .get_matches();

    let qq = init_pv_strip(NG, 0.4, 0.02, -0.01);

    let mut f = File::create("qq_init.r8").unwrap();
    let mut buf = [0u8; 8];
    f.write_all(&buf).unwrap();
    for col in qq {
        for row in col {
            LittleEndian::write_f64(&mut buf, row);
            f.write_all(&buf).unwrap();
        }
    }

    let mut f = File::create("dd_init.r8").unwrap();
    f.write_all(&[0u8; NG * NG * 8]).unwrap();

    let mut f = File::create("gg_init.r8").unwrap();
    f.write_all(&[0u8; NG * NG * 8]).unwrap();
}

fn init_pv_strip(ng: usize, width: f64, a2: f64, a3: f64) -> Vec<Vec<f64>> {
    let ngu: usize = 16 * ng;
    let qmax: f64 = 4.0 * PI;

    let mut qod0 = vec![0f64; ngu / 2];
    let mut qod1 = vec![0f64; ngu / 2];
    let mut qod2 = vec![0f64; ngu / 2];

    let mut qev0 = vec![0f64; (ngu / 2) + 1];
    let mut qev1 = vec![0f64; (ngu / 2) + 1];
    let mut qev2 = vec![0f64; (ngu / 2) + 1];

    let mut qa = vec![vec![0f64; ngu]; ngu];
    let mut qq = vec![vec![0f64; ng]; ng];

    let hwid = width / 2.0;

    let glu = 2.0 * PI / ngu as f64;

    for i in 0..ngu {
        let x = glu * i as f64 - PI;
        let y1 = -hwid;
        let y2 = hwid + a2 * (2.0 * x).sin() + a3 * (3.0 * x).sin();

        for j in 0..ngu {
            let y = glu * j as f64 - PI;
            if (y2 - y) * (y - y1) > 0.0 {
                qa[j][i] = 4.0 * qmax * (y2 - y) * (y - y1) / (y2 - y1).powf(2.0);
            } else {
                qa[j][i] = 0.0;
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
            qod2[iy] = qa[miy - 2][nguf - 1];
            qev2[iy + 1] = qa[miy - 1][nguf - 1];
        }

        qev2[0] = qa[nguf - 1][nguf - 1];

        for ix in 0..ngh {
            let mix = 2 * (ix + 1);
            let mixm1 = mix - 1;

            for iy in 0..ngh {
                let miy = 2 * iy;

                qod1[iy] = qa[miy][mixm1 - 1];
                qod0[iy] = qa[miy][mix - 1];
                qev1[iy + 1] = qa[miy + 1][mixm1 - 1];
                qev0[iy + 1] = qa[miy + 1][mix - 1];
            }

            qev1[0] = qev1[ngh];
            qev0[0] = qev0[ngh];

            for iy in 0..ngh {
                qa[iy][ix] = 0.0625 * (qev0[iy + 1] + qev0[iy] + qev2[iy + 1] + qev2[iy])
                    + 0.125 * (qev1[iy + 1] + qev1[iy] + qod0[iy] + qod2[iy])
                    + 0.25 * qod1[iy]
            }

            for iy in 0..ngh {
                qod2[iy] = qod0[iy];
                qev2[iy + 1] = qev0[iy + 1];
            }
            qev2[0] = qev0[0];
        }
    }

    let mut qavg = 0f64;
    for ix in 0..ng {
        for iy in 0..ng {
            qavg += qa[iy][ix];
        }
    }
    qavg /= (ng * ng) as f64;

    for ix in 0..ng {
        for iy in 0..ng {
            qq[iy][ix] = qa[iy][ix] - qavg;
        }
    }

    qq
}

#[cfg(test)]
mod test {
    use {super::init_pv_strip, insta::assert_debug_snapshot};

    #[test]
    fn test_snapshots() {
        assert_debug_snapshot!(init_pv_strip(32, 0.4, 0.02, -0.01));
    }
}
