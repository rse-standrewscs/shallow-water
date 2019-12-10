#[macro_use]
extern crate clap;

mod constants;

use {
    byteorder::{ByteOrder, LittleEndian},
    constants::*,
    std::f64::consts::PI,
    std::fs::File,
    std::io::prelude::*,
    toml::Value,
};

fn main() {
    let matches = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "Initialises a PV strip with zero fields of divergence and acceleration divergence.")
        (@arg PARAMETERS_FILE: +required +takes_value "Path to file containing parameters to be used during PV strip initialization.")
    )
    .get_matches();

    let mut parameters_string = String::new();
    let mut f = File::open(matches.value_of("PARAMETERS_FILE").unwrap()).unwrap();
    f.read_to_string(&mut parameters_string).unwrap();
    let config: Value = toml::from_str(&parameters_string).unwrap();

    let qq = init_pv_strip(
        config["numerical"]["vertical_layer_count"]
            .as_integer()
            .unwrap() as usize,
        0.4,
        0.02,
        -0.01,
    );

    let mut f = File::create("qq_init.r8").unwrap();
    let mut buf = [0u8; 8];
    f.write_all(&buf).unwrap();
    for x in 0..qq.len() {
        for row in &qq {
            LittleEndian::write_f64(&mut buf, row[x]);
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

        for (j, row) in qa.iter_mut().enumerate() {
            let y = glu * j as f64 - PI;

            row[i] = if (y2 - y) * (y - y1) > 0.0 {
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

            qod2[..ngh].clone_from_slice(&qod0[..ngh]);
            qev2[1..=ngh].clone_from_slice(&qev0[1..=ngh]);

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
    use {
        super::init_pv_strip,
        byteorder::{ByteOrder, LittleEndian},
        insta::assert_debug_snapshot,
    };

    /// Generates a .r8 file from the initial parameters
    fn gen_r8(ng: usize, width: f64, a2: f64, a3: f64) -> Vec<u8> {
        let qq = init_pv_strip(ng, width, a2, a3);

        let mut data = Vec::<u8>::new();

        let mut buf = [0u8; 8];
        data.append(&mut buf.to_vec());

        for x in 0..qq.len() {
            qq.iter().for_each(|row| {
                LittleEndian::write_f64(&mut buf, row[x]);
                data.append(&mut buf.to_vec());
            });
        }

        data
    }

    /// Asserts that the generated .r8 file for ng=128 is equivalent to the snapshot of the Fortran-created file.
    #[test]
    fn ng128_snapshot() {
        assert_debug_snapshot!(gen_r8(128, 0.4, 0.02, -0.01));
    }

    /// Asserts that the generated .r8 file for ng=32 is equivalent to the snapshot of the Fortran-created file.
    #[test]
    fn ng32_snapshot() {
        assert_debug_snapshot!(gen_r8(32, 0.4, 0.02, -0.01));
    }
}
