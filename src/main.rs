#[macro_use]
extern crate clap;

use {
    byteorder::{ByteOrder, LittleEndian},
    log::error,
    shallow_water::{balinit::balinit, nhswps::nhswps, swto3d::swto3d, vstrip::init_pv_strip},
    simplelog::{Config as LogConfig, LevelFilter, TermLogger, TerminalMode},
    std::{
        fs::{create_dir, File},
        io::prelude::*,
    },
    toml::Value,
};

fn main() {
    TermLogger::init(LevelFilter::Info, LogConfig::default(), TerminalMode::Mixed)
        .expect("No interactive terminal");

    let matches = clap_app!(shallow_water =>
        (version: crate_version!())
        (@arg PARAMETERS_FILE: +takes_value "Path to file containing simulation parameters.")
        (@subcommand vstrip =>
            (version: crate_version!())
            (about: "Initialises a PV strip with zero fields of divergence and acceleration divergence.")
        )
        (@subcommand balinit =>
            (version: crate_version!())
            (about: "Re-initialises a flow with balanced fields obtained from the conditions delta_t=gamma_t=0 using data previously set up with a data generation routine.")
        )
        (@subcommand swto3d =>
            (version: crate_version!())
            (about: "Converts 2D shallow-water fields to 3D fields needed by nhswps.")
        )
        (@subcommand nhswps =>
            (version: crate_version!())
            (about: "The Horizontally Doubly-Periodic Three-Dimensional Non-Hydrostatic Shallow-Water Pseudo-Spectral Method")
        )
    )
    .get_matches();

    let config: Value = {
        let mut parameters_string = String::new();
        let mut f = File::open(
            matches
                .value_of("PARAMETERS_FILE")
                .unwrap_or("parameters.toml"),
        )
        .unwrap();
        f.read_to_string(&mut parameters_string).unwrap();
        toml::from_str(&parameters_string).unwrap()
    };

    let ng = config["numerical"]["inversion_grid_resolution"]
        .as_integer()
        .unwrap() as usize;
    let nz = config["numerical"]["vertical_layer_count"]
        .as_integer()
        .unwrap() as usize;

    match matches.subcommand_name() {
        Some("vstrip") => {
            let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);

            let mut f = File::create("qq_init.r8").unwrap();
            let mut buf = [0u8; 8];
            f.write_all(&buf).unwrap();
            for i in 0..ng {
                for j in 0..ng {
                    LittleEndian::write_f64(&mut buf, qq[[j, i]]);
                    f.write_all(&buf).unwrap();
                }
            }

            let mut f = File::create("dd_init.r8").unwrap();
            f.write_all(&vec![0u8; ng * ng * 8]).unwrap();

            let mut f = File::create("gg_init.r8").unwrap();
            f.write_all(&vec![0u8; ng * ng * 8]).unwrap();
        }
        Some("balinit") => {
            let zz = {
                let mut f = File::open("qq_init.r8").unwrap();
                let mut zz = Vec::new();
                f.read_to_end(&mut zz).unwrap();

                zz.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let (qq, dd, gg) = balinit(&zz, ng, nz);

            let mut f = File::create("sw_init.r8").unwrap();
            let mut buf = [0u8; 8];
            [vec![0.0], qq, vec![0.0], dd, vec![0.0], gg]
                .concat()
                .iter()
                .for_each(|&x| {
                    LittleEndian::write_f64(&mut buf, x);
                    f.write_all(&buf).unwrap();
                });
        }
        Some("swto3d") => {
            let split = {
                let mut f = File::open("sw_init.r8").unwrap();
                let mut sw = Vec::new();
                f.read_to_end(&mut sw).unwrap();

                let sw_init = sw
                    .chunks(8)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>();

                sw_init
                    .chunks(ng * ng + 1)
                    .map(|xs| xs[1..].to_vec())
                    .collect::<Vec<Vec<f64>>>()
            };

            let (qq, dd, gg) = swto3d(&split[0], &split[1], &split[2], ng, nz);

            let mut buf = [0u8; 8];

            {
                let mut f = File::create("qq_init.r8").unwrap();
                f.write_all(&[0u8; 8]).unwrap();
                qq.iter().for_each(|x| {
                    LittleEndian::write_f64(&mut buf, *x);
                    f.write_all(&buf).unwrap();
                });
            }

            {
                let mut f = File::create("dd_init.r8").unwrap();
                f.write_all(&[0u8; 8]).unwrap();
                dd.iter().for_each(|x| {
                    LittleEndian::write_f64(&mut buf, *x);
                    f.write_all(&buf).unwrap();
                });
            }

            {
                let mut f = File::create("gg_init.r8").unwrap();
                f.write_all(&[0u8; 8]).unwrap();
                gg.iter().for_each(|x| {
                    LittleEndian::write_f64(&mut buf, *x);
                    f.write_all(&buf).unwrap();
                });
            }
        }
        Some("nhswps") => {
            let qq = {
                let mut f = File::open("qq_init.r8").unwrap();
                let mut qq = Vec::new();
                f.read_to_end(&mut qq).unwrap();
                qq.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let dd = {
                let mut f = File::open("dd_init.r8").unwrap();
                let mut dd = Vec::new();
                f.read_to_end(&mut dd).unwrap();
                dd.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let gg = {
                let mut f = File::open("gg_init.r8").unwrap();
                let mut gg = Vec::new();
                f.read_to_end(&mut gg).unwrap();
                gg.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let output = nhswps(&qq, &dd, &gg, ng, nz);

            write_file("monitor.asc", &output.monitor.as_bytes());
            write_file("ecomp.asc", &output.ecomp.as_bytes());
            write_file("spectra.asc", &output.spectra.as_bytes());

            create_dir("2d").ok();
            write_file("2d/d.r4", &output.d2d);
            write_file("2d/g.r4", &output.d2g);
            write_file("2d/h.r4", &output.d2h);
            write_file("2d/q.r4", &output.d2q);
            write_file("2d/zeta.r4", &output.d2zeta);

            create_dir("3d").ok();
            write_file("3d/d.r4", &output.d3d);
            write_file("3d/g.r4", &output.d3g);
            write_file("3d/pn.r4", &output.d3pn);
            write_file("3d/ql.r4", &output.d3ql);
            write_file("3d/r.r4", &output.d3r);
            write_file("3d/w.r4", &output.d3w);
        }
        _ => {
            error!("Please select a subcommand!");
            std::process::exit(1);
        }
    }
}

fn write_file(path: &str, data: &[u8]) {
    let mut f = File::create(path).unwrap();
    f.write_all(data).unwrap();
}
