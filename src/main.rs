#[macro_use]
extern crate clap;

use {
    anyhow::{bail, Result},
    byteorder::{ByteOrder, LittleEndian},
    log::{error, info},
    shallow_water::{
        balinit::balinit, nhswps::nhswps, parameters::Parameters, swto3d::swto3d,
        vstrip::init_pv_strip,
    },
    simplelog::{Config as LogConfig, LevelFilter, TermLogger, TerminalMode},
    std::{
        fs::{create_dir, create_dir_all, File},
        io::{self, prelude::*},
    },
};

#[quit::main]
fn main() {
    let matches = clap_app!(shallow_water =>
        (version: crate_version!())
        (@arg PARAMETERS: -p --parameters +takes_value +required "Path to file containing simulation parameters.")
        (@subcommand vstrip =>
            (about: "Initialises a PV strip with zero fields of divergence and acceleration divergence.")
        )
        (@subcommand balinit =>
            (about: "Re-initialises a flow with balanced fields obtained from the conditions delta_t=gamma_t=0 using data previously set up with a data generation routine.")
        )
        (@subcommand swto3d =>
            (about: "Converts 2D shallow-water fields to 3D fields needed by nhswps.")
        )
        (@subcommand nhswps =>
            (about: "The Horizontally Doubly-Periodic Three-Dimensional Non-Hydrostatic Shallow-Water Pseudo-Spectral Method")
        )
    )
    .get_matches();

    TermLogger::init(
        LevelFilter::Debug,
        LogConfig::default(),
        TerminalMode::Mixed,
    )
    .expect("Failed to initialize logger");

    let params = {
        // Should never panic as clap should return an error if the argument was not supplied
        let path = matches
            .value_of("PARAMETERS")
            .expect("Path to parameters file not supplied");

        let file = File::open(path).unwrap_or_else(|e| {
            error!("Failed to open {}: \"{}\"", path, e);
            quit::with_code(1);
        });

        let params = serde_yaml::from_reader::<_, Parameters>(file).unwrap_or_else(|e| {
            error!("Failed to parse parameters from {}: \"{}\"", path, e);
            quit::with_code(1);
        });

        info!(
            "Successfully loaded simulation parameters from \"{}\": \n{:#?}",
            path, params
        );

        params
    };

    run_subcommand(matches.subcommand_name(), params).unwrap_or_else(|e| {
        error!("Error: \"{}\"", e);
        quit::with_code(1);
    });
}

fn write_file(path: &str, data: &[u8]) -> Result<()> {
    let mut f = File::create(path)?;
    f.write_all(data)?;
    Ok(())
}

fn run_subcommand(subcmd: Option<&str>, params: Parameters) -> Result<()> {
    let ng = params.numerical.grid_resolution;

    let subcmd = match subcmd {
        Some(s) => s,
        None => bail!("No subcommand selected"),
    };

    create_dir_all(&params.environment.output_directory)?;

    info!("Starting {}", subcmd);

    match subcmd {
        "vstrip" => {
            let qq = init_pv_strip(&params);

            let mut f = File::create("qq_init.r8")?;
            let mut buf = [0u8; 8];
            f.write_all(&buf)?;
            for i in 0..ng {
                for j in 0..ng {
                    LittleEndian::write_f64(&mut buf, qq[[j, i]]);
                    f.write_all(&buf)?;
                }
            }

            let mut f = File::create("dd_init.r8")?;
            f.write_all(&vec![0u8; ng * ng * 8])?;

            let mut f = File::create("gg_init.r8")?;
            f.write_all(&vec![0u8; ng * ng * 8])?;
        }
        "balinit" => {
            let zz = {
                let mut f = File::open("qq_init.r8")?;
                let mut zz = Vec::new();
                f.read_to_end(&mut zz)?;

                zz.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let (qq, dd, gg) = balinit(&zz, &params);

            let mut f = File::create("sw_init.r8")?;
            [vec![0.0], qq, vec![0.0], dd, vec![0.0], gg]
                .concat()
                .iter()
                .map(|x| {
                    let mut buf = [0u8; 8];
                    LittleEndian::write_f64(&mut buf, *x);
                    f.write_all(&buf)
                })
                .collect::<io::Result<()>>()?;
        }
        "swto3d" => {
            let split = {
                let mut f = File::open("sw_init.r8")?;
                let mut sw = Vec::new();
                f.read_to_end(&mut sw)?;

                let sw = sw
                    .chunks(8)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>();

                sw.chunks(ng * ng + 1)
                    .map(|xs| xs[1..].to_vec())
                    .collect::<Vec<Vec<f64>>>()
            };

            let (qq, dd, gg) = swto3d(&split[0], &split[1], &split[2], &params);

            let qq = {
                let mut bytes = vec![];
                qq.iter().for_each(|x| {
                    let mut buf = [0u8; 8];
                    LittleEndian::write_f64(&mut buf, *x);
                    bytes.extend_from_slice(&buf);
                });
                bytes
            };

            let dd = {
                let mut bytes = vec![];
                dd.iter().for_each(|x| {
                    let mut buf = [0u8; 8];
                    LittleEndian::write_f64(&mut buf, *x);
                    bytes.extend_from_slice(&buf);
                });
                bytes
            };

            let gg = {
                let mut bytes = vec![];
                gg.iter().for_each(|x| {
                    let mut buf = [0u8; 8];
                    LittleEndian::write_f64(&mut buf, *x);
                    bytes.extend_from_slice(&buf);
                });
                bytes
            };

            let mut qq_file = File::create("qq_init.r8")?;
            let mut dd_file = File::create("dd_init.r8")?;
            let mut gg_file = File::create("gg_init.r8")?;

            qq_file.write_all(&[0u8; 8])?;
            dd_file.write_all(&[0u8; 8])?;
            gg_file.write_all(&[0u8; 8])?;

            qq_file.write_all(&qq)?;
            dd_file.write_all(&dd)?;
            gg_file.write_all(&gg)?;
        }
        "nhswps" => {
            let qq = {
                let mut f = File::open("qq_init.r8")?;
                let mut qq = Vec::new();
                f.read_to_end(&mut qq)?;
                qq.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let dd = {
                let mut f = File::open("dd_init.r8")?;
                let mut dd = Vec::new();
                f.read_to_end(&mut dd)?;
                dd.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let gg = {
                let mut f = File::open("gg_init.r8")?;
                let mut gg = Vec::new();
                f.read_to_end(&mut gg)?;
                gg.chunks(8)
                    .skip(1)
                    .map(LittleEndian::read_f64)
                    .collect::<Vec<f64>>()
            };

            let output = nhswps(&qq, &dd, &gg, &params);

            write_file("monitor.asc", &output.monitor.as_bytes())?;
            write_file("ecomp.asc", &output.ecomp.as_bytes())?;
            write_file("spectra.asc", &output.spectra.as_bytes())?;

            create_dir("2d").ok();
            write_file("2d/d.r4", &output.d2d)?;
            write_file("2d/g.r4", &output.d2g)?;
            write_file("2d/h.r4", &output.d2h)?;
            write_file("2d/q.r4", &output.d2q)?;
            write_file("2d/zeta.r4", &output.d2zeta)?;

            create_dir("3d").ok();
            write_file("3d/d.r4", &output.d3d)?;
            write_file("3d/g.r4", &output.d3g)?;
            write_file("3d/pn.r4", &output.d3pn)?;
            write_file("3d/ql.r4", &output.d3ql)?;
            write_file("3d/r.r4", &output.d3r)?;
            write_file("3d/w.r4", &output.d3w)?;
        }
        _ => {
            // Should be unreachable due to clap catching this error
            bail!("Unrecognized subcommand");
        }
    }

    info!("Finished {}", subcmd);

    Ok(())
}
