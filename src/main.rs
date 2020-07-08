#[macro_use]
extern crate clap;

use {
    anyhow::{bail, Result},
    log::{error, info},
    shallow_water::{
        balinit::balinit, nhswps::nhswps, parameters::Parameters, swto3d::swto3d,
        vstrip::init_pv_strip,
    },
    simplelog::{Config as LogConfig, LevelFilter, TermLogger, TerminalMode},
    std::{
        fs::{create_dir, create_dir_all, File},
        io::prelude::*,
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
    let subcmd = match subcmd {
        Some(s) => s,
        None => bail!("No subcommand selected"),
    };

    create_dir_all(&params.environment.output_directory)?;

    info!("Starting {}", subcmd);

    match subcmd {
        "vstrip" => init_pv_strip(&params)?,
        "balinit" => balinit(&params)?,
        "swto3d" => swto3d(&params)?,
        "nhswps" => {
            let output = nhswps(&params)?;

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
