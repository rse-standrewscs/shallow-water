#[macro_use]
extern crate clap;

use {
    anyhow::{bail, Result},
    log::{error, info},
    rayon::{current_num_threads, ThreadPoolBuilder},
    shallow_water::{
        balinit::balinit, nhswps::nhswps, parameters::Parameters, swto3d::swto3d,
        vstrip::init_pv_strip,
    },
    simplelog::{Config as LogConfig, LevelFilter, TermLogger, TerminalMode},
    std::fs::{create_dir_all, File},
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

fn run_subcommand(subcmd: Option<&str>, params: Parameters) -> Result<()> {
    let subcmd = match subcmd {
        Some(s) => s,
        None => bail!("No subcommand selected"),
    };

    // Recursively create output directiry and all parents
    create_dir_all(&params.environment.output_directory)?;
    info!("Writing data to {:?}", params.environment.output_directory);

    // Initialise global thread pool
    ThreadPoolBuilder::new()
        .num_threads(params.environment.threads)
        .build_global()?;
    info!(
        "Initialised global thread pool with {} threads",
        current_num_threads()
    );

    info!("Starting {}", subcmd);

    match subcmd {
        "vstrip" => init_pv_strip(&params)?,
        "balinit" => balinit(&params)?,
        "swto3d" => swto3d(&params)?,
        "nhswps" => nhswps(&params)?,
        _ => {
            // Should be unreachable due to clap catching this error
            bail!("Unrecognized subcommand");
        }
    }

    info!("Finished {}", subcmd);

    Ok(())
}
