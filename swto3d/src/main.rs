#[macro_use]
extern crate clap;

use {
    byteorder::{ByteOrder, LittleEndian},
    libswto3d::swto3d,
    simplelog::{Config as LogConfig, LevelFilter, TermLogger, TerminalMode},
    std::{fs::File, io::prelude::*},
    toml::Value,
};

fn main() {
    TermLogger::init(LevelFilter::Info, LogConfig::default(), TerminalMode::Mixed)
        .expect("No interactive terminal");

    let matches = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "Converts 2D shallow-water fields to 3D fields needed by nhswps.")
        (@arg PARAMETERS_FILE: +takes_value "Path to file containing parameters to be used during reinitialization.")
    )
    .get_matches();

    let (ng, nz) = {
        let mut parameters_string = String::new();
        let mut f = File::open(
            matches
                .value_of("PARAMETERS_FILE")
                .unwrap_or("parameters.toml"),
        )
        .unwrap();
        f.read_to_string(&mut parameters_string).unwrap();
        let config: Value = toml::from_str(&parameters_string).unwrap();

        (
            config["numerical"]["inversion_grid_resolution"]
                .as_integer()
                .unwrap() as usize,
            config["numerical"]["vertical_layer_count"]
                .as_integer()
                .unwrap() as usize,
        )
    };

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
