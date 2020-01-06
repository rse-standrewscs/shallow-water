#[macro_use]
extern crate clap;

use {
    byteorder::{ByteOrder, LittleEndian},
    libnhswps::nhswps,
    std::fs::File,
    std::io::prelude::*,
    toml::Value,
};

fn main() {
    let matches = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "The Horizontally Doubly-Periodic Three-Dimensional Non-Hydrostatic Shallow-Water Pseudo-Spectral Method")
        (@arg PARAMETERS_FILE: +takes_value "Path to file containing parameters to be used.")
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

    dbg!(output);
}
