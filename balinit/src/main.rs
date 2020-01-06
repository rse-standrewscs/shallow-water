#[macro_use]
extern crate clap;

use {
    byteorder::{ByteOrder, LittleEndian},
    libbalinit::balinit,
    std::{fs::File, io::prelude::*},
    toml::Value,
};

fn main() {
    let matches = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "Re-initialises a flow with balanced fields obtained from the conditions delta_t=gamma_t=0 using data previously set up with a data generation routine.")
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
