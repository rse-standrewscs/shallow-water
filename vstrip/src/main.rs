#[macro_use]
extern crate clap;

mod constants;

use {
    byteorder::{ByteOrder, LittleEndian},
    constants::*,
    libvstrip::init_pv_strip,
    std::fs::File,
    std::io::prelude::*,
    toml::Value,
};

fn main() {
    let matches = clap_app!(vstrip =>
        (version: crate_version!())
        (about: "Initialises a PV strip with zero fields of divergence and acceleration divergence.")
        (@arg PARAMETERS_FILE: +takes_value "Path to file containing parameters to be used during PV strip initialization.")
    )
    .get_matches();

    let ng = {
        let mut parameters_string = String::new();
        let mut f = File::open(
            matches
                .value_of("PARAMETERS_FILE")
                .unwrap_or("parameters.toml"),
        )
        .unwrap();
        f.read_to_string(&mut parameters_string).unwrap();
        let config: Value = toml::from_str(&parameters_string).unwrap();

        config["numerical"]["inversion_grid_resolution"]
            .as_integer()
            .unwrap() as usize
    };

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
    f.write_all(&[0u8; NG * NG * 8]).unwrap();

    let mut f = File::create("gg_init.r8").unwrap();
    f.write_all(&[0u8; NG * NG * 8]).unwrap();
}
