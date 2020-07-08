use {
    crate::parameters::Parameters,
    anyhow::Result,
    std::{
        fs::File,
        io::{Read, Write},
    },
};

pub fn swto3d(parameters: &Parameters) -> Result<()> {
    let ng = parameters.numerical.grid_resolution;
    let nz = parameters.numerical.vertical_layers;

    let split = {
        let mut f = File::open(parameters.environment.output_directory.join("sw_init.r8"))?;
        let mut sw = Vec::new();
        f.read_to_end(&mut sw)?;

        sw.chunks((ng * ng + 1) * 8)
            .map(|xs| xs[8..].to_vec())
            .collect::<Vec<Vec<u8>>>()
    };

    let mut qq_file = File::create(parameters.environment.output_directory.join("qq_init.r8"))?;
    let mut dd_file = File::create(parameters.environment.output_directory.join("dd_init.r8"))?;
    let mut gg_file = File::create(parameters.environment.output_directory.join("gg_init.r8"))?;

    qq_file.write_all(&[0u8; 8])?;
    dd_file.write_all(&[0u8; 8])?;
    gg_file.write_all(&[0u8; 8])?;

    for _ in 0..=nz {
        qq_file.write_all(&split[0])?;
        dd_file.write_all(&split[1])?;
        gg_file.write_all(&split[2])?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use {super::*, tempdir::TempDir};

    #[test]
    fn _18_2_qq() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut sw = File::create(tempdir.path().join("sw_init.r8")).unwrap();
        sw.write_all(include_bytes!("testdata/swto3d/18_2_sw_init.r8"))
            .unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;
        params.environment.output_directory = tempdir.path().to_owned();

        swto3d(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("qq_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/18_2_qq_init.r8")[i],
                byte.unwrap()
            );
        }
    }

    #[test]
    fn _18_2_dd() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut sw = File::create(tempdir.path().join("sw_init.r8")).unwrap();
        sw.write_all(include_bytes!("testdata/swto3d/18_2_sw_init.r8"))
            .unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;
        params.environment.output_directory = tempdir.path().to_owned();

        swto3d(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("dd_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/18_2_dd_init.r8")[i],
                byte.unwrap()
            );
        }
    }

    #[test]
    fn _18_2_gg() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut sw = File::create(tempdir.path().join("sw_init.r8")).unwrap();
        sw.write_all(include_bytes!("testdata/swto3d/18_2_sw_init.r8"))
            .unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 18;
        params.numerical.vertical_layers = 2;
        params.environment.output_directory = tempdir.path().to_owned();

        swto3d(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("gg_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/18_2_gg_init.r8")[i],
                byte.unwrap()
            );
        }
    }

    #[test]
    fn _128_32() {
        let tempdir = TempDir::new("shallow-water").unwrap();

        let mut sw = File::create(tempdir.path().join("sw_init.r8")).unwrap();
        sw.write_all(include_bytes!("testdata/swto3d/128_32_sw_init.r8"))
            .unwrap();

        let mut params = Parameters::default();
        params.numerical.grid_resolution = 128;
        params.numerical.vertical_layers = 32;
        params.environment.output_directory = tempdir.path().to_owned();

        swto3d(&params).unwrap();

        for (i, byte) in File::open(params.environment.output_directory.join("qq_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/128_32_qq_init.r8")[i],
                byte.unwrap()
            );
        }

        for (i, byte) in File::open(params.environment.output_directory.join("dd_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/128_32_dd_init.r8")[i],
                byte.unwrap()
            );
        }

        for (i, byte) in File::open(params.environment.output_directory.join("gg_init.r8"))
            .unwrap()
            .bytes()
            .enumerate()
        {
            assert_eq!(
                include_bytes!("testdata/swto3d/128_32_gg_init.r8")[i],
                byte.unwrap()
            );
        }
    }
}
