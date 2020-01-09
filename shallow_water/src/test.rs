use {libbalinit::balinit, libnhswps::nhswps, libswto3d::swto3d, libvstrip::init_pv_strip};

#[test]
fn complete_18_4() {
    let ng = 18;
    let nz = 4;
    let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);
    let (qq, dd, gg) = balinit(qq.as_slice_memory_order().unwrap(), ng, nz);
    let (qq, dd, gg) = swto3d(&qq, &dd, &gg, ng, nz);

    let output = nhswps(&qq, &dd, &gg, ng, nz);

    assert_eq!(
        include_str!("testdata/complete/18_4/monitor.asc")
            .to_string()
            .split_whitespace()
            .collect::<Vec<&str>>(),
        output
            .monitor
            .replace("\n", " ")
            .split(' ')
            .filter(|&s| !s.is_empty())
            .collect::<Vec<&str>>()
    );
    /*assert_eq!(
        include_str!("testdata/complete/18_4/ecomp.asc")
            .to_string()
            .split_whitespace()
            .collect::<Vec<&str>>(),
        output
            .ecomp
            .replace("\n", " ")
            .split(' ')
            .filter(|&s| !s.is_empty())
            .collect::<Vec<&str>>()
    );*/
    /* assert_eq!(
        include_str!("testdata/complete/18_4/spectra.asc")
            .to_string()
            .split_whitespace()
            .collect::<Vec<&str>>(),
        output
            .spectra
            .replace("\n", " ")
            .split(' ')
            .filter(|&s| !s.is_empty())
            .collect::<Vec<&str>>()
    );*/
}

#[test]
fn complete_32_4() {
    let ng = 32;
    let nz = 4;
    let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);
    let (qq, dd, gg) = balinit(qq.as_slice_memory_order().unwrap(), ng, nz);
    let (qq, dd, gg) = swto3d(&qq, &dd, &gg, ng, nz);

    let output = nhswps(&qq, &dd, &gg, ng, nz);

    assert_eq!(
        include_str!("testdata/complete/32_4/monitor.asc")
            .to_string()
            .split_whitespace()
            .collect::<Vec<&str>>(),
        output
            .monitor
            .replace("\n", " ")
            .split(' ')
            .filter(|&s| !s.is_empty())
            .collect::<Vec<&str>>()
    );
    //unimplemented!();
}
