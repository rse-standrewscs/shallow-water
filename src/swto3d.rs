pub fn swto3d(
    ql2d: &[f64],
    d2d: &[f64],
    g2d: &[f64],
    ng: usize,
    nz: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut gl = Vec::with_capacity(ng * ng * (nz + 1));
    let mut d = Vec::with_capacity(ng * ng * (nz + 1));
    let mut g = Vec::with_capacity(ng * ng * (nz + 1));

    for _ in 0..=nz {
        gl.extend_from_slice(ql2d);
        d.extend_from_slice(d2d);
        g.extend_from_slice(g2d);
    }

    (gl, d, g)
}

#[cfg(test)]
mod test {
    use {
        super::*,
        byteorder::{ByteOrder, LittleEndian},
    };

    #[test]
    fn _18_2_qq() {
        let sw_init = include_bytes!("testdata/swto3d/18_2_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let split = sw_init
            .chunks(18 * 18 + 1)
            .map(|xs| xs[1..].to_vec())
            .collect::<Vec<Vec<f64>>>();

        let qq2 = include_bytes!("testdata/swto3d/18_2_qq_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let (qq, _, _) = swto3d(&split[0], &split[1], &split[2], 18, 2);

        assert_eq!(qq2, qq);
    }

    #[test]
    fn _18_2_dd() {
        let sw_init = include_bytes!("testdata/swto3d/18_2_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let split = sw_init
            .chunks(18 * 18 + 1)
            .map(|xs| xs[1..].to_vec())
            .collect::<Vec<Vec<f64>>>();

        let dd2 = include_bytes!("testdata/swto3d/18_2_dd_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let (_, dd, _) = swto3d(&split[0], &split[1], &split[2], 18, 2);

        assert_eq!(dd2, dd);
    }

    #[test]
    fn _18_2_gg() {
        let sw_init = include_bytes!("testdata/swto3d/18_2_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let split = sw_init
            .chunks(18 * 18 + 1)
            .map(|xs| xs[1..].to_vec())
            .collect::<Vec<Vec<f64>>>();

        let gg2 = include_bytes!("testdata/swto3d/18_2_gg_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let (_, _, gg) = swto3d(&split[0], &split[1], &split[2], 18, 2);

        assert_eq!(gg2, gg);
    }

    #[test]
    fn _128_32() {
        let sw_init = include_bytes!("testdata/swto3d/128_32_sw_init.r8")
            .chunks(8)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let split = sw_init
            .chunks(128 * 128 + 1)
            .map(|xs| xs[1..].to_vec())
            .collect::<Vec<Vec<f64>>>();

        let qq2 = include_bytes!("testdata/swto3d/128_32_qq_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();
        let dd2 = include_bytes!("testdata/swto3d/128_32_dd_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();
        let gg2 = include_bytes!("testdata/swto3d/128_32_gg_init.r8")
            .chunks(8)
            .skip(1)
            .map(LittleEndian::read_f64)
            .collect::<Vec<f64>>();

        let (qq, dd, gg) = swto3d(&split[0], &split[1], &split[2], 128, 32);

        assert_eq!(qq2, qq);
        assert_eq!(dd2, dd);
        assert_eq!(gg2, gg);
    }
}
