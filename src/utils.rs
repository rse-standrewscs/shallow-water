use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, ShapeBuilder};

#[cfg(test)]
use std::{fs::File, io::Read, path::Path};

pub fn view2d<T>(xs: &[T], x: usize, y: usize) -> ArrayView2<T> {
    ArrayView2::from_shape((x, y).strides((1, x)), xs).unwrap()
}

pub fn view3d<T>(xs: &[T], x: usize, y: usize, z: usize) -> ArrayView3<T> {
    ArrayView3::from_shape((x, y, z).strides((1, x, x * y)), xs).unwrap()
}

pub fn viewmut2d<T>(xs: &mut [T], x: usize, y: usize) -> ArrayViewMut2<T> {
    ArrayViewMut2::from_shape((x, y).strides((1, x)), xs).unwrap()
}

pub fn viewmut3d<T>(xs: &mut [T], x: usize, y: usize, z: usize) -> ArrayViewMut3<T> {
    ArrayViewMut3::from_shape((x, y, z).strides((1, x, x * y)), xs).unwrap()
}

pub fn arr2zero(ng: usize) -> Array2<f64> {
    Array2::<f64>::from_shape_vec((ng, ng).strides((1, ng)), vec![0.0; ng * ng]).unwrap()
}

pub fn arr3zero(ng: usize, nz: usize) -> Array3<f64> {
    Array3::<f64>::from_shape_vec(
        (ng, ng, nz + 1).strides((1, ng, ng * ng)),
        vec![0.0; ng * ng * (nz + 1)],
    )
    .unwrap()
}

#[cfg(test)]
pub(crate) fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
    for (i, e) in a.iter().enumerate() {
        approx::assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-13);
    }
}

#[cfg(test)]
pub(crate) fn assert_approx_eq_files<A: AsRef<Path>, B: AsRef<Path>>(correct: A, test: B) {
    let mut correct_file = File::open(correct).unwrap();
    let mut test_file = File::open(test).unwrap();

    for _ in (0..correct_file.metadata().unwrap().len()).step_by(8) {
        let mut correct_buf = [0; 8];
        let mut test_buf = [0; 8];

        correct_file.read_exact(&mut correct_buf).unwrap();
        test_file.read_exact(&mut test_buf).unwrap();

        approx::assert_abs_diff_eq!(
            f64::from_le_bytes(correct_buf),
            f64::from_le_bytes(test_buf),
            epsilon = 1.0E-13
        );
    }
}

#[cfg(test)]
pub(crate) fn assert_approx_eq_files_f32<A: AsRef<Path>, B: AsRef<Path>>(correct: A, test: B) {
    let mut correct_file = File::open(correct).unwrap();
    let mut test_file = File::open(test).unwrap();

    for _ in (0..correct_file.metadata().unwrap().len()).step_by(4) {
        let mut correct_buf = [0; 4];
        let mut test_buf = [0; 4];

        correct_file.read_exact(&mut correct_buf).unwrap();
        test_file.read_exact(&mut test_buf).unwrap();

        approx::assert_abs_diff_eq!(
            f32::from_le_bytes(correct_buf),
            f32::from_le_bytes(test_buf),
            epsilon = 1.0E-8
        );
    }
}

#[macro_export]
macro_rules! array2_from_file {
    ($x:expr, $y:expr, $name:expr) => {
        Array2::from_shape_vec(
            ($x, $y).strides((1, $x)),
            include_bytes!($name)
                .chunks(8)
                .map(byteorder::NetworkEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
    };
}

#[macro_export]
macro_rules! array3_from_file {
    ($x:expr, $y:expr, $z:expr, $name:expr) => {
        Array3::from_shape_vec(
            ($x, $y, $z).strides((1, $x, $x * $y)),
            include_bytes!($name)
                .chunks(8)
                .map(byteorder::NetworkEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
    };
}
