use ndarray::{ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, ShapeBuilder};

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

#[cfg(test)]
pub fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
    use approx::assert_abs_diff_eq;

    for (i, e) in a.iter().enumerate() {
        assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-13);
    }
}

#[macro_export]
macro_rules! array3_from_file {
    ($x:expr, $y:expr, $z:expr, $name:expr) => {
        Array3::from_shape_vec(
            ($x, $y, $z).strides((1, $x, $x * $y)),
            include_bytes!($name)
                .chunks(8)
                .map(NetworkEndian::read_f64)
                .collect::<Vec<f64>>(),
        )
        .unwrap();
    };
}
