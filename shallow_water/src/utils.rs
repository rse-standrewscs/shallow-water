use ndarray::{
    Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis, ShapeBuilder,
};

pub fn view2d(xs: &[f64], x: usize, y: usize) -> ArrayView2<f64> {
    ArrayView2::from_shape((x, y).strides((1, x)), xs).unwrap()
}

pub fn view3d(xs: &[f64], x: usize, y: usize, z: usize) -> ArrayView3<f64> {
    ArrayView3::from_shape((x, y, z).strides((1, x, x * y)), xs).unwrap()
}

pub fn viewmut2d(xs: &mut [f64], x: usize, y: usize) -> ArrayViewMut2<f64> {
    ArrayViewMut2::from_shape((x, y).strides((1, x)), xs).unwrap()
}

pub fn viewmut3d(xs: &mut [f64], x: usize, y: usize, z: usize) -> ArrayViewMut3<f64> {
    ArrayViewMut3::from_shape((x, y, z).strides((1, x, x * y)), xs).unwrap()
}

pub fn slice_to_2d<T: Clone + Copy>(v: &[T], x: usize, y: usize) -> Vec<Vec<T>> {
    let mut out = vec![vec![v[0]; y]; x];

    for i in 0..x {
        for j in 0..y {
            let index = x * j + i;
            if index < v.len() {
                out[i][j] = v[x * j + i];
            }
        }
    }

    out
}

pub fn _2d_to_vec<T: Clone + Copy>(v: &[Vec<T>]) -> Vec<T> {
    let mut out = vec![v[0][0]; v.len() * v[0].len()];

    for i in 0..v.len() {
        for j in 0..v[0].len() {
            out[v.len() * j + i] = v[i][j];
        }
    }

    out
}

pub fn slice_to_3d<T: Clone + Copy>(v: &[T], x: usize, y: usize, z: usize) -> Vec<Vec<Vec<T>>> {
    let mut out = vec![vec![vec![v[0]; z]; y]; x];

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                let index = y * x * k + x * j + i;
                if index < v.len() {
                    out[i][j][k] = v[index];
                }
            }
        }
    }

    out
}

pub fn _3d_to_vec<T: Clone + Copy>(v: &[Vec<Vec<T>>]) -> Vec<T> {
    let x = v.len();
    let y = v[0].len();
    let z = v[0][0].len();
    let mut out = vec![v[0][0][0]; x * y * z];

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                out[y * x * k + x * j + i] = v[i][j][k];
            }
        }
    }

    out
}

pub fn slice_to_2d_nd(v: &[f64], x: usize, y: usize) -> Array2<f64> {
    let mut out = Array2::zeros((x, y));

    for i in 0..x {
        for j in 0..y {
            let index = x * j + i;
            if index < v.len() {
                out[[i, j]] = v[index];
            }
        }
    }

    out
}

pub fn slice_to_3d_nd(v: &[f64], x: usize, y: usize, z: usize) -> Array3<f64> {
    let mut out = Array3::zeros((x, y, z));

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                let index = y * x * k + x * j + i;
                if index < v.len() {
                    out[[i, j, k]] = v[index];
                }
            }
        }
    }

    out
}

pub fn _2d_to_vec_nd(v: &Array2<f64>) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];

    for i in 0..v.len_of(Axis(0)) {
        for j in 0..v.len_of(Axis(1)) {
            out[v.len() * j + i] = v[[i, j]];
        }
    }

    out
}

pub fn _3d_to_vec_nd(v: &Array3<f64>) -> Vec<f64> {
    let x = v.len_of(Axis(0));
    let y = v.len_of(Axis(1));
    let z = v.len_of(Axis(2));
    let mut out = vec![0.0; v.len()];

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                out[y * x * k + x * j + i] = v[[i, j, k]];
            }
        }
    }

    out
}

#[cfg(test)]
pub fn assert_approx_eq_slice(a: &[f64], b: &[f64]) {
    use approx::assert_abs_diff_eq;

    for (i, e) in a.iter().enumerate() {
        assert_abs_diff_eq!(*e, b[i], epsilon = 1.0E-13);
    }
}

#[cfg(test)]
mod test {
    use {super::*, insta::assert_debug_snapshot, quickcheck::quickcheck};

    quickcheck! {
        fn end_to_end_2d(xs: Vec<f64>) -> bool {
            for x in 0..xs.len() {
                for y in 0..xs.len() {
                    if x * y == xs.len() && xs != _2d_to_vec(&slice_to_2d(&xs, x, y)) {
                            return false;
                    }
                }
            }

            true
        }

        fn end_to_end_3d(xs: Vec<f64>) -> bool {
            for x in 0..xs.len() {
                for y in 0..xs.len() {
                    for z in 0..xs.len() {
                        if x * y * z == xs.len() && xs != _3d_to_vec(&slice_to_3d(&xs, x, y, z)) {
                            return false;
                        }
                    }
                }
            }

            true
        }
    }

    #[test]
    fn two_dim_48() {
        let a = vec![
            83.0, 82.0, 19.0, 26.0, 34.0, 29.0, 81.0, 93.0, 63.0, 75.0, 77.0, 75.0, 100.0, 85.0,
            21.0, 81.0, 22.0, 6.0, 71.0, 42.0, 81.0, 7.0, 66.0, 15.0, 18.0, 3.0, 37.0, 77.0, 61.0,
            57.0, 17.0, 55.0,
        ];

        assert_debug_snapshot!(slice_to_2d(&a, 4, 8));
    }

    #[test]
    fn three_dim_442() {
        let a = vec![
            83.0, 82.0, 19.0, 26.0, 34.0, 29.0, 81.0, 93.0, 63.0, 75.0, 77.0, 75.0, 100.0, 85.0,
            21.0, 81.0, 22.0, 6.0, 71.0, 42.0, 81.0, 7.0, 66.0, 15.0, 18.0, 3.0, 37.0, 77.0, 61.0,
            57.0, 17.0, 55.0,
        ];

        assert_debug_snapshot!(slice_to_3d(&a, 4, 4, 2));
    }

    #[test]
    fn three_dim_243() {
        let a = vec![
            83.0, 82.0, 19.0, 26.0, 34.0, 29.0, 81.0, 93.0, 63.0, 75.0, 77.0, 75.0, 100.0, 85.0,
            21.0, 81.0, 22.0, 6.0, 71.0, 42.0, 81.0, 7.0, 66.0, 15.0, 18.0, 3.0, 37.0, 77.0, 61.0,
            57.0, 17.0, 55.0,
        ];

        assert_debug_snapshot!(slice_to_3d(&a[8..], 2, 4, 3));
    }
}
