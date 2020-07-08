use {
    criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion},
    shallow_water::{
        balinit::balinit, parameters::Parameters, swto3d::swto3d, vstrip::init_pv_strip,
    },
    tempdir::TempDir,
};

fn _2d_to_vec<T: Clone + Copy>(v: &[Vec<T>]) -> Vec<T> {
    let mut out = vec![v[0][0]; v.len() * v[0].len()];

    for i in 0..v.len() {
        for j in 0..v[0].len() {
            out[v.len() * j + i] = v[i][j];
        }
    }

    out
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "setup",
        Benchmark::new("ng16_nz4", |b| {
            let tempdir = TempDir::new("shallow-water").unwrap();

            let mut params = Parameters::default();
            params.numerical.grid_resolution = 16;
            params.numerical.vertical_layers = 4;
            params.environment.output_directory = tempdir.path().to_owned();

            b.iter(|| {
                let params = black_box(&params);
                init_pv_strip(&params).unwrap();
                balinit(&params).unwrap();
                swto3d(&params).unwrap();
            })
        })
        .sample_size(20),
    );

    c.bench(
        "setup",
        Benchmark::new("ng32_nz4", |b| {
            let tempdir = TempDir::new("shallow-water").unwrap();

            let mut params = Parameters::default();
            params.numerical.grid_resolution = 32;
            params.numerical.vertical_layers = 4;
            params.environment.output_directory = tempdir.path().to_owned();

            b.iter(|| {
                let params = black_box(&params);
                init_pv_strip(&params).unwrap();
                balinit(&params).unwrap();
                swto3d(&params).unwrap();
            })
        })
        .sample_size(10),
    );

    c.bench(
        "setup",
        Benchmark::new("ng64_nz16", |b| {
            let tempdir = TempDir::new("shallow-water").unwrap();

            let mut params = Parameters::default();
            params.numerical.grid_resolution = 64;
            params.numerical.vertical_layers = 16;
            params.environment.output_directory = tempdir.path().to_owned();

            b.iter(|| {
                let params = black_box(&params);
                init_pv_strip(&params).unwrap();
                balinit(&params).unwrap();
                swto3d(&params).unwrap();
            })
        })
        .sample_size(10),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
