use {
    criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion},
    libbalinit::balinit,
    libswto3d::swto3d,
    libvstrip::init_pv_strip,
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
        "full_init",
        Benchmark::new("ng16_nz4", |b| {
            b.iter(|| {
                let ng = black_box(16);
                let nz = black_box(4);
                let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);
                let (qq, dd, gg) = balinit(qq.as_slice().unwrap(), ng, nz);
                let (_, _, _) = swto3d(&qq, &dd, &gg, ng, nz);
            })
        })
        .sample_size(20),
    );

    c.bench(
        "full_init",
        Benchmark::new("ng32_nz4", |b| {
            b.iter(|| {
                let ng = black_box(32);
                let nz = black_box(4);
                let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);
                let (qq, dd, gg) = balinit(qq.as_slice().unwrap(), ng, nz);
                let (_, _, _) = swto3d(&qq, &dd, &gg, ng, nz);
            })
        })
        .sample_size(10),
    );

    c.bench(
        "full_init",
        Benchmark::new("ng64_nz16", |b| {
            b.iter(|| {
                let ng = black_box(64);
                let nz = black_box(16);
                let qq = init_pv_strip(ng, 0.4, 0.02, -0.01);
                let (qq, dd, gg) = balinit(qq.as_slice().unwrap(), ng, nz);
                let (_, _, _) = swto3d(&qq, &dd, &gg, ng, nz);
            })
        })
        .sample_size(10),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
