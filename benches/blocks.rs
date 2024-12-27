use audio_blocks::{interleaved::Interleaved, ops::Ops, sequential::Sequential, stacked::Stacked};
use criterion::{criterion_group, criterion_main, Criterion};

pub fn bench_three_types(c: &mut Criterion, num_channels: u16, num_frames: usize) {
    let mut block = Interleaved::<f32>::empty(num_channels, num_frames);
    c.bench_function(
        &format!("for each interleaved {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|_, _, v| *v *= 2.0)),
    );

    let mut block = Sequential::<f32>::empty(num_channels, num_frames);
    c.bench_function(
        &format!("for each sequential {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|_, _, v| *v *= 2.0)),
    );

    let mut block = Stacked::<f32>::empty(num_channels, num_frames);
    c.bench_function(
        &format!("for each stacked {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|_, _, v| *v *= 2.0)),
    );
}

pub fn block_view(c: &mut Criterion) {
    // bench_three_types(c, 8, 16);
    // bench_three_types(c, 2, 16);
    // bench_three_types(c, 16, 16);
    // bench_three_types(c, 128, 16);
    //
    // bench_three_types(c, 1, 512);
    bench_three_types(c, 2, 512);
    // bench_three_types(c, 16, 512);
    // bench_three_types(c, 128, 512);

    // bench_three_types(c, 1, 1024);
    // bench_three_types(c, 2, 1024);
    // bench_three_types(c, 16, 1024);
    // bench_three_types(c, 128, 1024);
}

criterion_group!(benches, block_view);
criterion_main!(benches);
