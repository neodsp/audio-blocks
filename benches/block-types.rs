use audio_block::{
    AudioBlockMut, AudioBlockOpsMut, interleaved::Interleaved, planar::Planar,
    sequential::Sequential,
};
use criterion::{Criterion, criterion_group, criterion_main};

pub fn bench_three_types(c: &mut Criterion, num_channels: u16, num_frames: usize) {
    let mut block = Interleaved::<f32>::new(num_channels, num_frames);
    c.bench_function(
        &format!("for each interleaved {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|v| *v *= 2.0)),
    );

    let mut block = Sequential::<f32>::new(num_channels, num_frames);
    c.bench_function(
        &format!("for each sequential {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|v| *v *= 2.0)),
    );

    let mut block = Planar::<f32>::new(num_channels, num_frames);
    c.bench_function(
        &format!("for each planar {num_channels}ch {num_frames}fr"),
        |b| b.iter(|| block.for_each(|v| *v *= 2.0)),
    );
}

pub fn block_view(c: &mut Criterion) {
    bench_three_types(c, 2, 512);
}

pub fn bench_gain_comparison(c: &mut Criterion, num_channels: u16, num_frames: usize) {
    // Compare iteration performance on full buffer (visible == allocated)
    let mut block = Interleaved::<f32>::new(num_channels, num_frames);

    c.bench_function(
        &format!("gain for_each interleaved {num_channels}ch {num_frames}fr"),
        |b| {
            b.iter(|| {
                block.for_each(|v| *v *= 0.5);
            })
        },
    );

    let mut block = Interleaved::<f32>::new(num_channels, num_frames);

    c.bench_function(
        &format!("gain for_each_allocated interleaved {num_channels}ch {num_frames}fr"),
        |b| {
            b.iter(|| {
                block.for_each_allocated(|v| *v *= 0.5);
            })
        },
    );
}

pub fn bench_gain_comparison_half_visible(c: &mut Criterion, num_channels: u16, num_frames: usize) {
    // Compare when buffer is resized to half the allocation
    // for_each processes only visible (half), gain processes all (full)
    let mut block = Interleaved::<f32>::new(num_channels, num_frames);
    block.set_visible(num_channels, num_frames / 2);

    c.bench_function(
        &format!("gain for_each half visible interleaved {num_channels}ch {num_frames}fr"),
        |b| {
            b.iter(|| {
                block.for_each(|v| *v *= 0.5);
            })
        },
    );

    let mut block = Interleaved::<f32>::new(num_channels, num_frames);
    block.set_visible(num_channels, num_frames / 2);

    c.bench_function(
        &format!(
            "gain for_each_allocated half visible interleaved {num_channels}ch {num_frames}fr"
        ),
        |b| {
            b.iter(|| {
                block.for_each_allocated(|v| *v *= 0.5);
            })
        },
    );
}

pub fn gain_comparison(c: &mut Criterion) {
    bench_gain_comparison(c, 8, 512);
    bench_gain_comparison(c, 16, 512);
    bench_gain_comparison(c, 32, 512);
}

pub fn gain_comparison_half_visible(c: &mut Criterion) {
    bench_gain_comparison_half_visible(c, 8, 512);
    bench_gain_comparison_half_visible(c, 16, 512);
    bench_gain_comparison_half_visible(c, 32, 512);
}

criterion_group!(
    benches,
    block_view,
    gain_comparison,
    gain_comparison_half_visible
);
criterion_main!(benches);
