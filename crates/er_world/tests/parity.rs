use er_core::rng::rng_from_seed;
use er_core::seed::PlanetSeed;
use er_world::biome::elevation_low_freq_metric;
use er_world::brushes::BrushSet;
use er_world::elevation::{
    elevation, elevation_at, elevation_params, ElevationNoise, ElevationParams,
};
use er_world::terrain_space::metric_surface_point;
use glam::DVec3;
use rand::RngCore;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("../assets/shaders/elevation.wgsl");

fn u2f(u: u64) -> f64 {
    (u as f64 / u64::MAX as f64) * 2.0 - 1.0
}

fn generate_dirs(seed: u64, count: usize) -> Vec<DVec3> {
    let mut rng = rng_from_seed(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let x = u2f(rng.next_u64());
        let y = u2f(rng.next_u64());
        let z = u2f(rng.next_u64());
        let v = DVec3::new(x, y, z);
        let l2 = v.length_squared();
        if l2 > 1e-6 && l2 < 1.0 {
            out.push(v.normalize());
        }
    }
    out
}

async fn run_compute(
    params: &ElevationParams,
    dirs: &[DVec3],
    entry_point: &str,
) -> Option<Vec<f32>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok()?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("elevation"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("elevation_eval"),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group_layout = pipeline.get_bind_group_layout(0);

    let n = dirs.len();

    let dirs_data: Vec<[f32; 4]> = dirs
        .iter()
        .map(|d| [d.x as f32, d.y as f32, d.z as f32, 0.0])
        .collect();
    let dirs_bytes = bytemuck::cast_slice(&dirs_data);

    let dirs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dirs"),
        contents: dirs_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let elevs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("elevs"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_bytes: &[u8] = bytemuck::bytes_of(params);
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dirs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: elevs_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (n as u32).div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&elevs_buffer, 0, &staging, 0, (n * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let elevs: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Some(elevs)
}

#[test]
fn parity_cpu_vs_wgsl() {
    let n: usize = 10_000;

    let params = elevation_params(PlanetSeed(0xC0FFEE));
    let noise = ElevationNoise::new(&params);

    let dirs = generate_dirs(0xABCDEF, n);

    let cpu_elevs: Vec<f64> = dirs
        .iter()
        .map(|d| elevation(*d, &noise, &params))
        .collect();

    let gpu_elevs = match pollster::block_on(run_compute(&params, &dirs, "elevation_eval")) {
        Some(e) => e,
        None => {
            eprintln!("No GPU adapter available; skipping parity test");
            return;
        }
    };

    assert_eq!(
        cpu_elevs.len(),
        gpu_elevs.len(),
        "CPU and GPU elevation counts differ"
    );

    let mut max_diff: f64 = 0.0;
    for (i, (cpu, gpu)) in cpu_elevs.iter().zip(gpu_elevs.iter()).enumerate() {
        let diff = (*cpu - *gpu as f64).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= 1e-4,
            "parity failed at index {i}: cpu={cpu}, gpu={gpu}, diff={diff}"
        );
    }
    eprintln!("Max diff: {max_diff} (tolerance 1e-4)");
}

#[test]
fn metric_low_frequency_parity_cpu_vs_wgsl() {
    let radius = 6_371_000.0;
    let params = elevation_params(PlanetSeed(0xC0FFEE));
    let noise = ElevationNoise::new_metric(&params);
    let dirs = generate_dirs(0x51_0A_E1, 10_000);
    let cpu_elevs: Vec<f64> = dirs
        .iter()
        .map(|dir| {
            elevation_low_freq_metric(metric_surface_point(*dir, radius), &noise).low_freq_elev
        })
        .collect();

    let gpu_elevs = match pollster::block_on(run_compute(
        &params,
        &dirs,
        "elevation_metric_low_freq_eval",
    )) {
        Some(elevations) => elevations,
        None => {
            eprintln!("No GPU adapter available; skipping parity test");
            return;
        }
    };

    let mut max_diff = 0.0_f64;
    for (index, (cpu, gpu)) in cpu_elevs.iter().zip(gpu_elevs.iter()).enumerate() {
        let diff = (*cpu - *gpu as f64).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= 1e-3,
            "metric macro parity failed at index {index}: cpu={cpu}, gpu={gpu}, diff={diff}"
        );
    }
    eprintln!("Metric macro max diff: {max_diff} (tolerance 1e-3)");
}

#[test]
fn metric_full_elevation_parity_cpu_vs_wgsl() {
    let radius = 6_371_000.0;
    let params = elevation_params(PlanetSeed(0xC0FFEE));
    let noise = ElevationNoise::new_metric(&params);
    let dirs = generate_dirs(0x7A_11_E2, 10_000);
    let cpu_elevs: Vec<f64> = dirs
        .iter()
        .map(|dir| elevation_at(metric_surface_point(*dir, radius), &noise))
        .collect();

    let gpu_elevs =
        match pollster::block_on(run_compute(&params, &dirs, "elevation_metric_full_eval")) {
            Some(elevations) => elevations,
            None => {
                eprintln!("No GPU adapter available; skipping parity test");
                return;
            }
        };

    let mut max_diff = 0.0_f64;
    for (index, (cpu, gpu)) in cpu_elevs.iter().zip(gpu_elevs.iter()).enumerate() {
        let diff = (*cpu - *gpu as f64).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= 1e-2,
            "metric full parity failed at index {index}: cpu={cpu}, gpu={gpu}, diff={diff}"
        );
    }
    eprintln!("Metric full max diff: {max_diff} (tolerance 1e-2)");
}

/// Isolated brush-displacement CPU-vs-WGSL parity regression test.
///
/// Compares the WGSL `brush_displacement_eval` entry point against the CPU
/// `BrushSet::displacement_exhaustive` over fixed seeded directions. The brush
/// integer hash uses wrapping `u32` arithmetic, so the `params.seed` (`i32`)
/// field is bit-reinterpreted: a negative `i32` seed must produce the same
/// `u32` on both sides. We therefore cover a positive seed and a negative
/// (i32) seed.
#[test]
fn brush_displacement_parity_cpu_vs_wgsl() {
    // Seeds exercised for params.seed (i32). 0x7FFF_FFFF is the largest
    // positive i32. 0x8000_0000 has its sign bit set, so as an i32 it is
    // negative (-2147483648), stressing the bitcast path in both the WGSL
    // `bitcast<u32>(seed)` and the Rust `seed as u32` reinterpretation.
    let seeds: [i32; 2] = [0x7FFF_FFFF, 0x8000_0000u32 as i32];

    for seed in seeds {
        // Reuse the production params builder, then override only the seed
        // (public field). All other params are irrelevant to brush displacement.
        let mut params = elevation_params(PlanetSeed(0xC0FFEE));
        params.seed = seed;

        // Build the CPU brush set from the bit-reinterpreted seed. This
        // mirrors how ElevationNoise constructs BrushSet internally
        // (`BrushSet::from_seed(seed as u32)`).
        let brushes = BrushSet::from_seed(seed as u32);

        let n: usize = 10_000;
        let dirs = generate_dirs(0xB6E5EE0, n);

        // CPU reference: exhaustive loop (the WGSL entry point also loops all
        // BRUSH_CAP brushes; the spatial index is covered by a separate test).
        let cpu_elevs: Vec<f32> = dirs
            .iter()
            .map(|d| {
                let dir_f = glam::Vec3::new(d.x as f32, d.y as f32, d.z as f32).normalize();
                brushes.displacement_exhaustive(dir_f)
            })
            .collect();

        let gpu_elevs =
            match pollster::block_on(run_compute(&params, &dirs, "brush_displacement_eval")) {
                Some(elevations) => elevations,
                None => {
                    eprintln!("No GPU adapter available; skipping brush parity test");
                    return;
                }
            };

        assert_eq!(
            cpu_elevs.len(),
            gpu_elevs.len(),
            "CPU and GPU brush displacement counts differ (seed={seed:#x})"
        );

        let mut max_diff: f32 = 0.0;
        let mut active = 0usize;
        for (index, (cpu, gpu)) in cpu_elevs.iter().zip(gpu_elevs.iter()).enumerate() {
            let diff = (*cpu - *gpu).abs();
            if cpu.abs() > 0.01 {
                active += 1;
            }
            max_diff = max_diff.max(diff);
            assert!(
                diff <= 1e-2,
                "brush parity failed at seed={seed:#x} index={index}: cpu={cpu}, gpu={gpu}, diff={diff}"
            );
        }
        eprintln!(
            "Brush parity seed={seed:#x}: max diff: {max_diff} (tolerance 1e-2), active samples: {active}/{}",
            n
        );
    }
}
