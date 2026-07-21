use er_core::rng::rng_from_seed;
use er_core::seed::PlanetSeed;
use er_world::biome::{biome, elevation_split};
use er_world::elevation::{elevation_params, ElevationNoise, ElevationParams};
use er_world::params::{climate_noise, planet_params, PlanetParams};
use glam::DVec3;
use rand::RngCore;
use wgpu::util::DeviceExt;

const ELEVATION_SHADER: &str = include_str!("../assets/shaders/elevation.wgsl");
const BIOME_SHADER: &str = include_str!("../assets/shaders/biome.wgsl");

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BiomeShaderParams {
    pub sea_level: f32,
    pub lapse_rate: f32,
    pub temp_gradient: f32,
    pub temp_noise_freq: f32,
    pub temp_noise_amp: f32,
    pub moisture_noise_freq: f32,
    pub moisture_noise_amp: f32,
    pub rain_shadow_strength: f32,
    pub high_alt_threshold: f32,
    pub beach_threshold: f32,
    pub volcanic_threshold: f32,
    pub toxic_moisture_threshold: f32,
    pub toxic_temp_threshold: f32,
    pub temp_noise_seed: i32,
    pub moisture_noise_seed: i32,
    pub lacunarity: f32,
    pub gain: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl BiomeShaderParams {
    fn from(elev_params: &ElevationParams, planet: &PlanetParams) -> Self {
        Self {
            sea_level: planet.sea_level as f32,
            lapse_rate: planet.lapse_rate as f32,
            temp_gradient: planet.temp_gradient as f32,
            temp_noise_freq: planet.temp_noise_freq,
            temp_noise_amp: planet.temp_noise_amp,
            moisture_noise_freq: planet.moisture_noise_freq,
            moisture_noise_amp: planet.moisture_noise_amp,
            rain_shadow_strength: planet.rain_shadow_strength,
            high_alt_threshold: planet.high_alt_threshold as f32,
            beach_threshold: planet.beach_threshold as f32,
            volcanic_threshold: planet.volcanic_threshold as f32,
            toxic_moisture_threshold: planet.toxic_moisture_threshold as f32,
            toxic_temp_threshold: planet.toxic_temp_threshold as f32,
            temp_noise_seed: planet.temp_noise_seed,
            moisture_noise_seed: planet.moisture_noise_seed,
            lacunarity: elev_params.lacunarity,
            gain: elev_params.gain,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

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

async fn run_biome_compute(
    elev_params: &ElevationParams,
    biome_params: &BiomeShaderParams,
    dirs: &[DVec3],
) -> Option<Vec<u32>> {
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

    let source = format!("{ELEVATION_SHADER}\n{BIOME_SHADER}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("biome"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("biome_eval"),
        layout: None,
        module: &shader,
        entry_point: Some("biome_eval"),
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
        label: Some("biome_dirs"),
        contents: dirs_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let biomes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("biomes"),
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

    let elev_params_bytes: &[u8] = bytemuck::bytes_of(elev_params);
    let elev_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elev_params"),
        contents: elev_params_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let biome_params_bytes: &[u8] = bytemuck::bytes_of(biome_params);
    let biome_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("biome_params"),
        contents: biome_params_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: elev_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: biome_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dirs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: biomes_buffer.as_entire_binding(),
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
    encoder.copy_buffer_to_buffer(&biomes_buffer, 0, &staging, 0, (n * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let biomes: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Some(biomes)
}

#[test]
fn parity_biome_cpu_vs_wgsl() {
    let n: usize = 10_000;

    let seed = PlanetSeed(0xC0FFEE);
    let elev_params = elevation_params(seed);
    let noise = ElevationNoise::new(&elev_params);
    let planet = planet_params(seed);
    let climate = climate_noise(&planet);
    let biome_shader_params = BiomeShaderParams::from(&elev_params, &planet);

    let dirs = generate_dirs(0xABCDEF, n);

    let cpu_biomes: Vec<u32> = dirs
        .iter()
        .map(|d| {
            let split = elevation_split(*d, &noise, &elev_params);
            biome(
                *d,
                split.full_elev,
                split.low_freq_elev,
                split.mountain_influence,
                &planet,
                &climate,
            ) as u32
        })
        .collect();

    let gpu_biomes =
        match pollster::block_on(run_biome_compute(&elev_params, &biome_shader_params, &dirs)) {
            Some(b) => b,
            None => {
                eprintln!("No GPU adapter available; skipping biome parity test");
                return;
            }
        };

    assert_eq!(
        cpu_biomes.len(),
        gpu_biomes.len(),
        "CPU and GPU biome counts differ"
    );

    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (i, (cpu, gpu)) in cpu_biomes.iter().zip(gpu_biomes.iter()).enumerate() {
        if cpu != gpu {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, *cpu, *gpu, dirs[i], {
                    let split = elevation_split(dirs[i], &noise, &elev_params);
                    let temp =
                        er_world::biome::temperature(dirs[i], split.full_elev, &planet, &climate);
                    let moist = er_world::biome::moisture(
                        dirs[i],
                        split.mountain_influence,
                        &planet,
                        &climate,
                    );
                    (split.full_elev, split.low_freq_elev, temp, moist)
                }));
            }
        }
    }

    let mismatch_pct = (mismatches as f64 / n as f64) * 100.0;
    eprintln!("Biome parity: {mismatches}/{n} mismatches ({mismatch_pct:.3}%)");

    if let Some((i, cpu, gpu, dir, (elev, lfe, temp, moist))) = first_mismatch {
        eprintln!(
            "First mismatch at i={i}: cpu_biome={cpu}, gpu_biome={gpu}, dir={dir:?}, elev={elev:.6}, low_freq={lfe:.6}, temp={temp:.6}, moisture={moist:.6}"
        );
    }

    assert!(
        mismatch_pct <= 0.1,
        "Biome parity failed: {mismatch_pct:.3}% > 0.1% mismatch threshold ({mismatches}/{n})"
    );
}
