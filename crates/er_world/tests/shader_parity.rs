use er_core::math::uv_to_dir;
use glam::DVec3;
use wgpu::util::DeviceExt;

const SPHERIFY: &str = include_str!("../../er_terrain/assets/shaders/spherify.wgsl");

const HARNESS: &str = r#"
@group(0) @binding(0) var<storage, read> faces: array<u32>;
@group(0) @binding(1) var<storage, read> us: array<f32>;
@group(0) @binding(2) var<storage, read> vs: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_dirs: array<vec4<f32>>;

@compute @workgroup_size(64)
fn uv_to_dir_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&faces)) {
        return;
    }
    let d = uv_to_dir(i32(faces[i]), us[i], vs[i]);
    out_dirs[i] = vec4<f32>(d, 0.0);
}
"#;

fn samples() -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let uv = [0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, -0.1];
    let mut faces = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();
    for face in 0u32..6 {
        for &u in &uv {
            for &v in &uv {
                faces.push(face);
                us.push(u);
                vs.push(v);
            }
        }
    }
    (faces, us, vs)
}

async fn run_gpu(faces: &[u32], us: &[f32], vs: &[f32]) -> Option<Vec<[f32; 4]>> {
    let n = faces.len();
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

    let source = format!("{SPHERIFY}\n{HARNESS}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spherify_parity"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("uv_to_dir_eval"),
        layout: None,
        module: &shader,
        entry_point: Some("uv_to_dir_eval"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);

    let faces_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("faces"),
        contents: bytemuck::cast_slice(faces),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let us_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("us"),
        contents: bytemuck::cast_slice(us),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let vs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vs"),
        contents: bytemuck::cast_slice(vs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_dirs"),
        size: (n * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 16) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: faces_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: us_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n as u32 + 63) / 64) as u32;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (n * 16) as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Some(out)
}

#[test]
fn parity_uv_to_dir_wgsl_vs_cpu() {
    let (faces, us, vs) = samples();
    let gpu = match pollster::block_on(run_gpu(&faces, &us, &vs)) {
        Some(o) => o,
        None => {
            eprintln!("No GPU adapter available; skipping spherify parity test");
            return;
        }
    };

    assert_eq!(gpu.len(), faces.len());
    let mut max_diff: f64 = 0.0;
    for i in 0..faces.len() {
        let cpu = uv_to_dir(faces[i] as u8, us[i] as f64, vs[i] as f64);
        let gpu_dir = DVec3::new(gpu[i][0] as f64, gpu[i][1] as f64, gpu[i][2] as f64);
        let diff = (cpu - gpu_dir).length();
        max_diff = max_diff.max(diff);
        assert!(
            diff <= 1e-5,
            "uv_to_dir parity failed at i={}: face={} u={} v={} cpu={:?} gpu={:?} diff={diff}",
            i,
            faces[i],
            us[i],
            vs[i],
            cpu,
            gpu_dir
        );
    }
    eprintln!(
        "uv_to_dir max diff: {max_diff} (tolerance 1e-5) over {} samples",
        faces.len()
    );
}
