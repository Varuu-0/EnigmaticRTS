//! Optional, minimal adapter-wide GPU memory telemetry.
//!
//! Windows-only module that queries DXGI for live VRAM usage of the adapter
//! with the largest dedicated-memory budget. DXGI exposes adapter-wide memory
//! budget and usage, not per-process allocation data. On non-Windows targets (or if DXGI is unavailable at
//! runtime) the API degrades to a safe unavailable stub.
//!
//! All work is performed synchronously on the calling thread — no background
//! threads are spawned. The module does not depend on NVML, PDH counters, or
//! any external crate; it uses raw FFI against `dxgi.dll` / `dxgi.lib`.

#[cfg(target_os = "windows")]
mod windows_backend {
    use std::ffi::c_void;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::ptr;

    use super::{GpuTelemetrySample, GpuTelemetryStatus};

    const S_OK: i32 = 0;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Guid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }

    const IID_IDXGIFACTORY1: Guid = Guid {
        data1: 0x770AAE78,
        data2: 0xF26F,
        data3: 0x4DBA,
        data4: [0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87],
    };

    const IID_IDXGIADAPTER3: Guid = Guid {
        data1: 0x645967A4,
        data2: 0x1392,
        data3: 0x4310,
        data4: [0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD],
    };

    #[repr(C)]
    struct DxgiAdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        adapter_luid_low: u32,
        adapter_luid_high: i32,
        flags: u32,
    }

    #[repr(C)]
    struct DxgiQueryVideoMemoryInfo {
        budget: u64,
        current_usage: u64,
        available_for_reservation: u64,
        current_reservation: u64,
    }

    type QueryInterfaceFn =
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32;
    type EnumAdapters1Fn = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
    type GetDesc1Fn = unsafe extern "system" fn(*mut c_void, *mut DxgiAdapterDesc1) -> i32;
    type QueryVideoMemoryInfoFn =
        unsafe extern "system" fn(*mut c_void, u32, u32, *mut DxgiQueryVideoMemoryInfo) -> i32;
    type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;

    const VTBL_QUERY_INTERFACE: usize = 0;
    const VTBL_RELEASE: usize = 2;
    const FACTORY1_VTBL_ENUM_ADAPTERS1: usize = 12;
    const ADAPTER1_VTBL_GET_DESC1: usize = 10;
    const ADAPTER3_VTBL_QUERY_VIDEO_MEMORY_INFO: usize = 14;

    unsafe fn vtable_fn<T>(com: *mut c_void, offset: usize) -> T {
        let vtable: *const *mut c_void = *(com as *const *const *mut c_void);
        let func_ptr: *mut c_void = *vtable.add(offset);
        std::mem::transmute_copy(&func_ptr)
    }

    pub fn sample() -> GpuTelemetrySample {
        let factory = match create_factory1() {
            Some(f) => f,
            None => return unavailable("Failed to create IDXGIFactory1"),
        };

        let adapter = match enum_highest_memory_adapter(factory) {
            Some(a) => a,
            None => {
                unsafe { release(factory) };
                return unavailable("No DXGI adapters found");
            }
        };

        let desc = match get_adapter_desc1(adapter) {
            Some(d) => d,
            None => {
                unsafe {
                    release(adapter);
                    release(factory);
                }
                return unavailable("GetDesc1 failed");
            }
        };

        let description = wide_to_string(&desc.description);
        let dedicated_video_memory = desc.dedicated_video_memory as u64;

        let memory = query_adapter3_memory(adapter)
            .map(|info| (info.budget, info.current_usage, info.current_reservation));

        unsafe {
            release(adapter);
            release(factory);
        }

        match memory {
            Some((budget, usage, reservation)) => GpuTelemetrySample {
                status: GpuTelemetryStatus::Available,
                description,
                vendor_id: desc.vendor_id,
                device_id: desc.device_id,
                dedicated_video_memory_bytes: dedicated_video_memory,
                vram_budget_bytes: budget,
                vram_usage_bytes: usage,
                vram_reservation_bytes: reservation,
            },
            None => GpuTelemetrySample {
                status: GpuTelemetryStatus::Partial,
                description,
                vendor_id: desc.vendor_id,
                device_id: desc.device_id,
                dedicated_video_memory_bytes: dedicated_video_memory,
                vram_budget_bytes: 0,
                vram_usage_bytes: 0,
                vram_reservation_bytes: 0,
            },
        }
    }

    fn create_factory1() -> Option<*mut c_void> {
        unsafe {
            let mut factory: *mut c_void = ptr::null_mut();
            let result = CreateDXGIFactory1(&IID_IDXGIFACTORY1, &mut factory as *mut *mut c_void);
            if result == S_OK && !factory.is_null() {
                Some(factory)
            } else {
                None
            }
        }
    }

    fn enum_highest_memory_adapter(factory: *mut c_void) -> Option<*mut c_void> {
        let mut best_adapter = None;
        let mut best_dedicated_memory = 0usize;

        for index in 0.. {
            let Some(adapter) = enum_adapter1(factory, index) else {
                break;
            };

            let dedicated_memory = get_adapter_desc1(adapter)
                .map(|desc| desc.dedicated_video_memory)
                .unwrap_or(0);
            if dedicated_memory > best_dedicated_memory || best_adapter.is_none() {
                if let Some(previous) = best_adapter.replace(adapter) {
                    unsafe { release(previous) };
                }
                best_dedicated_memory = dedicated_memory;
            } else {
                unsafe { release(adapter) };
            }
        }

        best_adapter
    }

    fn enum_adapter1(factory: *mut c_void, index: u32) -> Option<*mut c_void> {
        unsafe {
            let func: EnumAdapters1Fn = vtable_fn(factory, FACTORY1_VTBL_ENUM_ADAPTERS1);
            let mut adapter: *mut c_void = ptr::null_mut();
            let result = func(factory, index, &mut adapter);
            (result == S_OK && !adapter.is_null()).then_some(adapter)
        }
    }

    fn get_adapter_desc1(adapter: *mut c_void) -> Option<DxgiAdapterDesc1> {
        unsafe {
            let func: GetDesc1Fn = vtable_fn(adapter, ADAPTER1_VTBL_GET_DESC1);
            let mut desc = DxgiAdapterDesc1 {
                description: [0u16; 128],
                vendor_id: 0,
                device_id: 0,
                sub_sys_id: 0,
                revision: 0,
                dedicated_video_memory: 0,
                dedicated_system_memory: 0,
                shared_system_memory: 0,
                adapter_luid_low: 0,
                adapter_luid_high: 0,
                flags: 0,
            };
            let result = func(adapter, &mut desc);
            if result == S_OK {
                Some(desc)
            } else {
                None
            }
        }
    }

    fn query_adapter3_memory(adapter: *mut c_void) -> Option<DxgiQueryVideoMemoryInfo> {
        unsafe {
            let qi: QueryInterfaceFn = vtable_fn(adapter, VTBL_QUERY_INTERFACE);
            let mut adapter3: *mut c_void = ptr::null_mut();
            let result = qi(adapter, &IID_IDXGIADAPTER3, &mut adapter3);
            if result != S_OK || adapter3.is_null() {
                return None;
            }

            let func: QueryVideoMemoryInfoFn =
                vtable_fn(adapter3, ADAPTER3_VTBL_QUERY_VIDEO_MEMORY_INFO);
            let mut info = DxgiQueryVideoMemoryInfo {
                budget: 0,
                current_usage: 0,
                available_for_reservation: 0,
                current_reservation: 0,
            };
            let result = func(adapter3, 0, 0, &mut info);
            release(adapter3);
            if result == S_OK {
                Some(info)
            } else {
                None
            }
        }
    }

    unsafe fn release(com: *mut c_void) {
        if com.is_null() {
            return;
        }
        let func: ReleaseFn = vtable_fn(com, VTBL_RELEASE);
        func(com);
    }

    fn wide_to_string(buf: &[u16; 128]) -> String {
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        let os_string = OsString::from_wide(&buf[..len]);
        os_string.to_string_lossy().into_owned()
    }

    fn unavailable(reason: &str) -> GpuTelemetrySample {
        GpuTelemetrySample {
            status: GpuTelemetryStatus::Unavailable(reason.to_owned()),
            description: String::new(),
            vendor_id: 0,
            device_id: 0,
            dedicated_video_memory_bytes: 0,
            vram_budget_bytes: 0,
            vram_usage_bytes: 0,
            vram_reservation_bytes: 0,
        }
    }

    #[link(name = "dxgi")]
    extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, pp_factory: *mut *mut c_void) -> i32;
    }
}

#[cfg(target_os = "windows")]
pub use windows_backend::sample;

#[cfg(not(target_os = "windows"))]
mod stub {
    use super::{GpuTelemetrySample, GpuTelemetryStatus};

    pub fn sample() -> GpuTelemetrySample {
        GpuTelemetrySample {
            status: GpuTelemetryStatus::Unavailable(
                "GPU telemetry not supported on this OS".to_owned(),
            ),
            description: String::new(),
            vendor_id: 0,
            device_id: 0,
            dedicated_video_memory_bytes: 0,
            vram_budget_bytes: 0,
            vram_usage_bytes: 0,
            vram_reservation_bytes: 0,
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub use stub::sample;

#[derive(Clone, Debug, PartialEq)]
pub enum GpuTelemetryStatus {
    Available,
    Partial,
    Unavailable(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuTelemetrySample {
    pub status: GpuTelemetryStatus,
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_video_memory_bytes: u64,
    pub vram_budget_bytes: u64,
    pub vram_usage_bytes: u64,
    pub vram_reservation_bytes: u64,
}

impl GpuTelemetrySample {
    pub fn is_available(&self) -> bool {
        matches!(self.status, GpuTelemetryStatus::Available)
    }

    pub fn vram_usage_percent(&self) -> Option<f32> {
        if self.vram_budget_bytes > 0 {
            Some((self.vram_usage_bytes as f64 / self.vram_budget_bytes as f64 * 100.0) as f32)
        } else {
            None
        }
    }

    pub fn vram_usage_gib(&self) -> f32 {
        (self.vram_usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) as f32
    }

    pub fn vram_budget_gib(&self) -> f32 {
        (self.vram_budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) as f32
    }

    pub fn dedicated_video_memory_gib(&self) -> f32 {
        (self.dedicated_video_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_does_not_panic() {
        let s = sample();
        assert!(!s.description.is_empty() || !s.is_available());
    }

    #[test]
    fn percentage_is_none_when_budget_zero() {
        let s = GpuTelemetrySample {
            status: GpuTelemetryStatus::Unavailable("test".to_owned()),
            description: String::new(),
            vendor_id: 0,
            device_id: 0,
            dedicated_video_memory_bytes: 0,
            vram_budget_bytes: 0,
            vram_usage_bytes: 0,
            vram_reservation_bytes: 0,
        };
        assert!(s.vram_usage_percent().is_none());
    }

    #[test]
    fn percentage_is_some_when_budget_nonzero() {
        let s = GpuTelemetrySample {
            status: GpuTelemetryStatus::Unavailable("test".to_owned()),
            description: String::new(),
            vendor_id: 0,
            device_id: 0,
            dedicated_video_memory_bytes: 0,
            vram_budget_bytes: 1000,
            vram_usage_bytes: 250,
            vram_reservation_bytes: 0,
        };
        assert_eq!(s.vram_usage_percent(), Some(25.0));
    }
}
