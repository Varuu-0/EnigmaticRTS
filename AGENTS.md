# EnigmaticRTS — Agent Notes

Deterministic planet + mini solar-system simulator in Rust/Bevy (Planetary Annihilation style).
See `.kilo/plans/1782755385739-planet-solar-sim-plan.md` for the full 13-phase build plan.

## Build / Run / Test

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"   # refresh cargo PATH each session
cargo build --workspace
cargo run -p er_game          # opens a Bevy window "Planet Solar Sim"
cargo test --workspace
```

## Verified toolchain (Windows / MSVC)

- rustc 1.96.0, target `x86_64-pc-windows-msvc` (host). GNU/MinGW toolchain is BROKEN on this machine — do not use it.
- MSVC compiler/linker installed via VS 2022 BuildTools + components:
  - `Microsoft.VisualStudio.Component.VC.Tools.x86.x64` (cl.exe / link.exe)
  - `Microsoft.VisualStudio.Component.Windows11SDK.22621`
  - `link.exe` at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\link.exe`
- rustc auto-discovers the linker via vswhere; no manual PATH for link.exe needed.
- VS Installer quirk: `--passive`/`--quiet` modes **refuse to self-elevate** (exit code 5007 =
  "Commands with --quiet or --passive should be run elevated from the beginning"). To add VS
  components, launch the installer ELEVATED (e.g. via a `.bat` + `Start-Process -Verb RunAs`), and
  quote the `--installPath` (it contains spaces). Passing args as a `Start-Process -ArgumentList`
  array splits `"C:\Program Files..."` at the space — use a single argument string or a `.bat`.

## Verified dependency versions (workspace)

| crate        | version | notes                                                |
|--------------|---------|------------------------------------------------------|
| bevy         | 0.19    | latest available                                     |
| glam         | 0.32    | MUST match bevy's glam (0.32.1); pinning 0.30 causes a duplicate glam compile + future type-mismatch across the er_core→bevy boundary |
| rand         | 0.9     | rand_core 0.9                                        |
| rand_chacha  | 0.9     | MUST match rand's rand_core (0.9); pinning 0.10 pulls rand_core 0.10 → two incompatible rand_core versions → `from_seed`/`fill_bytes` trait-bound errors |
| fastnoise-lite | 1.1  |                                                      |
| bytemuck     | 1       |                                                      |

Rule of thumb: `rand` and `rand_chacha` share a minor version (both 0.9). `glam` must equal
bevy's bundled glam (currently 0.32).

## Workspace layout (5 crates)

- `er_core`  — no Bevy; seeds, RNG (ChaCha8), config/tunables, math foundation
- `er_world`  — procgen: elevation, biomes, water, planet params
- `er_terrain` — quadtree LOD, chunk mesh generation
- `er_render` — materials, shaders, atmosphere, ocean
- `er_game`   — Bevy app entry point (`main.rs`)

RTS crates (sim/nav/gpu_sim/net/save) are deferred — out of scope for the current "see the planets" milestone.

## Performance notes (this machine: Optimus laptop)

- Panel is 144 Hz, driven by the Intel iGPU; the NVIDIA RTX 3060 dGPU renders (logs show
  `AdapterInfo ... NVIDIA`). This is a hybrid/Optus setup.
- **Windowed + VSync ON (`PresentMode::AutoVsync`/FIFO) caps at ~74 fps** here (~half the panel
  refresh) due to the dGPU→iGPU cross-adapter copy being vsync-locked every other frame. This is
  why an empty scene showed ~74 fps, not 144.
- **`PresentMode::Immediate` (VSync OFF) DOES NOT WORK on this Optimus setup**: it uncaps to
  ~144–400 fps but the sustained high-frequency presenting through the dGPU→iGPU cross-adapter
  copy triggers a Vulkan `DeviceLost` after ~35 s and the app quits. Do NOT use `Immediate` here.
- **`PresentMode::AutoNoVsync` (Mailbox)** is the safe "no-vsync" choice: stable (no DeviceLost;
  verified 9500+ frames), runs ~150–300 fps when focused / ~60 when occluded. `er_game` defaults
  to this.
- **`PresentMode::AutoVsync` (FIFO)** is the stable "vsync on" choice (windowed ~74 due to the
  cross-adapter half-refresh cap, fullscreen ~144).
- **Runtime present-mode / window-mode changes recreate the Vulkan swapchain**, and on this
  Optimus setup even a few spaced-out changes lose the GPU device (`DeviceLost` after ~3 toggles).
  So `er_game` applies VSync/Fullscreen **at startup only**, read from a persisted file
  (`er_game_settings.txt` next to the exe). In-menu toggles just save + log "(restart to apply)"
  and never touch the live window, so they can no longer crash. MSAA is a per-camera component
  (no swapchain recreation) and applies live on menu close.
- **Exclusive fullscreen** (`WindowMode::Fullscreen(MonitorSelection::Current, VideoModeSelection::Current)`)
  bypasses the cross-adapter copy. It MUST be set at runtime (after the window exists); setting it
  at creation panics with "Unable to get monitor" because `MonitorSelection::Current` needs an
  existing window. `er_game` applies it once at startup (`apply_startup_window_mode`) if saved.
- `er_game` defaults to VSync OFF (AutoNoVsync) + windowed, with an ESC menu
  (Fullscreen / VSync / MSAA / Quit). `Msaa` is a per-camera Component (not a Resource) in 0.19.

## Conventions

- f64 world math / f32 render boundary; spherified-cube projection; origin-shifting.
- Elevation is a deterministic pure function `(seed, dir)`.
- Max 3 concurrent subagents for implementation work.
- Do not commit unless explicitly asked.
