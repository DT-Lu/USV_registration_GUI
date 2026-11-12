# USV Registration GUI
NEWLY NAMED: "JWI-EAK"

A Julia/GTK 4 graphical interface for **registering and inspecting mouse ultrasonic vocalizations (USVs)** in WaveSurfer `.h5` recordings.

`Jwi-eak.jl` is the current version of the GUI. On top of manual box annotation, it adds a one-click **automatic USV detection** pipeline (SDTS denoising → time-axis detection → ridge/harmonic frequency bounds) whose results load straight into the editor for review and correction.

## Features

* Load WaveSurfer `.h5` recordings (or reopen a saved `.jld` annotation)
* Interactive spectrogram with a kHz frequency axis
* **Automatic USV detection** with the **USV detection** button (new in temp9)
* Manually draw, move, resize and delete USV bounding boxes
* Per-USV close-up view with acoustic features
* Population histograms of acoustic features
* Save annotations as `.jld`, optionally exporting each USV as an image (image modes: `lite`, `gui`, `fixed`, `fullband`, `legacy128`) with a `metadata.txt`
* Command-line launch with options (`--help`), including a no-GUI `--export-only`
* Window title shows the current file and an unsaved-changes marker (`*`)
* Keyboard and mouse-wheel navigation

---

## Repository Structure

> `Jwi-eak.jl` lives on the **`temp9`** branch.

```text
USV_registration_GUI/
│
├── Jwi-eak.jl                         ← current GUI (this README)
├── temp8.jl, temp7.3.jl, temp7.2.jl ← earlier versions
├── jwi-eak.sh / jwi-eak.bat         ← command-line launchers
├── usv.ui                           ← GTK 4 layout (window, buttons, panels)
├── Project.toml / Manifest.toml     ← Julia environment
│
├── Necessities/
│   ├── wavesurfer.jl                ← WaveSurfer / NI-DAQ .h5 loader
│   ├── acoustic.jl                  ← STFT parameters and spectrogram (spec_data)
│   ├── gui_function_temp2.jl        ← file loading, coordinate conversion, features
│   ├── GUItype_temp2.jl             ← USVBox type for saving annotations
│   ├── GDK_KEYmap.jl                ← key constants
│   ├── JPEGsaver_function_temp.jl   ← per-USV image export
│   ├── image_modes.jl               ← image modes, startup chooser, metadata.txt
│   ├── cli.jl                       ← command-line options
│   ├── usv_detection_3.jl           ← automatic detection pipeline
│   └── sdts_filter.jl               ← SDTS transient-suppression filter
│
└── Release note/
```

Jwi-eak.jl loads only the files listed above from `Necessities/`.

---

## Requirements

* Julia ≥ 1.10 (developed and tested primarily on **Windows**)
* **GTK 4** through [Gtk4.jl](https://github.com/JuliaGtk/Gtk4.jl) (0.7) and GtkObservables 2. GTK itself comes as a Julia artifact, so nothing is installed system-wide.
* Packages declared in `Project.toml`:

```text
Gtk4  GtkObservables  Graphics  Colors  Cairo  CairoMakie
JLD  HDF5  FFTW  DSP  Distributions  LinearAlgebra
Images  ImageTransformations  ImageMagick  FileIO  Interpolations
```

The standard libraries `Dates`, `Printf`, `Random`, `SparseArrays` and `Statistics` are also used; they ship with Julia.

---

## Installation

```bash
git clone https://github.com/NeuroParkUSVregistrators/USV_registration_GUI.git
cd USV_registration_GUI
git checkout temp9
```

Then, in Julia started from that directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

#### Moving an existing checkout from GTK 3 to GTK 4

The `Gtk4` branch replaces `Gtk` (GTK 3) with `Gtk4` and needs GtkObservables 2. After switching branches, update the environment once:

```bash
julia --project=. -e "using Pkg; Pkg.resolve(); Pkg.instantiate()"
```

`Gtk` and `Gtk4` cannot be loaded in the same Julia session, so restart Julia if `Gtk` was loaded earlier. If `Jwi-eak.jl` is started before this step, it prints the command above and exits.

#### What changed with GTK 4

* **Layout:** `usv.glade` became `usv.ui`, in GTK 4 format (validated with `gtk4-builder-tool`). Glade cannot edit GTK 4 files; use [Cambalache](https://gitlab.gnome.org/jpu/cambalache) or edit the XML. The layout and widget IDs are the same.
* **Dialogs:** the startup mode dialog and the file dialogs are now GTK 4 dialogs.
* **Shortcuts:** they work no matter which widget was clicked last; there's no need to click the window first.
* **Mouse wheel:** zoom works over the spectrogram and air-flow panels and on empty areas, and sliders still scroll themselves.
* **Window size:** GTK 4's default theme draws larger widgets, so the smallest possible window is about 1800×1360 px (GTK 3: about 1650×1250).


---

## Running

### From the command line (recommended)

```bash
julia --project=. -t auto Jwi-eak.jl [FILE] [options]
```

The launchers do the same from any folder: `./jwi-eak.sh` on Linux/macOS, `jwi-eak.bat` on Windows. Julia stays open until the window is closed.

| Option | Meaning |
|---|---|
| `FILE` | `.h5` (new registration) or `.jld` (reopen; the matching `.h5` must be in the same folder). Omit for a file dialog. |
| `-c`, `--channel N` | analog channel 1 or 2 (default 1) |
| `-m`, `--mode NAME` | image mode for Save (see below). Omit to choose in a dialog at startup. |
| `--timescale S` | reference GUI timescale in seconds (gui-style modes) |
| `--canvas WxH` | reference window size, e.g. `1600x500`, or `live` |
| `--band LO-HI` | image band in kHz, e.g. `30-125`, or `full` |
| `--pad-ms MS` | time padding on each side of a box |
| `--format FMT` | `png` or `jpeg` |
| `--export-only` | no GUI: write images + `metadata.txt` for a `.jld` and exit (needs `--mode`) |
| `--sr X` | contrast cap for `--export-only` (the GUI's sngscale value, default 0.5) |
| `--list-modes`, `-h`, `--help` | print modes / help |

```bash
julia --project=. -t auto Jwi-eak.jl rec_001.h5 --mode fixed
```

```bash
julia --project=. -t auto Jwi-eak.jl rec_001.jld -c 2 --mode gui --timescale 1.0
```

```bash
julia --project=. Jwi-eak.jl rec_001.jld --export-only --mode fullband --sr 0.3
```

`-t auto` lets the SDTS filter in automatic detection use all CPU cores.

### From the REPL

```julia
include("Jwi-eak.jl")                         # dialogs, as before
main(["rec_001.h5", "--mode", "lite"])        # or with options
```

---

## Basic Workflow

```text
   WaveSurfer recording (.h5)
              │
              ▼
          Load in GUI
              │
      ┌───────┴────────┐
      ▼                ▼
 USV detection    Draw boxes by hand
      └───────┬────────┘
              ▼
   Review / move / resize / delete
              │
              ▼
         Save (Ctrl+S)
           /       \
          ▼         ▼
       .jld     <name>_jpegs/
```

1. **Load** a recording (startup dialog or the **Load** button). Loading a new file resets the editor state.
2. **Detect** — press **USV detection** to auto-annotate the whole recording, or skip this and annotate by hand.
3. **Review** — step through boxes with `↑`/`↓`; the view jumps to each one and the side panel shows its close-up and features. Fix or delete boxes as needed.
4. **Save** with the **Save** button or `Ctrl+S`.

---

## Automatic USV Detection

Pressing **USV detection** runs, on the loaded channel:

1. **SDTS filter** (`sdts_filter.jl`) — Short-Duration Transient Suppression. Smooths the spectrogram with time-elongated and diagonal Gaussians (to keep tonal and FM components) and subtracts a frequency-elongated Gaussian (to suppress short broadband clicks).
2. **Denoising and time-axis detection** (`denoise_usv`, `usv_whole_2d`) — finds USV onsets/offsets.
3. **Frequency bounds with harmonics** (`usv_detect_2d(...; harmonics=true, harm_kw=HARM_KW_MORE)`) — sets each box's lower/upper frequency from the ridge, extended to include weak harmonics found on an SNR spectrogram.

The console prints the number of detected USVs, and the boxes appear in the editor marked as unsaved.

> **Detection replaces all current boxes**, including any drawn by hand or loaded from a `.jld`. Run it before manual edits, or don't save if the result is not wanted.

### Tuning

Settings are globals that can be changed from the REPL (after `include("Jwi-eak.jl")`) before pressing the button:

```julia
SDTS_ENABLED[] = true     # false = skip SDTS
SDTS_ALPHA[]   = 0.5      # weight of the original spectrogram (1.0 = no filtering)
SDTS_M[]       = 7        # kernel size in bins
SDTS_A[]       = nothing  # Gaussian scale; nothing → SDTS_M/10
```

Harmonic search parameters are in `HARM_KW_MORE` (`usv_detection_3.jl`); detection thresholds are keyword arguments of `usv_whole_2d` in the same file.

### Frequency-bound refinement (branch `10`)

Box frequency bounds are computed by `usv_harmonic_analysis` with three refinements (all on by default):

| Option | Default | What it does |
|---|---|---|
| `measure_sng`, `measure_floor` | `spec.fout`, `row_noise_floor(spec.fout)` (set in `Jwi-eak.jl`) | Ridges are **found** on the SDTS map, but their width is **measured** on the raw spectrogram, so SDTS no longer shaves the ridge skirts. `nothing` = measure on the SDTS map (old behaviour). |
| `subbin` | `true` | Edges are placed where the ridge crosses the threshold, interpolated in dB between bins (the ridge contour uses parabolic peak interpolation). Removes the ±0.5-bin (~±244 Hz) quantisation. |
| `bound_mode`, `edge_jump_hz` | `:continuous`, `4000.0` | The lowest/highest edge counts only if a neighbouring column's edge is within `edge_jump_hz`. A single noisy column can no longer stretch the box, while sweep ends that continue smoothly are kept. `:quantile` = old `ridge_q_lo`/`ridge_q_hi` rule. |

`usv_harmonic_analysis(...; bound_mode=:quantile, subbin=false)` without `measure_sng` reproduces the temp9 result.
Each result also carries `contour_t` / `contour_hz`, the sub-bin fundamental ridge.

Benchmark on synthetic calls with known frequency ranges (old vs new, several SNRs):

```bash
julia --project=. benchmark/freq_bounds_benchmark.jl
```

`compare_boxes` in the same file compares automatic boxes against manually corrected ones.

---

## Controls

| Control | Action |
| --- | --- |
| Left drag | Draw a new USV box |
| `Shift` + left drag (inside a box) | Move that box |
| Right drag (inside a box) | Resize: moves the box's second corner |
| Mouse wheel | Zoom time window (0.1–15 s), centered on the current view |
| `←` / `→` | Scroll back / forward through the recording |
| `↑` / `↓` | Select next / previous USV |
| `Ctrl + S` | Save |
| `Delete`, `Backspace`, `Ctrl + Z` | Delete the selected USV |

Box colors: **orange** = selected, **blue** = other boxes, **red** = being drawn, **green** = being moved, **yellow** = being resized.

The selected USV can also be chosen with the slider/spin button in the single-USV panel, and the time position with the slider/spin button under the spectrogram. The vertical sliders set the display intensity scale.

> `Ctrl + Z` deletes the selected box; it is **not** undo.

---

## Window Title

The title shows the loaded file and the image mode, with `*` while there are unsaved changes:

```text
USV260801_001.h5  [fixed]      ← saved
USV260801_001.h5  [fixed] *    ← unsaved changes
```

If a callback fails, the title changes to `ERROR: <action> - Check console` and the details are printed in the Julia console.

---

## Output Files

### Annotations (`.jld`)

Saved next to the recording with the same name:

```text
USV260801_001.h5
USV260801_001.jld
```

Boxes are sorted by time and stored as plain `USVBox(x1, y1, x2, y2)` records under the key `"liness"` (x in seconds, y as a 0–1 frequency ratio), with `"y_unit" = "ratio"`. No Gtk objects are stored, so the file can be read without the GUI:

```julia
using JLD
include("Necessities/GUItype_temp2.jl")   # needs XY/UserUnit from Graphics
d = JLD.load("USV260801_001.jld")
d["liness"]   # Vector{USVBox}
```

`.jld` files from earlier versions load in temp9; the y unit (ratio or frequency bin) is detected automatically.

### Images (image modes)

What Save writes besides the `.jld` depends on the **image mode**. Choose it with `--mode`, or in a dialog at startup. The mode is shown in the window title, e.g. `rec_001.h5  [fixed]`.

| Mode | Output |
|---|---|
| `lite` | nothing (only the `.jld`; fastest) |
| `gui` | each box as seen in the GUI at timescale 1.5 s, **current window size**, 30–125 kHz, PNG |
| `fixed` | like `gui`, but with a fixed 1600×500 reference window, so the pixel scale is the same every session (recommended for datasets) |
| `fullband` | like `fixed`, full frequency band |
| `legacy128` | the old 128×128 stretched JPEG (30–125 kHz, percentile contrast) → `_jpegs/` |

`--timescale`, `--canvas`, `--band`, `--pad-ms` and `--format` override the preset (gui-style modes). To add a mode, add one line to `IMAGE_MODES` in `Necessities/image_modes.jl`.

How gui-style images are built:

* Images look like the GUI spectrogram at the reference timescale, not squashed to 128 × 128. Each image's size follows the call's duration and band.
  * width = columns × canvas_width × dt / timescale
  * height = bins × canvas_height / nfreq
* Contrast is the same as the GUI, `clamp(fout, 0, sngscale) / sngscale`, using the slider value at save time.
* The top row of the image is the highest frequency.

```text
USV260801_001_images/
  metadata.txt
  0001_12.3456-12.4012_gui.png
  0002_...
```

#### `metadata.txt`

Every export writes a `metadata.txt` in the image folder. It has two parts:

* `# key: value` header lines: date, source files, channel, mode, STFT parameters (fs, nfft, hop, dt, df), canvas size and where it came from, px per second / column / bin, band, contrast (`sr`), padding and format.
* A tab-separated table, one row per image: `index, file, t_start_s, t_end_s, duration_ms, col_start, col_end, box_f_lo_hz, box_f_hi_hz, img_f_lo_hz, img_f_hi_hz, width_px, height_px`. `legacy128` has a shorter table.

```python
pd.read_csv("metadata.txt", sep="\t", comment="#")
```

```julia
CSV.read("metadata.txt", DataFrame; delim='\t', comment="#")
```

These images are intended for image-based analyses such as dimensionality reduction, clustering, or neural-network models.

---

## Acoustic Measurements

For each box the GUI shows:

| Measurement | Unit |
| --- | --- |
| Duration | ms |
| Loudness | dB |
| Spectral purity | ratio |
| Mean frequency | kHz |
| Pitch variance | log₁₀(Hz²) |

The bottom panel shows the distribution of each measurement over all boxes in the file.

In temp9(Jwi-eak), purity, mean frequency and pitch variance are computed on the **fundamental band** of a box when it contains harmonics, so harmonics do not skew them. To use the whole box as in earlier versions:

```julia
FEATURE_BAND[] = :box    # default :fundamental
```

---

## What's New in temp9

* **USV detection** button: SDTS denoising + 2-D detection + harmonic-aware frequency bounds
* Frequency coordinates shared between manual and detected boxes (no ±1 bin offset when loading detected boxes; no hard-coded 257 bins)
* Features computed on the fundamental band by default (`FEATURE_BAND`)
* Old `.jld` files with y stored in frequency bins are now converted correctly when loaded
* kHz frequency axis drawn on the spectrogram
* Analog channel 1 selected by default, with `channel_select()` available
* Clearer errors for unsupported file types or a missing `.h5` next to a `.jld`

---

## Notes

* The GUI assumes WaveSurfer HDF5 recordings in the structure read by `wavesurfer.jl`, sampled at **250 kHz** (STFT: 512-point Hann window, hop 128). Check sampling rate and channel layout before using data from a different setup.
* Saving overwrites the existing `.jld`. JPEGs with the same name are overwritten, but images from boxes that were since deleted or moved stay in the folder; clear `<name>_jpegs/` first if you need an exact match to the `.jld`.

---

## Authors / Contributors

NeuroPark USV Registrators

## Repository

https://github.com/NeuroParkUSVregistrators/USV_registration_GUI
