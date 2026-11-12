# =====================================================================
# image_modes.jl
#   Save 할 때 USV 이미지를 어떻게 export 할지 정하는 "모드" (preset) 모음,
#   시작 시 모드 선택 (dialog / terminal), metadata.txt 작성.
#
#   모드는 --mode 로 고르거나, GUI 시작 시 dialog 에서 고름.
#   --timescale / --canvas / --band / --pad-ms / --format 으로 preset 값을 덮어쓸 수 있음.
#   새 모드를 추가하려면 IMAGE_MODES 에 한 줄 추가.
# =====================================================================

using Dates
using Printf

# style = :none (저장 안 함) | :gui (export_usv_images_gui) | :legacy128 (export_usv_jpegs_lowcut)
#   canvas_px = nothing → 저장 시점의 실제 spectrogram 창 크기 (창이 없으면 DEFAULT_CANVAS_PX)
#   band      = :box → 각 box 자신의 주파수 범위 (선택한 box 그대로), nothing → 전체 대역, (lo, hi) Hz → 고정 대역
const DEFAULT_CANVAS_PX = (1600, 500)

const IMAGE_MODES = [
    (; name="lite", style=:none,
       desc="no images: Save writes the .jld only (fast)"),
    (; name="gui", style=:gui, timescale=1.5, canvas_px=nothing,
       band=:box, pad_ms=0.0, ext="png",
       desc="exactly the selected box (its time and frequency range), as seen in the GUI at timescale 1.5 s, current window size, PNG"),
    (; name="fixed", style=:gui, timescale=1.5, canvas_px=DEFAULT_CANVAS_PX,
       band=:box, pad_ms=0.0, ext="png",
       desc="like gui (exact box), but a fixed 1600x500 reference window: same pixel scale every session (datasets)"),
    (; name="fullband", style=:gui, timescale=1.5, canvas_px=DEFAULT_CANVAS_PX,
       band=nothing, pad_ms=0.0, ext="png",
       desc="like fixed, but the full frequency band (0 Hz - Nyquist)"),
    (; name="legacy128", style=:legacy128,
       desc="old 128x128 stretched JPEG, 30-125 kHz, percentile contrast"),
]

image_mode_names() = [m.name for m in IMAGE_MODES]

function find_image_mode(name::AbstractString)
    i = findfirst(m -> m.name == lowercase(strip(name)), IMAGE_MODES)
    i === nothing && throw(ArgumentError(
        "unknown image mode \"$name\" (choose from: $(join(image_mode_names(), ", ")))"))
    return IMAGE_MODES[i]
end

# preset + CLI override. override 는 :gui 스타일에만 의미가 있음.
function image_mode_config(name::AbstractString; overrides::NamedTuple=(;))
    m = find_image_mode(name)
    if m.style !== :gui
        isempty(overrides) ||
            @warn "image mode \"$(m.name)\" ignores the options $(join(keys(overrides), ", "))"
        return m
    end
    return merge(m, overrides)
end

function describe_image_modes(io::IO=stdout)
    for m in IMAGE_MODES
        @printf(io, "  %-10s %s\n", m.name, m.desc)
    end
end

# --- 시작 시 모드 선택 ------------------------------------------------
# GTK4 alert dialog (버튼 = 모드, 비동기). 고른 모드 이름으로 callback(name) 호출.
#   Esc / 창 닫기 → "lite". dialog 를 못 띄우면 terminal 에서 물어봄.
#   (Gtk4.ask_dialog 와 같은 방식: GtkAlertDialog + set_buttons + choose)
function choose_image_mode(callback::Function; parent=nothing)
    names = image_mode_names()
    detail = "What Save writes besides the .jld:\n\n" *
             join(["• $(m.name): $(m.desc)" for m in IMAGE_MODES], "\n") *
             "\n\nTip: skip this dialog with  --mode <name>"
    try
        dlg = Gtk4.GtkAlertDialog("Image mode for this session"; modal=true)
        Gtk4.G_.set_detail(dlg, detail)
        Gtk4.G_.set_buttons(dlg, names)
        Gtk4.choose(dlg, parent) do d, resobj
            idx = try
                Gtk4.choose_finish(d, resobj)          # 0-based 버튼 번호
            catch e
                e isa Gtk4.GLib.GErrorException || rethrow()
                -1                                     # Esc / 닫기
            end
            name = 0 <= idx < length(names) ? names[idx+1] : "lite"
            idx < 0 && println("Mode dialog closed without a choice -> lite")
            callback(name)
        end
    catch e
        @warn "mode dialog failed; asking in the terminal" exception = (e, catch_backtrace())
        callback(choose_image_mode_terminal())
    end
    return nothing
end

function choose_image_mode_terminal()
    stdin isa Base.TTY || (println("No terminal input -> lite"); return "lite")
    names = image_mode_names()
    println("Image mode for this session:")
    for (i, m) in enumerate(IMAGE_MODES)
        @printf("  %d) %-10s %s\n", i, m.name, m.desc)
    end
    while true
        print("Enter number or name [1]: ")
        s = strip(readline())
        isempty(s) && return names[1]
        k = tryparse(Int, s)
        k !== nothing && 1 <= k <= length(names) && return names[k]
        lowercase(s) in names && return lowercase(s)
        println("Invalid choice.")
    end
end

# --- export 실행 + metadata ---------------------------------------
# boxes : [(t1, t2, f_lo_hz, f_hi_hz), ...]  (시간 순서, .jld 에 저장된 순서와 같아야 함)
# sr    : 대비 상한 (GUI sngscale 값)
# live_canvas_px : 현재 GUI spectrogram 창 크기 (없으면 nothing)
# 반환: 출력 폴더 (lite 면 nothing)
function run_image_export(cfg, nmjld::AbstractString, spec, boxes;
                          sr::Real, channel::Int=1,
                          live_canvas_px::Union{Nothing,Tuple{Int,Int}}=nothing)
    cfg.style === :none && (println("Image mode lite: image export skipped"); return nothing)

    dotloc = findlast('.', nmjld)
    base = (dotloc === nothing) ? nmjld : nmjld[1:(dotloc-1)]
    f = spec_freq_axis(spec)
    t = spec.t_sng
    dt = length(t) >= 2 ? Float64(t[2] - t[1]) : NaN
    nfreq = size(spec.fout, 1)

    hdr = Pair{String,Any}[
        "created" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "source_jld" => abspath(nmjld),
        "source_h5" => abspath(string(base, ".h5")),
        "channel" => channel,
        "mode" => cfg.name,
        "mode_description" => cfg.desc,
        "fs_hz" => hasproperty(spec, :fs) ? spec.fs : "unknown",
        "nfft" => hasproperty(spec, :sp) ? spec.sp.nfft : "unknown",
        "hop" => hasproperty(spec, :sp) ? spec.sp.skip : "unknown",
        "dt_s" => dt,
        "df_hz" => length(f) >= 2 ? Float64(f[2] - f[1]) : NaN,
        "n_freq_bins" => nfreq,
        "n_boxes" => length(boxes),
    ]

    if cfg.style === :legacy128
        outdir = string(base, "_jpegs")
        export_usv_jpegs_lowcut(nmjld; outdir=outdir, outH=128, outW=128,
                                contrast=:quantile, keep_aspect=false,
                                qlo=0.02, qhi=0.98, gamma=1.0, idx=channel)
        append!(hdr, Pair{String,Any}[
            "image_size_px" => "128x128 (stretched, not padded)",
            "band_hz" => "30000-125000",
            "contrast" => "quantile 0.02-0.98 per image",
            "format" => "jpeg"])
        rows = [(; index=i, file=@sprintf("%04d_%.4f-%.4f_lowcut.jpeg", i, b[1], b[2]),
                   t_start_s=b[1], t_end_s=b[2], duration_ms=1000 * (b[2] - b[1]),
                   box_f_lo_hz=b[3], box_f_hi_hz=b[4]) for (i, b) in enumerate(boxes)]
        write_image_metadata(joinpath(outdir, "metadata.txt"), hdr, rows)
        return outdir
    end

    # :gui
    px, px_src = if cfg.canvas_px !== nothing
        cfg.canvas_px, "fixed (mode/option)"
    elseif live_canvas_px !== nothing
        live_canvas_px, "GUI window at save time"
    else
        DEFAULT_CANVAS_PX, "default (no GUI window)"
    end
    outdir = string(base, "_images")
    tspans = [(b[1], b[2]) for b in boxes]
    fspans = [(b[3], b[4]) for b in boxes]
    rows = export_usv_images_gui(spec.fout, f, t, tspans;
                                 outdir=outdir, sr=sr, timescale=cfg.timescale,
                                 canvas_px=px, band=cfg.band, pad_ms=cfg.pad_ms,
                                 ext=cfg.ext, box_freqs=fspans)
    append!(hdr, Pair{String,Any}[
        "reference_timescale_s" => cfg.timescale,
        "canvas_px" => "$(px[1])x$(px[2])",
        "canvas_px_source" => px_src,
        "px_per_second" => px[1] / cfg.timescale,
        "px_per_time_column" => px[1] * dt / cfg.timescale,
        "px_per_freq_bin" => px[2] / nfreq,
        "band_hz" => cfg.band === nothing ? "full (0-Nyquist)" :
                     cfg.band === :box ? "box (each image = that box's own frequency range; see img_f_lo_hz / img_f_hi_hz)" :
                     @sprintf("%.0f-%.0f", cfg.band[1], cfg.band[2]),
        "contrast" => "clamp(power, 0, sr) / sr  (same as GUI), sr = $(sr)",
        "resampling" => "area average when shrinking, bilinear when enlarging",
        "frequency_orientation" => "top row = highest frequency",
        "time_pad_ms" => cfg.pad_ms,
        "format" => cfg.ext])
    write_image_metadata(joinpath(outdir, "metadata.txt"), hdr, rows)
    return outdir
end

_meta_fmt(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.10g", x)   # 긴 녹음의 시간도 0.1 ms 이하 정밀도
_meta_fmt(x) = string(x)

# metadata.txt : "# key: value" 헤더 + tab 구분 표 (첫 줄 = 열 이름)
#   pandas: pd.read_csv(path, sep="\t", comment="#")
#   Julia : CSV.read(path, DataFrame; delim='\t', comment="#")
function write_image_metadata(path::AbstractString, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# USV image export metadata")
        for (k, v) in header
            println(io, "# ", k, ": ", v)
        end
        isempty(rows) && return
        cols = collect(keys(first(rows)))
        println(io, join(string.(cols), '\t'))
        for r in rows
            println(io, join((_meta_fmt(getproperty(r, c)) for c in cols), '\t'))
        end
    end
    println("Metadata written: ", path)
    return path
end
