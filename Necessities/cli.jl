# =====================================================================
# cli.jl : Jwi-eak 명령행 옵션
#   ArgParse 없이 직접 파싱 (Manifest 변경 없음).
# =====================================================================

const CLI_USAGE = """
Usage:
  julia --project=. -t auto Jwi-eak.jl [FILE] [options]

  FILE                 .h5 recording or .jld annotation (omit -> file dialog)

Options:
  -c, --channel N      analog channel to use: 1 or 2            (default 1)
  -m, --mode NAME      image mode for Save (omit -> asked at startup)
      --timescale S    reference GUI timescale in seconds       (gui-style modes)
      --canvas WxH     reference window size in px, e.g. 1600x500, or "live"
      --band B         image band: "box" (each box's own range), "full", or LO-HI in kHz (e.g. 30-125)
      --pad-ms MS      time padding on each side of a box
      --format FMT     png | jpeg
      --export-only    no GUI: export images (+ metadata.txt) for FILE (.jld) and exit
      --sr X           contrast cap for --export-only (GUI sngscale value; default 0.5)
      --window W       start window size: fit (default, ~90% of the screen), max, or WxH (e.g. 1280x720)
      --list-modes     print the image modes and exit
  -h, --help           print this help and exit

Image modes:
"""

function print_cli_help(io::IO=stdout)
    print(io, CLI_USAGE)
    describe_image_modes(io)
    println(io, """

Examples:
  julia --project=. -t auto Jwi-eak.jl rec_001.h5 --mode fixed
  julia --project=. -t auto Jwi-eak.jl rec_001.jld -c 2 --mode gui --timescale 1.0
  julia --project=. Jwi-eak.jl rec_001.jld --export-only --mode fullband --sr 0.3""")
end

struct CLIError <: Exception
    msg::String
end
Base.showerror(io::IO, e::CLIError) = print(io, e.msg)

function _parse_band(s::AbstractString)
    lowercase(s) == "full" && return nothing
    lowercase(s) == "box" && return :box
    m = match(r"^\s*([0-9.]+)\s*-\s*([0-9.]+)\s*$", s)
    m === nothing && throw(CLIError("--band expects \"box\", \"full\" or LO-HI in kHz (e.g. 30-125), got \"$s\""))
    lo, hi = parse(Float64, m[1]) * 1e3, parse(Float64, m[2]) * 1e3
    lo < hi || throw(CLIError("--band: LO must be below HI, got \"$s\""))
    return (lo, hi)
end

function _parse_canvas(s::AbstractString)
    lowercase(s) == "live" && return nothing
    m = match(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$", s)
    m === nothing && throw(CLIError("--canvas expects WxH (e.g. 1600x500) or \"live\", got \"$s\""))
    w, h = parse(Int, m[1]), parse(Int, m[2])
    (w > 0 && h > 0) || throw(CLIError("--canvas: sizes must be positive"))
    return (w, h)
end

function _parse_num(T, s, opt)
    x = tryparse(T, s)
    x === nothing && throw(CLIError("$opt expects a number, got \"$s\""))
    return x
end

"""
    parse_cli(args) -> NamedTuple

(; file, channel, mode, overrides, export_only, sr, help, list_modes)
`overrides` 는 image_mode_config 에 넘길 NamedTuple.
"""
function parse_cli(args::AbstractVector{<:AbstractString})
    file = nothing; channel = 1; mode = nothing
    overrides = (;); export_only = false; sr = 0.5
    help = false; list_modes = false; window = :fit
    i = 1
    function value(opt)
        i < length(args) || throw(CLIError("$opt needs a value"))
        i += 1
        return args[i]
    end
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            help = true
        elseif a == "--list-modes"
            list_modes = true
        elseif a in ("-c", "--channel")
            channel = _parse_num(Int, value(a), a)
            channel in (1, 2) || throw(CLIError("--channel must be 1 or 2"))
        elseif a in ("-m", "--mode")
            mode = lowercase(value(a))
            mode in image_mode_names() ||
                throw(CLIError("unknown mode \"$mode\" (choose from: $(join(image_mode_names(), ", ")))"))
        elseif a == "--timescale"
            ts = _parse_num(Float64, value(a), a)
            ts > 0 || throw(CLIError("--timescale must be > 0"))
            overrides = merge(overrides, (; timescale=ts))
        elseif a == "--canvas"
            overrides = merge(overrides, (; canvas_px=_parse_canvas(value(a))))
        elseif a == "--band"
            overrides = merge(overrides, (; band=_parse_band(value(a))))
        elseif a == "--pad-ms"
            p = _parse_num(Float64, value(a), a)
            p >= 0 || throw(CLIError("--pad-ms must be >= 0"))
            overrides = merge(overrides, (; pad_ms=p))
        elseif a == "--format"
            fm = lowercase(value(a))
            fm == "jpg" && (fm = "jpeg")
            fm in ("png", "jpeg") || throw(CLIError("--format must be png or jpeg"))
            overrides = merge(overrides, (; ext=fm))
        elseif a == "--export-only"
            export_only = true
        elseif a == "--window"
            v = lowercase(value(a))
            window = v == "fit" ? :fit : v == "max" ? :max : _parse_canvas(v)
            window === nothing && throw(CLIError("--window expects fit, max or WxH (e.g. 1280x720)"))
        elseif a == "--sr"
            sr = _parse_num(Float64, value(a), a)
            sr > 0 || throw(CLIError("--sr must be > 0"))
        elseif startswith(a, "-") && length(a) > 1
            throw(CLIError("unknown option \"$a\" (see --help)"))
        else
            file === nothing || throw(CLIError("only one FILE can be given (got \"$file\" and \"$a\")"))
            file = a
        end
        i += 1
    end
    if file !== nothing
        isfile(file) || throw(CLIError("file not found: $file"))
        ext = lowercase(splitext(file)[2])
        ext in (".h5", ".jld") || throw(CLIError("FILE must be .h5 or .jld, got \"$file\""))
        file = abspath(file)
    end
    if export_only && !help && !list_modes
        file === nothing && throw(CLIError("--export-only needs a .jld FILE"))
        endswith(lowercase(file), ".jld") || throw(CLIError("--export-only needs a .jld FILE (boxes come from it)"))
        mode === nothing && throw(CLIError("--export-only needs --mode"))
        mode == "lite" && throw(CLIError("--export-only with --mode lite would export nothing"))
    end
    return (; file, channel, mode, overrides, export_only, sr, help, list_modes, window)
end
