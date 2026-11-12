using Pkg
Pkg.activate(@__DIR__)   # 어느 폴더에서 실행해도 이 repository 의 환경 사용
#Pkg.precompile()

# Gtk4 가 설치되지 않은 (Gtk3 시절) 환경이면 안내 후 종료
if Base.find_package("Gtk4") === nothing || Base.find_package("GtkObservables") === nothing
    println(stderr, """
    Gtk4 / GtkObservables 2 are not installed in this environment yet.
    Run once in the repository folder:

        julia setup_gtk4.jl
    """)
    isinteractive() ? error("Gtk4 not installed") : exit(1)
end

using Gtk4, GtkObservables, Graphics, Colors, CairoMakie, Cairo
# 주의: width / height 는 Gtk4(Graphics) 와 CairoMakie(GeometryBasics) 가 서로 다른 함수로 export
#       → 이름만 쓰면 UndefVarError (ambiguous). 위젯 크기는 항상 Gtk4.width / Gtk4.height 로.
using Gtk4.GLib
using JLD
using HDF5
using FFTW
using DSP
using LinearAlgebra
using Random, Distributions
using Dates
using Base.Filesystem: basename

const Axis = CairoMakie.Axis

# --- GTK4 helpers ---------------------------------------------------
# GTK4: 이벤트 mask 없음 → event controller. 수정자 키는 BitFlags (ModifierType).
_hasmod(state, flag) = (Gtk4.ModifierType(state & Gtk4.MODIFIER_MASK) & flag) == flag
_ctrl(state)  = _hasmod(state, Gtk4.ModifierType_CONTROL_MASK)
_shift(state) = _hasmod(state, Gtk4.ModifierType_SHIFT_MASK)
_alt(state)   = _hasmod(state, Gtk4.ModifierType_ALT_MASK)
_plain(state) = !(_ctrl(state) || _shift(state) || _alt(state))

set_text!(lbl, s) = (lbl.label = string(s); nothing)          # GtkLabel (GAccessor.text 대체)
set_title!(win, s) = (win.title = string(s); nothing)         # GtkWindow (GAccessor.title 대체)

# 컨트롤러가 자식 위젯보다 먼저 이벤트를 받도록 (GtkObservables canvas 가 scroll 을 먹기 때문)
function _capture_phase!(ec)
    try
        ec.propagation_phase = Gtk4.PropagationPhase_CAPTURE
    catch
        try
            Gtk4.G_.set_propagation_phase(ec, Gtk4.PropagationPhase_CAPTURE)
        catch e
            @warn "could not set capture phase" exception = e
        end
    end
    return ec
end

idle_add_compat(f::Function) = g_idle_add(f)
timeout_add_compat(f::Function, ms::Integer) = g_timeout_add(f, ms)
timeout_add_compat(ms::Integer, f::Function) = g_timeout_add(f, ms)

# h5 file 
include("Necessities/wavesurfer.jl")
include("Necessities/acoustic.jl")
include("Necessities/gui_function_temp2.jl")
include("Necessities/GUItype_temp2.jl")
include("Necessities/GDK_KEYmap.jl")
include("Necessities/JPEGsaver_function_temp.jl")
include("Necessities/usv_detection_3.jl")
include("Necessities/sdts_filter.jl")
include("Necessities/image_modes.jl")
include("Necessities/cli.jl")

function gui_idle(tag::AbstractString, f::Function)
    idle_add_compat() do
        try
            f()
        catch e
            @error "idle callback failed: $tag" exception = (e, catch_backtrace())
        end
        return false  # run once 
    end
    return nothing
end
gui_idle(f::Function, tag::AbstractString) = gui_idle(tag, f)

function safe_cb(tag::AbstractString, f::Function)
    try
        return f()
    catch e
        bt = catch_backtrace()
        @error "GTK callback failed: $tag" exception = (e, bt)
        try
            if isdefined(Main, :win) && win isa GtkWindow
                set_title!(win, "ERROR: $tag - Check console")
            end
        catch
        end
        return nothing
    end
end
safe_cb(f::Function, tag::AbstractString) = safe_cb(tag, f)

const hist_pending = Ref(false)
const hist_running = Ref(false)
const pending_liness_notify = Ref(false)
const last_notify_time = Ref(time())

dt_sng_obs = Observable(0.000512)

function sanity_check_liness!(lns, tmax)
    bad = 0
    for (i, b) in enumerate(lns)
        x1 = b[1].x.val
        x2 = b[2].x.val
        if !(isfinite(x1) && isfinite(x2)) || x2 < 0 || x1 > tmax
            bad += 1
        end
    end
    @info "sanity_check" total = length(lns) bad = bad tmax = tmax
end

# GUI box (XY{UserUnit}) → (t1, t2, f_lo_hz, f_hi_hz). 이미지 export / metadata 용.
function liness_to_boxes(lns, spec)
    f = spec_freq_axis(spec)
    nf = size(spec.fout, 1)
    map(lns) do l
        t1, t2 = minmax(Float64(l[1].x.val), Float64(l[2].x.val))
        fa = Float64(f[userunit_to_freq_bin(Float64(l[1].y.val), nf)])
        fb = Float64(f[userunit_to_freq_bin(Float64(l[2].y.val), nf)])
        (t1, t2, min(fa, fb), max(fa, fb))
    end
end

# GUI 창이 닫히면 notify (명령행 실행 시 julia 가 바로 끝나지 않도록 main 에서 wait)
const GUI_CLOSED = Condition()

function export_jpegs_after_save(nmjld::AbstractString;
    outdir::Union{Nothing,String}=nothing,
    outH::Int=128,
    outW::Int=128,
    contrast::Symbol=:quantile,
    qlo::Real=0.02,
    qhi::Real=0.98,
    gamma::Real=1.0
)
    # outdir 기본값 설정 
    if outdir === nothing
        dotloc = findlast('.', nmjld)
        base = (dotloc === nothing) ? nmjld : nmjld[1:(dotloc-1)]
        outdir = string(base, "_jpegs")
    end

    println("Exporting JPEGs to : ", outdir)

    export_usv_jpegs_lowcut(
        nmjld;
        outdir=outdir,
        outH=outH,
        outW=outW,
        contrast=contrast,
        keep_aspect=false,
        qlo=qlo,
        qhi=qhi,
        gamma=gamma,
        idx=choice
    )

    println("JPEG export finished.")
    return outdir
end

# 시작 창 크기 (창은 이후 마우스로 자유롭게 조절 가능; 화면보다 작아지면 scrollbar)
#   :fit   → 첫 번째 모니터의 약 90% × 85% (Windows 배율이 반영된 logical px, 작업표시줄 여유)
#   :max   → 최대화
#   (w, h) → 그 크기
function size_window!(win, spec=:fit)
    if spec === :max
        Gtk4.maximize(win)
        println("Window: maximized")
        return nothing
    end
    w, h = if spec isa Tuple
        spec
    else
        sw, sh = try
            Int.(Gtk4.screen_size())
        catch e
            @warn "could not read the screen size; using 1600x1000" exception = e
            (1600, 1000)
        end
        (round(Int, 0.90sw), round(Int, 0.85sh))
    end
    Gtk4.default_size(win, w, h)
    println("Window: $(w)x$(h)  (resize freely; start options: --window fit | max | WxH)")
    return nothing
end

# 녹음/annotation 파일을 못 읽었을 때 (손상·잘린 파일, 짝 .h5 없음, 없는 channel ...)
struct FileLoadError <: Exception
    path::String
    cause::Any
end
Base.showerror(io::IO, e::FileLoadError) = print(io, "could not open ", e.path, ": ", load_error_hint(e))

# 원인별로 사용자가 할 일을 한 줄로
function load_error_hint(e::FileLoadError)
    msg = sprint(showerror, e.cause)
    if occursin("truncated", msg)
        m = match(r"eof = (\d+).*?stored_eof = (\d+)", msg)
        sizes = m === nothing ? "" :
            " (on disk $(round(parse(Int, m[1]) / 1e6, digits=1)) MB of $(round(parse(Int, m[2]) / 1e6, digits=1)) MB)"
        return "the .h5 file is incomplete$(sizes): an unfinished download/copy or a recording still being written. Copy the file again from its source."
    elseif occursin("짝이 되는 h5", msg) || occursin("matching", msg)
        return "the matching .h5 recording is missing; it must be in the same folder as the .jld with the same name."
    elseif e.cause isa BoundsError
        return "the recording does not have analog channel $(choice). Start with --channel 1 or pick the other channel."
    elseif occursin("No such file", msg) || occursin("not found", lowercase(msg))
        return "the file was not found."
    elseif occursin("unable to lock", lowercase(msg)) || occursin("Permission", msg)
        return "the file is locked or not readable (open in another program, e.g. WaveSurfer?)."
    else
        return first(split(msg, '\n'))
    end
end

function report_load_error(e::FileLoadError)
    println(stderr, "\n[Could not open file] ", basename(e.path))
    println(stderr, "  ", load_error_hint(e))
    println(stderr, "  Full path: ", e.path)
    @debug "load error details" exception = e.cause
    return nothing
end

# GUI 시작 (GTK4: dialog 가 비동기라 callback 으로 이어짐)
#   mode dialog (image_cfg 가 없을 때) → file dialog (file 이 없을 때) → build_gui
# file      : 열 파일 (.h5/.jld). nothing → file dialog
# image_cfg : image_mode_config(...) 결과. nothing → 시작할 때 dialog 로 선택
# overrides : --timescale 등 CLI 값 (dialog 로 고른 모드에도 적용)
# 창이 끝내 안 만들어지면 (취소/오류) GUI_CLOSED 를 notify → 명령행 실행이 끝남
function make_gui(; file::Union{Nothing,AbstractString}=nothing, image_cfg=nothing,
                  overrides::NamedTuple=(;), window=:fit)
    function start(cfg)
        println("Image mode: ", cfg.name, "  (", cfg.desc, ")")
        if file !== nothing
            # 명령행으로 준 파일: 못 열면 (메시지 출력 후) 종료
            build_gui(String(file), cfg; overrides=overrides, window=window) isa GtkWindow || notify(GUI_CLOSED)
            return nothing
        end
        pick_file(cfg)
        return nothing
    end

    # 파일 dialog. 고른 파일을 못 열면 이유를 title 에 보여주고 dialog 를 다시 띄움 (취소하면 종료).
    function pick_file(cfg, title="Pick a .h5 recording or .jld annotation")
        open_dialog(title) do nm
            try
                if nm === nothing || isempty(nm)
                    println("No file chosen.")
                    notify(GUI_CLOSED)
                    return
                end
                r = build_gui(String(nm), cfg; overrides=overrides, window=window)
                if r === :load_failed
                    pick_file(cfg, "Could not open $(basename(nm)) (see console) - pick another file, or Cancel to quit")
                elseif !(r isa GtkWindow)
                    notify(GUI_CLOSED)
                end
            catch e
                @error "GUI start failed" exception = (e, catch_backtrace())
                notify(GUI_CLOSED)
            end
        end
        return nothing
    end

    if image_cfg === nothing
        choose_image_mode() do name
            try
                start(image_mode_config(name; overrides=overrides))
            catch e
                @error "GUI start failed" exception = (e, catch_backtrace())
                notify(GUI_CLOSED)
            end
        end
    else
        start(image_cfg)
    end
    return nothing
end

# 파일 nm 을 열어 main window 를 만듦.
#   반환: window | :load_failed (파일을 못 읽음; 메시지는 이미 출력) | nothing (그 밖의 시작 오류)
#   overrides : CLI 값 — 창 안의 Images dropdown 으로 모드를 바꿀 때도 적용
function build_gui(nm::AbstractString, cfg; overrides::NamedTuple=(;), window=:fit)
    try
        nm_ref = Ref{String}("")
        dirty_ref = Ref(false)
        image_mode = Ref{Any}(cfg)

        spec_obs = Observable{Any}(nothing)
        fout_obs = Observable{Matrix{Float64}}(zeros(1, 1))
        liness = Observable(Vector{Vector{XY{UserUnit}}}())
        t_sng_obs = Observable(range(0.0, step=dt_sng_obs[], length=1))
        currtime = Observable(0.0)

        function update_window_title!(
            win,
            filename::AbstractString;
            dirty::Bool=false,
        )
            file_label = isempty(filename) ? "USV Manual" : basename(filename)
            file_label = string(file_label, "  [", image_mode[].name, "]  [ch ", choice, "]")
            set_title!(win, dirty ? "$(file_label) *" : file_label)
            return nothing
        end

        # 새 spectrogram (파일 또는 channel 이 바뀜) 을 GUI 상태에 반영. box (liness) 는 건드리지 않음.
        function set_spec!(spec)
            if hasproperty(spec, :t_sng) && length(spec.t_sng) >= 2
                dt_sng_obs[] = spec.t_sng[2] - spec.t_sng[1]
                t_sng_obs[] = spec.t_sng
            else
                t_sng_obs[] = range(0.0, step=dt_sng_obs[], length=size(spec.fout, 2))
            end
            spec_obs[] = spec
            fout_obs[] = spec.fout
            return nothing
        end

        # 실패하면 FileLoadError 를 던지고 GUI 상태 (열린 파일 이름, spectrogram, box) 는 그대로 둠.
        #   (예전에는 nm_ref 를 먼저 바꿔서, Load 가 실패한 뒤 Save 하면 실패한 파일 이름의 .jld 로 저장됐음)
        function load_into_gui!(nm::AbstractString) # seperation : 2026-03-04 
            spec, tmp = try
                spec, raw_liness = load_file(String(nm), choice)
                raw = raw_liness isa Observable ? raw_liness[] : raw_liness
                tmp = normalize_liness(raw; nfreq=size(spec.fout, 1))
                (spec, tmp isa Observable ? tmp[] : tmp)
            catch e
                throw(FileLoadError(String(nm), e))
            end

            nm_ref[] = String(nm)
            set_spec!(spec)
            liness[] = tmp

            sanity_check_liness!(liness[], last(t_sng_obs[]))
            return nothing
        end

        # open the file 
        load_into_gui!(nm)

        # GTK4 UI definition 
        b = GtkBuilder(joinpath(@__DIR__, "usv.ui"))

        # whole window 
        win = b["win"]

        function set_dirty!(value::Bool)
            dirty_ref[] == value && return nothing
            dirty_ref[] = value
            update_window_title!(win, nm_ref[]; dirty=value)
            return nothing
        end

        update_window_title!(win, nm_ref[]; dirty=false)

        # --- helper (focus control) --- 
        unwrap_widget(x) = hasfield(typeof(x), :widget) ? getfield(x, :widget) : x

        function prevent_focus_on_click!(x)
            w = unwrap_widget(x)
            # GTK4: focusable = 위젯 자신, can-focus = 자식(예: spinbutton 안의 text)까지.
            #   GTK3 처럼 slider/spinbutton 이 키 입력을 가져가지 않게 둘 다 끔
            #   (window 의 CAPTURE 단축키가 spinbutton 타이핑 중 BackSpace 를 "box 삭제" 로 먹지 않도록).
            try
                w.focusable = false
            catch
            end
            try
                w.can_focus = false
            catch
            end
            try
                w.focus_on_click = false
            catch
            end
            return nothing
        end

        # display box 
        airdis = b["airdis"]
        sngdis = b["sngdis"]
        airlabel = b["airlabel"]
        snglabel = b["snglabel"]
        timelabel = b["timelabel"]
        timespanbox = b["timespanbox"]
        timescalebox = b["timescalebox"]
        currentindexbox = b["currentindexbox"]
        airscalebox = b["airscalebox"]
        sngscalebox = b["sngscalebox"]

        timestartlabel = b["timestart"]
        timeendlabel = b["timeend"]
        currentusvdur = b["currentusvdur"]
        currentusvamp = b["currentusvamp"]
        currentusvspur = b["currentusvspur"]
        currentusvpitch = b["currentusvpitch"]
        currentusvpitchvariance = b["currentusvpitchvariance"]


        function request_liness_notify(; min_dt=1 / 30)  # 30fps limitation 
            pending_liness_notify[] && return
            nowt = time()
            if nowt - last_notify_time[] < min_dt
                pending_liness_notify[] = true
                timeout_add_compat(Int(round(min_dt * 1000))) do
                    pending_liness_notify[] = false
                    last_notify_time[] = time()
                    try
                        Observables.notify(liness)
                    catch e
                        @error "notify(liness) failed" exception = (e, catch_backtrace())
                    end
                    return false
                end
                return
            end

            pending_liness_notify[] = true
            idle_add_compat() do
                pending_liness_notify[] = false
                last_notify_time[] = time()
                try
                    Observables.notify(liness)
                catch e
                    @error "notify(liness) failed" exception = (e, catch_backtrace())
                end
                return false
            end
        end


        function request_hist_update()
            hist_pending[] = true
            hist_running[] && return

            spec_obs[] === nothing && return false

            hist_running[] = true
            timeout_add_compat(50) do
                if !hist_pending[]
                    hist_running[] = false
                    return false
                end
                hist_pending[] = false

                lns_copy = deepcopy(liness[])  # capture 

                # Makie Figure on gtk thread 
                idle_add_compat() do
                    try
                        fig = three(spec_obs[], lns_copy)
                        @guarded draw(usvcanvas) do _
                            fill!(usvcanvas, colorant"white")
                            drawonto(usvcanvas, fig)
                        end
                    catch e
                        @error "hist update failed" exception = (e, catch_backtrace())
                    end
                    return false
                end

                return true
            end
        end

        function refresh_timewidgets!()
            tw = float(timescale.observable[])
            spixel = max(1, floor(Int, tw / dt_sng_obs[]))
            npixel = size(fout_obs[], 2)
            maxstart = max(npixel - spixel + 1, 1)
            timespan.observable[] = clamp(round(Int, timespan.observable[]), 1, maxstart)
            setindex!(timespan, 1:maxstart)
            setindex!(timespin, 1:100:maxstart)
        end

        # single example usv 
        # sng 
        singlesngdisbox = b["singlesngdis"] # box 
        singleusvcanvas = canvas(UserUnit) # canvas 
        push!(singlesngdisbox, singleusvcanvas)

        img1 = rand(257, 200)
        singleimg = Observable(img1)

        singledraw = draw(singleusvcanvas, singleimg) do scnvs, sim
            try
                fill!(scnvs, colorant"white")   # background is white 
                set_coordinates(scnvs, BoundingBox(0, 1, 0, 1))  # set coords to 0..1 along each axis 
                sctx = getgc(scnvs)
                copy!(sctx, sim)
            catch e
                @error "single example drawing failed" exception = (e, catch_backtrace())
            end
        end

        # spin button and scale 
        singlesngindbox = b["singlesngind"]

        usvboxind = Observable(1)

        # obs = the number of total usv 
        if isempty(liness[]) # if there is no usv 
            obs = Observable(1)
            singleslider = slider(1:2; observable=usvboxind)
            singlespin = spinbutton(1:2; observable=usvboxind)
        else # if there is usv
            numusv = length(liness[])
            obs = Observable(numusv)
            smax = max(numusv, 2) # slider needs min < max
            singleslider = slider(1:smax; observable=usvboxind)
            singlespin = spinbutton(1:smax; observable=usvboxind)
        end

        push!(singlesngindbox, singleslider) # slider    
        push!(singlesngindbox, singlespin)

        # single idx widgets click focus preventation 
        prevent_focus_on_click!(singleslider)
        prevent_focus_on_click!(singlespin)

        # multiple 
        usvparambox = b["usvparambox"]
        usvcanvas = GtkCanvas()
        # GTK4: box 안의 canvas 는 expand 를 켜야 보임
        Gtk4.hexpand(usvcanvas, true)
        Gtk4.vexpand(usvcanvas, true)
        push!(usvparambox, usvcanvas)

        gui_idle("initial hist") do
            request_hist_update()
        end


        # initialized 
        @guarded draw(usvcanvas) do w
            try
                ctx = getgc(usvcanvas)
                fill!(usvcanvas, colorant"white")
                fig1 = three(spec_obs[], liness[])
                drawonto(usvcanvas, fig1)
            catch e
                @error "usvcanvas initialization failed" exception = (e, catch_backtrace())
            end
        end


        # fill the boxes for air and spectrogram display 
        aircanvas = canvas(UserUnit)
        push!(airdis, aircanvas)

        @guarded draw(aircanvas) do widget
            try
                ctx = getgc(aircanvas)
                fill!(aircanvas, colorant"black")
                set_coordinates(aircanvas, BoundingBox(0, 1, 0, 1))
                copy!(ctx, rand(100, 1000))
            catch e
                @error "aircanvas draw failed" exception = (e, catch_backtrace())
            end
        end

        updating_timewidgets = Ref(false)

        # time span slider 
        npixel = size(fout_obs[], 2)
        spanpixel = Int(floor(2.0 / dt_sng_obs[]))
        maxstart0 = max(npixel - spanpixel, 1)
        timespan = slider(1:maxstart0)
        # 값 표시 끄기 (frame 번호라 의미 없음; 시작/끝 초는 spectrogram 아래에 표시) → 작은 화면에서 세로 공간 절약
        try
            Gtk4.draw_value(GtkObservables.widget(timespan), false)
        catch
        end
        push!(timespanbox, timespan)

        # time scale slider 
        timescale = slider(0.1:0.1:15.0)
        push!(timescalebox, timescale)

        # time spin button 
        timespin = spinbutton(1:100:(npixel-spanpixel); observable=timespan.observable)
        push!(currentindexbox, timespin)

        # time widgets click focus preventation 
        prevent_focus_on_click!(timespan)
        prevent_focus_on_click!(timescale)
        prevent_focus_on_click!(timespin)

        # air yaxis scale 
        airscale = slider(0.1:0.1:3.0, orientation="vertical")
        push!(airscalebox, airscale)

        # sng yaxis scale 
        sngscale = slider(0.01:0.01:1.0, orientation="vertical")
        push!(sngscalebox, sngscale)

        # vertical scale widgets click focus preventation 
        prevent_focus_on_click!(airscale)
        prevent_focus_on_click!(sngscale)


        # sng canvas 
        sngcanvas = canvas(UserUnit)
        push!(sngdis, sngcanvas)

        # 2026-02-19 focus policy: 
        # - 키/휠 단축 입력은 win에서 전역 처리 
        # - sngcanvas는 마우스 드래그/그리기만 담당하므로 포커스를 갖지 않게 함 
        sngw = widget(sngcanvas)
        prevent_focus_on_click!(sngw)

        # GTK4: 키/휠은 window 의 event controller 가 CAPTURE 단계에서 먼저 받음
        #   → 어느 위젯에 포커스가 있어도 단축키가 동작 (GTK3 의 "클릭하면 win 으로 포커스" 불필요)

        function do_load() # 2026-03-04 added feature 
            safe_cb("load clicked") do
                # GTK4: dialog 는 비동기 → 나머지는 callback 안에서
                open_dialog("Pick a .h5 recording or .jld annotation", win) do newnm
                    (newnm === nothing || newnm == "") && return
                    safe_cb("load file") do
                        load_after_dialog!(newnm)
                    end
                end
            end
        end

        function load_after_dialog!(newnm::AbstractString)
            begin
                try
                    load_into_gui!(newnm)
                catch e
                    e isa FileLoadError || rethrow()
                    report_load_error(e)
                    println("Still showing: ", nm_ref[])
                    return
                end

                # reset 
                drawing[] = false
                moving[] = false
                modifying[] = false
                newline[] = [XY{UserUnit}(0.0, 0.0), XY{UserUnit}(0.0, 0.0)]

                refresh_view!()

                set_dirty!(false)
                println("Loaded: ", newnm)
            end
        end

        # spectrogram 이나 box 가 바뀐 뒤 slider 범위 / 화면 / histogram / 단일 USV 패널 갱신
        function refresh_view!()
            obs[] = max(length(liness[]), 1)
            usvboxind[] = clamp(usvboxind[], 1, obs[])

            refresh_timewidgets!()

            # renewal 
            ts = round(Int, timespan.observable[])
            tw = float(timescale.observable[])

            currtime[] = t_sng_obs[][ts]

            set_text!(timestartlabel, "$(t_sng_obs[][ts]) sec ")
            set_text!(timeendlabel, "$(t_sng_obs[][ts] + tw) sec ")

            obsimg[] = update_sng(fout_obs[], ts, tw, sngscale.observable[]; dt_sng=dt_sng_obs[])

            # notify/redraw 
            gui_idle("after data change") do
                request_liness_notify()
                request_hist_update()
                Observables.notify(usvboxind)  # single panel update trigger 
            end
            return nothing
        end

        # analog channel 바꾸기: 같은 .h5 의 다른 channel 로 spectrogram 만 다시 계산, box 는 유지.
        #   실패하면 (예: channel 이 하나뿐인 파일) 원래 channel 로 되돌리고 false.
        function switch_channel!(n::Int)
            n == choice && return true
            old = choice
            try
                path = nm_ref[]
                dotloc = findlast('.', path)
                nmh5 = endswith(lowercase(path), ".jld") ? string(path[1:dotloc-1], ".h5") : path
                println("Switching to channel $n ...")
                global choice = n
                spec, _ = load_file(nmh5, n)      # .h5 → box 없이 spectrogram 만
                set_spec!(spec)
                refresh_view!()
                update_window_title!(win, nm_ref[]; dirty=dirty_ref[])
                println("Channel $n loaded (boxes kept: $(length(liness[])))")
                return true
            catch e
                global choice = old
                @error "channel switch to $n failed (does the file have that analog channel?); staying on channel $old" exception = (e, catch_backtrace())
                return false
            end
        end

        function do_delete()
            safe_cb("delete clicked") do
                if !isempty(liness[])
                    temp = usvboxind[]
                    deletebox(liness, temp)

                    obs[] = max(length(liness[]), 1)
                    usvboxind[] = clamp(usvboxind[], 1, obs[])
                    println("$(temp) th USV was removed")

                    gui_idle("after delete") do
                        request_liness_notify()
                        request_hist_update()

                        #Observables.notify(usvboxind) 
                    end
                    set_dirty!(true)
                else
                    println("no box to be removed")
                end
            end
        end

function do_detection()
    safe_cb("automatic USV detection") do
        spec = spec_obs[]
        if spec === nothing
            println("No file loaded.")
            return
        end
        println("Automatic USV detection starts...")

        f  = spec_freq_axis(spec)
        fh = Float64(maximum(f))
        t  = t_sng_obs[]
        dt = dt_sng_obs[]

        # --- 1) SDTS: 짧은 broadband transient 억제 ---
        fout_in = Matrix{Float64}(spec.fout)
        if SDTS_ENABLED[]
            println("SDTS denoising (alpha=$(SDTS_ALPHA[]), m=$(SDTS_M[]))...")
            @time fout_in = sdts_filter(fout_in;
                alpha = SDTS_ALPHA[], m = SDTS_M[], a = SDTS_A[])
        end

        # --- 2) 기존 denoise + detection (SDTS 결과를 입력으로) ---
        sng = denoise_usv(fout_in, f; fhigh=fh)
        det = usv_detect_2d(sng, dt, t, f;
            fhigh = fh,
            harmonics = true,
            harm_sng = snr_spectrogram(fout_in),        # ridge 검출: SDTS 맵
            measure_sng = spec.fout,                    # 경계 측정: 원본 맵 (SDTS 가 ridge 폭을 깎지 않은 것)
            measure_floor = row_noise_floor(spec.fout),
            harm_kw = HARM_KW_MORE)

        boxes = bounds_to_liness(det.ON, det.OFF, det.bounds, (; f=f, t_sng=t_sng_obs[]))

        println("$(length(boxes)) USVs detected.")
        liness[] = boxes
        obs[] = max(length(liness[]), 1)
        usvboxind[] = 1
        set_dirty!(true)

        gui_idle("after automatic detection") do
            request_liness_notify()
            request_hist_update()
            Observables.notify(usvboxind)
        end
        println("Detection result loaded into GUI.")
    end
end

        function do_save()
            safe_cb("save clicked") do

                cur_nm = nm_ref[]
                dotloc = findlast('.', cur_nm)
                nmjld = (dotloc === nothing) ? string(cur_nm, ".jld") : string(cur_nm[1:(dotloc-1)], ".jld")

                linesave = deepcopy(liness[])
                sequence_check!(linesave)

                println("save starts...waiting")
                pure = strip_gui_types(linesave)
                JLD.save(nmjld, "liness", pure, "y_unit", "ratio")

                export_images_after_save(nmjld, linesave)

                set_dirty!(false)
                println("save is done: okay to close the window")
            end
        end

        # 현재 image mode 로 이미지 + metadata.txt 저장 (lite 면 아무것도 안 함)
        function export_images_after_save(nmjld::AbstractString, lns)
            spec = spec_obs[]
            spec === nothing && return nothing
            w, h = Gtk4.width(sngw), Gtk4.height(sngw)
            live = (w > 1 && h > 1) ? (Int(w), Int(h)) : nothing
            return run_image_export(image_mode[], nmjld, spec, liness_to_boxes(lns, spec);
                                    sr=float(sngscale.observable[]), channel=choice,
                                    live_canvas_px=live)
        end

        # delete button 
        deletecb = b["delete"]
        signal_connect(deletecb, "clicked") do _
            do_delete()
        end
        # save button 
        savecb = b["save"]
        signal_connect(savecb, "clicked") do _
            do_save()
        end
        # load button 
        loadcb = b["load"]
        signal_connect(loadcb, "clicked") do _
            do_load()
        end
        # detection button
        detectioncb = b["detection"]
        signal_connect(detectioncb, "clicked") do _
            do_detection()
        end

        # --- 창을 연 채로 바꾸는 설정 (2026-11): analog channel / image mode -------------
        #   버튼 열 (Load / Save / USV detection / Delete) 아래에 dropdown 2 개.
        #   channel: 같은 파일의 다른 channel 로 spectrogram 재계산 (box 유지, 저장 안 된 box 도 유지)
        #   image mode: 다음 Save 부터 적용
        funcbox = b["function"]
        ch_names = ["Channel 1", "Channel 2"]
        chdd = dropdown(ch_names; value=ch_names[clamp(choice, 1, length(ch_names))])
        mode_label(name) = "Images: " * name
        modedd = dropdown(mode_label.(image_mode_names()); value=mode_label(image_mode[].name))
        push!(funcbox, GtkObservables.widget(chdd))
        push!(funcbox, GtkObservables.widget(modedd))
        prevent_focus_on_click!(chdd)      # 화살표 키가 dropdown 대신 창 단축키로 가도록
        prevent_focus_on_click!(modedd)

        reverting_channel = Ref(false)
        on(observable(chdd)) do sel
            (reverting_channel[] || sel === nothing) && return
            n = parse(Int, last(split(sel)))
            n == choice && return
            gui_idle("channel change") do
                if !switch_channel!(n)
                    reverting_channel[] = true
                    try
                        chdd[] = ch_names[choice]      # dropdown 을 실제 channel 로 되돌림
                    finally
                        reverting_channel[] = false
                    end
                end
            end
        end

        on(observable(modedd)) do sel
            sel === nothing && return
            name = replace(sel, "Images: " => "")
            name == image_mode[].name && return
            safe_cb("image mode change") do
                # CLI 의 --timescale 등은 gui 계열 모드에만 의미가 있음 (다른 모드면 경고 없이 무시)
                ov = find_image_mode(name).style === :gui ? overrides : (;)
                image_mode[] = image_mode_config(name; overrides=ov)
                println("Image mode: ", image_mode[].name, "  (", image_mode[].desc, ")")
                update_window_title!(win, nm_ref[]; dirty=dirty_ref[])
            end
        end

        function step_timespan!(d::Int)
            gui_idle("step_timespan") do
                safe_cb("step_timespan") do
                    cur = Int(round(timespan.observable[])) # Starting idx 

                    npixel = size(fout_obs[], 2)
                    spixel = max(1, round(Int, (float(timescale.observable[])) / dt_sng_obs[]))
                    maxstart = max(npixel - spixel + 1, 1)

                    timespan.observable[] = clamp(cur + d, 1, maxstart)
                end
            end
        end

        function step_box!(d::Int) # up +1, down -1 
            gui_idle("step_box") do
                safe_cb("step_box") do
                    usvboxind[] = clamp(usvboxind[] + d, 1, obs[])
                end
            end
        end

        eck = _capture_phase!(GtkEventControllerKey(win))
        signal_connect(eck, "key-pressed") do controller, keyval, keycode, state
            key = keyval
            ctrl = _ctrl(state)

            # Ctrl+Z == delete (또는 Delete/BackSpace) 
            if (ctrl && (key == GDK_KEY_z || key == GDK_KEY_Z)) ||
                key == GDK_KEY_Delete || key == GDK_KEY_BackSpace
                do_delete()
                return true
            end

            # Ctrl+S == save 
            if ctrl && (key == GDK_KEY_s || key == GDK_KEY_S)
                do_save()
                return true
            end

            # Left/Right == timespan 이동 
            if key == GDK_KEY_Left
                step_timespan!(-1000)  # 속도는 여기서 조절 
                return true
            elseif key == GDK_KEY_Right
                step_timespan!(+1000)
                return true
            end

            # Up/Down == 박스 선택 (Up=다음, Down=이전) 
            if key == GDK_KEY_Up
                step_box!(+1)
                return true
            elseif key == GDK_KEY_Down
                step_box!(-1)
                return true
            end

            return false
        end


        # Mouse wheel == timespan control :: down = expand, up = shrink 
        function zoom_timescale!(factor::Float64)
            gui_idle("zoom_timescale") do
                safe_cb("zoom_timescale") do
                    # current window center time maintaining 
                    ts = Int(round(timespan.observable[]))
                    tw = float(timescale.observable[])
                    center_t = t_sng_obs[][clamp(ts + Int(round((tw / dt_sng_obs[]) / 2)), 1, length(t_sng_obs[]))]

                    new_tw = clamp(tw * factor, 0.1, 15.0)
                    timescale.observable[] = new_tw

                    spixel_new = max(1, Int(floor(new_tw / dt_sng_obs[])))
                    center_i, = find_the_time(center_t, t_sng_obs[])
                    new_ts = center_i - Int(floor(spixel_new / 2))

                    npixel = size(fout_obs[], 2)
                    maxstart = max(npixel - spixel_new + 1, 1)
                    timespan.observable[] = clamp(new_ts, 1, maxstart)
                end
            end
        end

        function on_wheel(dy)
            if dy > 0 # scroll donw = expand 
                zoom_timescale!(1 / 1.15)
                return true
            elseif dy < 0
                zoom_timescale!(1.15)
                return true
            end
            return false
        end
        # 창의 빈 곳 (bubble 단계: slider 는 자기 스크롤을 먼저 처리, GTK3 와 같음)
        ecs = GtkEventControllerScroll(Gtk4.EventControllerScrollFlags_VERTICAL, win)
        signal_connect(ecs, "scroll") do controller, dx, dy
            on_wheel(dy)
        end
        # spectrogram / air canvas: GtkObservables 의 scroll handler 가 이벤트를 소비하므로 CAPTURE 로 먼저
        for cw in (sngw, widget(aircanvas))
            ecc = _capture_phase!(GtkEventControllerScroll(Gtk4.EventControllerScrollFlags_VERTICAL, cw))
            signal_connect(ecc, "scroll") do controller, dx, dy
                on_wheel(dy)
            end
        end


        # initial image update 
        img = update_sng(fout_obs[], timespan.observable[], timescale.observable[], sngscale.observable[])
        obsimg = Observable(img)

        # update the current image container for plot 
        sl = onany(timespan, timescale, sngscale) do timestart, timewidth, sngrange
            gui_idle("timespan/timescale/sngscale changed") do
                safe_cb("timespan/timescale/sngscale changed") do
                    timewidth = float(timewidth)
                    # time label 
                    spixel = max(1, round(Int, (timewidth / dt_sng_obs[])))
                    npixel = size(fout_obs[], 2)
                    maxstart = max(npixel - spixel + 1, 1)

                    ts = clamp(timestart, 1, maxstart)

                    currtime[] = t_sng_obs[][ts]
                    set_text!(timestartlabel, "$(t_sng_obs[][ts]) sec ")
                    set_text!(timeendlabel, "$(t_sng_obs[][ts] + timewidth) sec ")

                    # plot 
                    obsimg[] = update_sng(fout_obs[], ts, timewidth, sngrange; dt_sng=dt_sng_obs[])
                end
            end
        end

        # when time scales has changed, then the limit of timespan changes as well 

        on(timescale) do val
            gui_idle("timescale changed") do
                updating_timewidgets[] && return
                updating_timewidgets[] = true
                try
                    val = float(val)
                    npixel = size(fout_obs[], 2)
                    spixel = max(1, Int(floor(val / dt_sng_obs[])))
                    maxstart = max(npixel - spixel + 1, 1)

                    # slider range 
                    setindex!(timespan, 1:maxstart)
                    timespan.observable[] = clamp(round(Int, timespan.observable[]), 1, maxstart)

                    # spinbutton range 
                    setindex!(timespin, 1:100:maxstart)
                finally
                    updating_timewidgets[] = false
                end
            end
        end

        on(obs) do val
            try
                safe_cb("obs changed") do
                    val = max(val, 1)
                    if usvboxind.val > val
                        usvboxind[] = val
                    end
                    setindex!(singleslider, 1:val)
                    setindex!(singlespin, 1:val)
                end
            catch e
                @error "obs changing update failed" exception = (e, catch_backtrace())
            end
        end

        currentusvindex = b["currentusvindex"]

        onany(usvboxind, obs) do current, total
            gui_idle("usvboxind/obs changed") do
                set_text!(currentusvindex, "Current / total:\n $(current) / $(total) ")

                if isempty(liness[]) || current < 1 || current > length(liness[])
                    return
                end

                cctime = liness[][current][1].x.val
                cctime2 = liness[][current][2].x.val

                (isfinite(cctime) && isfinite(cctime2)) || return

                ti1, = find_the_time(cctime, t_sng_obs[])
                singleimg[] = update_sng(fout_obs[], ti1, cctime2 - cctime, sngscale.observable[])

                if current <= total
                    # current usv features 
                    sdur, sdB, spur, smeanF, spitchV = single_usvfeatures(spec_obs[], liness[], current)
                    sdur = Int(round(sdur))
                    sdB = Int(round(sdB))
                    spur = round(spur, sigdigits=3)
                    smeanF = Int(round(smeanF / 1000))
                    spitchV = Int(round(spitchV))
                    set_text!(currentusvdur, "$(sdur)ms ")
                    set_text!(currentusvamp, "$(sdB)dB ")
                    set_text!(currentusvspur, "$(spur) ")

                    set_text!(currentusvpitch, "$(smeanF)kHz ")
                    set_text!(currentusvpitchvariance, "$(spitchV) log₁₀Hz² ")
                end

                if cctime < currtime[] || cctime > currtime[] + timescale.observable[]
                    ti, = find_the_time(cctime, t_sng_obs[])
                    npixel = size(fout_obs[], 2)
                    spixel = max(1, Int(floor(timescale.observable[] / dt_sng_obs[])))
                    maxstart = max(npixel - spixel, 1)
                    timespan.observable[] = clamp(ti - 10, 1, maxstart)
                end
            end
        end

        # flag 
        drawing = Observable(false)  # this will be true if we're dragging a new line 
        moving = Observable(false)
        modifying = Observable(false)

        newline = Observable([XY{UserUnit}(0.0, 0.0), XY{UserUnit}(0.0, 0.0)]) # the in-progress line (will be added to list above) 
        click_pivot = Observable([XY{UserUnit}(0.0, 0.0)])

        sigstart = on(sngcanvas.mouse.buttonpress) do btn
            try
                safe_cb("mouse buttonpress") do
                    if btn.button == 1 && _plain(btn.modifiers)
                        drawing[] = true   # start extending the line 
                        newline[][1] = btn.position
                    end

                    if btn.button == 1 && _shift(btn.modifiers) && !_ctrl(btn.modifiers) && !_alt(btn.modifiers)
                        click_pivot[] = [unit_to_time(btn.position, currtime[], timescale.observable[])]

                        flag, id = checkmoving(btn, currtime, liness, timescale)
                        if flag
                            moving[] = true
                            usvboxind[] = id
                        end

                    end
                    if btn.button == 3
                        flag, id = checkmoving(btn, currtime, liness, timescale)

                        if flag
                            modifying[] = true
                            usvboxind[] = id
                        end
                    end
                end
            catch e
                @error "mouse buttonpress failed" exception = (e, catch_backtrace())
            end
        end

        # const dummybutton = MouseButton{UserUnit}() 
        sigextend = on(sngcanvas.mouse.motion) do btn
            try
                safe_cb("mouse motion") do
                    if drawing[]
                        newline[][2] = btn.position
                        request_liness_notify()
                    end
                    if moving[]
                        bb = mov(btn, currtime, timescale, click_pivot)
                        updatebox(bb, liness, usvboxind)
                        click_pivot[] = [unit_to_time(btn.position, currtime[], timescale.observable[])]
                        set_dirty!(true)
                        request_liness_notify()
                    end

                    if modifying[]
                        liness[][usvboxind[]][2] =
                            unit_to_time(btn.position, currtime[], timescale.observable[])
                        set_dirty!(true)
                        request_liness_notify()
                    end
                end
            catch e
                @error "mouse motion failed" exception = (e, catch_backtrace())
            end
        end

        sigend = on(sngcanvas.mouse.buttonrelease) do btn
            try
                safe_cb("mouse buttonrelease") do
                    if btn.button == 1 || btn.button == 3

                        moving[] = false

                        # if the new location goes before the first one then switch 
                        if modifying[]
                            if liness[][usvboxind[]][1].x > liness[][usvboxind[]][2].x
                                liness[][usvboxind[]][1], liness[][usvboxind[]][2] = liness[][usvboxind[]][2], liness[][usvboxind[]][1]
                            end
                        end
                        modifying[] = false

                        if !isempty(newline[]) && drawing[]
                            # if the second one is the earlier one then swap 
                            if newline[][1].x > newline[][2].x
                                newline[][1], newline[][2] = newline[][2], newline[][1]
                            end

                            push!(liness[], unit_to_time(deepcopy(newline[]), currtime[], timescale.observable[]))
                            usvboxind[] = length(liness[])
                            obs[] = length(liness[])

                        end
                        drawing[] = false  # stop extending the line 

                        gui_idle("after box commit") do
                            try
                                #Observables.notify(liness) 
                                #Observables.notify(usvboxind) 
                                #hist_tick[] += 1 
                                request_hist_update()
                                set_dirty!(true)
                            catch e
                                @error "after commit failed" exception = (e, catch_backtrace())
                            end
                            false
                        end
                    end
                end
            catch e
                @error "mouse buttonrelease failed" exception = (e, catch_backtrace())
            end
        end

        # Draw on the canvas 
        redraw = draw(sngcanvas, liness, newline, obsimg, usvboxind) do cnvs, lns, newl, imgss, ubi
            try
                w, h = Gtk4.width(cnvs), Gtk4.height(cnvs)
                (w <= 1 || h <= 1) && return

                fill!(cnvs, colorant"white")   # background is white 
                set_coordinates(cnvs, BoundingBox(0, 1, 0, 1))  # set coords to 0..1 along each axis 
                ctx = getgc(cnvs)

                if size(imgss, 1) > 0 && size(imgss, 2) > 0
                    Cairo.save(ctx)
                    Cairo.rectangle(ctx, 0, 0, 1, 1)
                    Cairo.clip(ctx)
                    copy!(ctx, imgss)
                    Cairo.restore(ctx)
                end

                draw_freq_axis!(ctx, Int(w), Int(h), spec_obs[]; n_ticks=6, font_px=12, show_khz=true)


                n = length(lns)
                n == 0 && return
                idx = clamp(ubi, 1, n)

                for (i, l) in enumerate(lns)
                    try
                        if length(l) >= 2
                            ll = time_to_unit(l, currtime[], timescale.observable[])

                            x1 = clamp(ll[1].x.val, 0, 1)
                            x2 = clamp(ll[2].x.val, 0, 1)
                            y1 = clamp(ll[1].y.val, 0, 1)
                            y2 = clamp(ll[2].y.val, 0, 1)

                            x1, x2 = min(x1, x2), max(x1, x2)
                            y1, y2 = min(y1, y2), max(y1, y2)

                            col = (i == idx) ? colorant"orange" : colorant"blue"
                            drawline(ctx, [XY{UserUnit}(x1, y1), XY{UserUnit}(x2, y2)], col)
                        end
                    catch e
                        @debug "Box $i draw failed" exception = e
                    end
                end


                if drawing[] && length(newl) >= 2
                    drawline(ctx, newl, colorant"red")
                end

                if moving[] && idx <= n && length(lns[idx]) >= 2
                    pp = time_to_unit(lns[idx], currtime[], timescale.observable[])
                    drawline(ctx, pp, colorant"green")
                end

                if modifying[] && idx <= n && length(lns[idx]) >= 2
                    pp = time_to_unit(lns[idx], currtime[], timescale.observable[])
                    drawline(ctx, pp, colorant"yellow")
                end

            catch e
                @error "sngcanvas draw failed" exception = (e, catch_backtrace())
            end
            return nothing
        end

        signal_connect(win, :close_request) do _
            notify(GUI_CLOSED)
            return false          # false = 창 닫기 허용
        end
        update_window_title!(win, nm_ref[]; dirty=dirty_ref[])
        size_window!(win, window) # 화면에 맞춘 시작 크기 (이후 자유롭게 조절 가능)
        present(win)              # GTK4: showall 없음 (위젯은 기본으로 보임)
        return win
    catch e
        if e isa FileLoadError
            report_load_error(e)
            return :load_failed
        end
        @error "GUI initialization crashed BEFORE setup" exception = (e, catch_backtrace())
        println(stderr, "\n=== CRITICAL INIT ERROR ===\n")
        showerror(stderr, e)
        println(stderr, "\nStacktrace:")
        Base.show_backtrace(stderr, catch_backtrace())
        println(stderr, "\n===========================\n")
        return nothing
    end
end

function deletebox(lines, ind)
    if lines isa Observable
        deleteat!(lines[], ind)
        # Observables.notify(lines) 
    else
        deleteat!(lines, ind)
    end
    nothing
end

function drawonto(canvas, figure)
    try
        w = Gtk4.width(canvas)
        h = Gtk4.height(canvas)
        (w <= 1 || h <= 1) && return

        surf = Gtk4.cairo_surface(canvas)
        surf === nothing && return

        #@guarded draw(canvas) do _ 
        scene = figure.scene
        #resize!(scene, Gtk.width(canvas), Gtk.height(canvas)) 
        resize!(scene, w, h)
        config = CairoMakie.ScreenConfig(1.0, 1.0, :good, true, true, nothing)
        screen = CairoMakie.Screen(scene, config, surf)
        CairoMakie.cairo_draw(screen, scene)
    catch e
        @error "drawonto failed" exception = (e, catch_backtrace())
    end
end

# y-axis overlay (Hz) for spectrogram canvas 
function draw_freq_axis!(ctx, w::Int, h::Int, spec;
    n_ticks::Int=6, x_px::Int=36, tick_px::Int=8,
    font_px::Int=12, show_khz::Bool=true)

    # ---- get fmax ---- 
    fmax = nothing
    if hasproperty(spec, :f_sng)
        try
            fmax = maximum(getproperty(spec, :f_sng))
        catch
        end
    end
    if fmax === nothing && hasproperty(spec, :fs)
        fmax = getproperty(spec, :fs) / 2
    end
    fmax === nothing && return
    fmaxf = float(fmax)

    Cairo.save(ctx)

    # pixel coordinates 
    oldm = Cairo.get_matrix(ctx)
    Cairo.set_matrix(ctx, Cairo.CairoMatrix(1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    # background box 
    Cairo.set_source_rgba(ctx, 1, 1, 1, 0.75)
    Cairo.rectangle(ctx, 0, 0, x_px + 60, h)
    Cairo.fill(ctx)

    # style 
    set_source(ctx, colorant"black")
    set_line_width(ctx, 2.0)
    select_font_face(ctx, "Sans", Cairo.FONT_SLANT_NORMAL, Cairo.FONT_WEIGHT_NORMAL)
    set_font_size(ctx, float(font_px))

    # axis line 
    move_to(ctx, x_px, 0)
    line_to(ctx, x_px, h)
    stroke(ctx)

    n_ticks = max(n_ticks, 2)
    for k in 0:(n_ticks-1)
        f = (k / (n_ticks - 1)) * fmaxf
        y = (h - 1) - (f / fmaxf) * (h - 1)   # 위=고주파( reverse!(dims=1) 기준 ) 

        # tick 
        move_to(ctx, x_px, y)
        line_to(ctx, x_px + tick_px, y)
        stroke(ctx)

        # label 
        label_val = show_khz ? f / 1000 : f
        #unit = show_khz ? "k" : "" 
        txt = string(Int(round(label_val)))

        move_to(ctx, x_px + tick_px + 6, clamp(y + font_px * 0.35, 0, h - 1))
        show_text(ctx, txt)
    end

    # unit 
    move_to(ctx, 4, font_px + 4)
    show_text(ctx, show_khz ? "kHz" : "Hz")

    Cairo.set_matrix(ctx, oldm)
    Cairo.restore(ctx)
end


function three(spec, lns)
    try
        nm = :probability
        if length(lns) == 0
            # pseudo data for initial plot 
            duration = rand(Normal(100, 100), 1000)
            amp = rand(Normal(0, 10), 1000)
            spur = rand(Normal(0.3, 1), 1000)
            meanf = rand(Normal(60_000, 10000), 1000)
            pitchvar = rand(Normal(3, 5), 1000)
        else
            duration = zeros(length(lns))
            amp = zeros(length(lns))
            spur = zeros(length(lns))
            meanf = zeros(length(lns))
            pitchvar = zeros(length(lns))

            gui_usvfeatures!(spec, lns, duration, amp, spur, meanf, pitchvar)
        end
        fig = Figure()

        whole = fig[1, 1] = GridLayout()
        ax1 = CairoMakie.Axis(whole[1, 1])
        hist!(ax1, duration, bins=0:10:300, normalization=nm, color=(:blue, 0.5))
        xlims!(ax1, 0, 300)

        ax2 = Axis(whole[1, 2])
        hist!(ax2, amp, bins=-10:30, normalization=nm, color=(:darkred, 0.5))
        xlims!(ax2, -11, 31)

        ax3 = Axis(whole[1, 3])
        hist!(ax3, spur, bins=0:0.02:0.61, normalization=nm, color=(:orange, 0.5))
        xlims!(ax3, 0, 0.7)

        ax4 = Axis(whole[1, 4])
        hist!(ax4, meanf ./ 1000, bins=30:2:125, normalization=nm, color=(:green, 0.5))
        xlims!(ax4, 30, 125)

        ax5 = Axis(whole[1, 5])
        hist!(ax5, pitchvar, bins=5:0.1:9, normalization=nm, color=(:purple, 0.5))
        xlims!(ax5, 0, 9)

        ax1.xlabel = "duration (ms)"
        # ax1.xticks = ([0,150],["0","150"]) 
        # ax1.yticks =([0.0,0.1],["0","0.1"]) 
        ax2.xlabel = "loudness (dB)"
        # ax2.xticks = ([-10,0,20],["-10","0","20"]) 
        ax3.xlabel = "spectral purity (ratio)"
        ax4.xlabel = "mean frequency (kHz)"
        # ax4.xticks =([50,100],["50","100"]) 
        ax5.xlabel = "pitch varience [Log10(Hz²)]"


        ylims!(ax1, 0, 0.3)
        ylims!(ax2, 0, 0.3)
        ylims!(ax3, 0, 0.3)
        ylims!(ax4, 0, 0.3)
        ylims!(ax5, 0, 0.3)
        return fig
    catch e
        @error "three() plot generation failed" exception = (e, catch_backtrace())
        fig = Figure()
        Axis(fig[1, 1], title="Plot generation failed")
        return fig
    end
end


function update_sng(fout::AbstractMatrix{<:Real}, ts::Int, tw::Real, sr::Real; dt_sng::Real=0.000512)

    _, ncol = size(fout)

    spixel = max(1, Int(floor(tw / dt_sng)))

    startcol = clamp(ts, 1, ncol)
    endcol = clamp(startcol + spixel - 1, startcol, ncol)

    temp = @view fout[:, startcol:endcol]
    sr = max(float(sr), eps(Float64))
    img = clamp.(Float32.(temp), 0f0, Float32(sr)) ./ Float32(sr)

    reverse!(img, dims=1)
    return img
end
update_sng(
    fout::AbstractMatrix{<:Real},
    ts::Int,
    tw::Real,
    sr::Real,
    dt_sng::Real;
    kwargs...,
) = update_sng(fout, ts, tw, sr; dt_sng=dt_sng)

function checkmoving(btn, currtime, liness, timescale)
    flag = false
    ind = 1
    for i in 1:length(liness[])
        f = first(time_to_unit(liness[][i], currtime[], timescale.observable[]))
        l = last(time_to_unit(liness[][i], currtime[], timescale.observable[]))

        xs = f.x.val
        xe = l.x.val

        ys = f.y.val
        ye = l.y.val

        if ys > ye # xs are already confirmed 
            ys, ye = ye, ys
        end

        if btn.position.x.val > xs && btn.position.x.val < xe && btn.position.y.val > ys && btn.position.y.val < ye
            flag = true
            ind = i
            break
        end
    end
    return flag, ind
end

# cacluate the difference of one step 
function mov(btn, currtime, timescale, click_pivot)

    btn_unit = unit_to_time(btn.position, currtime[], timescale.observable[])

    tempx = btn_unit.x.val
    tempy = btn_unit.y.val

    x = tempx - click_pivot[][1].x.val
    y = tempy - click_pivot[][1].y.val

    XY{UserUnit}(x, y) # as time 
end

# move the box with the step of bb 
function updatebox(bb::XY{UserUnit}, liness, usvboxind)
    i = usvboxind[]
    cl = liness[][i]
    movingbox(cl, bb)
end

# the actual function of the updatebox 
function movingbox(currline::Vector{XY{UserUnit}}, moving::XY{UserUnit})
    # i = 2 
    # currline = the start and last points of a box 
    for i in 1:length(currline)
        x = currline[i].x.val
        y = currline[i].y.val
        xnew = x + moving.x.val
        ynew = y + moving.y.val
        currline[i] = XY{UserUnit}(xnew, ynew)
    end
    return nothing
end

function drawline(ctx, l, color)
    isempty(l) && return
    p = first(l)
    pp = last(l)
    move_to(ctx, p.x, p.y)
    set_source(ctx, color)

    line_to(ctx, pp.x, p.y)
    line_to(ctx, pp.x, pp.y)
    line_to(ctx, p.x, pp.y)
    line_to(ctx, p.x, p.y)

    stroke(ctx)
    return nothing
end

function channel_select()
    println("Choose channel index (anl[1] or anl[2])")
    choice = nothing
    while !(choice in (1, 2))
        print("Enter 1 or 2: ")
        input = strip(readline())

        if isempty(input)
            println("Input was empty. Please enter 1 or 2.")
            continue
        end

        parsed = tryparse(Int, input)

        if parsed === nothing || !(parsed in (1, 2))
            println("Invalid input. Please enter 1 or 2.")
        else
            choice = parsed
        end
    end
    return choice
end

# ---------------------------------------------------------------------
# 진입점: 명령행 / REPL 공통
#   julia --project=. -t auto Jwi-eak.jl [FILE] [options]   (--help 참고)
#   REPL: include("Jwi-eak.jl") → 예전처럼 dialog 로 시작
#         main(["rec.h5", "--mode", "fixed"]) 처럼 옵션을 줘서 다시 열 수도 있음
# ---------------------------------------------------------------------
choice = 1   # analog channel (load / legacy export 에서 사용; --channel 로 설정)

# GUI 없이 .jld 의 box 로 이미지 + metadata.txt 만 만들기
function export_only_run(file::AbstractString, cfg; sr::Real, channel::Int)
    println("Export only: ", file, "  (mode ", cfg.name, ", channel ", channel, ")")
    spec, lns_obs = load_file(String(file), channel)
    lns = lns_obs isa Observable ? lns_obs[] : lns_obs
    sequence_check!(lns)
    return run_image_export(cfg, file, spec, liness_to_boxes(lns, spec); sr=sr, channel=channel)
end

function main(args::AbstractVector{<:AbstractString}=ARGS)
    opts = try
        parse_cli(args)
    catch e
        e isa CLIError || rethrow()
        println(stderr, "Error: ", e.msg, "\n")
        print_cli_help(stderr)
        isinteractive() || exit(2)
        return nothing
    end
    opts.help && (print_cli_help(); return nothing)
    opts.list_modes && (describe_image_modes(); return nothing)

    global choice = opts.channel
    cfg = opts.mode === nothing ? nothing : image_mode_config(opts.mode; overrides=opts.overrides)

    if opts.export_only
        return export_only_run(opts.file, cfg; sr=opts.sr, channel=opts.channel)
    end

    # GTK4: REPL 에서는 GLib main loop 가 자동으로 돌지만, 명령행 실행에서는 직접 시작해야 함.
    #       dialog 가 비동기라 loop 를 먼저 띄운 뒤 GUI 를 시작.
    script = !isinteractive()
    script && Gtk4.GLib.start_main_loop()

    make_gui(; file=opts.file, image_cfg=cfg, overrides=opts.overrides, window=opts.window)

    # 명령행 실행: 창이 닫히거나 (dialog 취소 / 시작 실패 포함) 할 때까지 기다림
    if script
        wait(GUI_CLOSED)
        println("Window closed.")
    end
    return nothing
end

main(ARGS)