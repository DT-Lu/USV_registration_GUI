# =====================================================================
# freq_bounds_benchmark.jl
#   주파수(세로) 경계 정확도 / 잡음 강건성 벤치마크
#
#   1) 합성 USV (정답 주파수 범위를 아는 콜) 를 white noise 에 SNR 별로 섞고,
#      정답 ON/OFF 를 그대로 줘서 "주파수축만" 평가 (시간축 detection 오차와 분리).
#   2) 설정별 오차 표 출력:
#        old  : temp9 동작 (quantile 경계, 정수 bin, SDTS 맵에서 측정)
#        +B+E : 서브빈 edge + 연속성 경계 (SDTS 맵에서 측정)
#        new  : +B+E + 원본 맵에서 측정 (A)   ← Jwi-eak.jl 기본
#   3) 수동 box(.jld) 와 비교하는 helper: compare_boxes
#
#   실행 (repository 폴더에서):
#       julia --project=. benchmark/freq_bounds_benchmark.jl
#   또는 REPL 에서 include 후 run_benchmark(; snrs=[20, 10, 5, 0], nrep=10)
# =====================================================================

using FFTW, DSP, LinearAlgebra, Statistics, Random, Printf, SparseArrays

const _ROOT = joinpath(@__DIR__, "..")
include(joinpath(_ROOT, "Necessities", "acoustic.jl"))
include(joinpath(_ROOT, "Necessities", "sdts_filter.jl"))
include(joinpath(_ROOT, "Necessities", "usv_detection_3.jl"))

const FS = 250_000

# --- 합성 콜 ----------------------------------------------------------
# finst(τ) : 순간 주파수 (Hz), τ ∈ [0, dur]. harm : (차수, 상대 진폭) 목록
struct SynthCall
    name::String
    dur::Float64
    finst::Function
    harm::Vector{Tuple{Int,Float64}}
end

const CALLS = [
    SynthCall("flat 70k",        0.030, τ -> 70e3,                                   Tuple{Int,Float64}[]),
    SynthCall("up 55-80k",       0.020, τ -> 55e3 + 25e3 * τ / 0.020,                Tuple{Int,Float64}[]),
    SynthCall("steep down 90-55k", 0.008, τ -> 90e3 - 35e3 * τ / 0.008,              Tuple{Int,Float64}[]),
    SynthCall("sinFM 65±8k",     0.040, τ -> 65e3 + 8e3 * sin(2π * 40 * τ),          Tuple{Int,Float64}[]),
    SynthCall("harm 40-48k (+2f)", 0.030, τ -> 40e3 + 8e3 * τ / 0.030,               [(2, 0.3)]),
]

# 정답 주파수 범위 (harmonic 포함)
function true_extent(c::SynthCall)
    τs = range(0, c.dur, length=2001)
    fs_ = c.finst.(τs)
    lo, hi = minimum(fs_), maximum(fs_)
    for (k, _) in c.harm
        hi = max(hi, k * maximum(fs_))
    end
    return lo, hi
end

# 콜 하나의 파형 (진폭 1, 양 끝 1 ms raised-cosine ramp)
function synth_wave(c::SynthCall)
    n = round(Int, c.dur * FS)
    τ = (0:n-1) ./ FS
    ϕ = 2π .* cumsum(c.finst.(τ)) ./ FS
    x = sin.(ϕ)
    for (k, a) in c.harm
        x .+= a .* sin.(k .* ϕ)
    end
    nr = round(Int, 0.001 * FS)
    w = ones(n)
    for i in 1:min(nr, n ÷ 2)
        w[i] = w[n-i+1] = 0.5 * (1 - cos(π * (i - 1) / nr))
    end
    return x .* w
end

# per-bin SNR(dB) → 사인파 진폭 (잡음 σ = 1).
#   Hann 창 N 점: tone peak |X|² = (A·N/4)²,  white noise E|X|² = Σw² = 3N/8
#   → SNR = A²·N/6  →  A = sqrt(6·SNR/N)
amp_for_snr(snr_db; nfft=512) = sqrt(6 * 10^(snr_db / 10) / nfft)

# 녹음 생성: 콜들을 gap 간격으로 배치. 반환 (audio, [(s_on, s_off, call)])
function synth_recording(calls, snr_db; gap=0.15, rng=Random.default_rng())
    A = amp_for_snr(snr_db)
    segs = Tuple{Int,Int,SynthCall}[]
    total = round(Int, gap * FS) * (length(calls) + 1) + sum(round(Int, c.dur * FS) for c in calls)
    audio = randn(rng, total)
    s = round(Int, gap * FS)
    for c in calls
        w = A .* synth_wave(c)
        audio[s+1:s+length(w)] .+= w
        push!(segs, (s + 1, s + length(w), c))
        s += length(w) + round(Int, gap * FS)
    end
    return audio, segs
end

# 샘플 구간 → STFT 컬럼 구간 (창 중심이 구간 안에 있는 컬럼)
function sample_to_cols(s_on, s_off, sp::stft_param, ncol)
    half = sp.nfft ÷ 2
    a = clamp(ceil(Int, (s_on - half) / sp.skip) + 1, 1, ncol)
    b = clamp(floor(Int, (s_off - half) / sp.skip) + 1, 1, ncol)
    return a, max(a, b)
end

# --- 설정 -------------------------------------------------------------
const BENCH_BASE = merge(HARM_KW_MORE, (; margin_bins=0, min_height_bins=0))

bench_configs(fout, nfl) = [
    ("old",  (; bound_mode=:quantile,   subbin=false)),
    ("+B+E", (; bound_mode=:continuous, subbin=true)),
    ("new",  (; bound_mode=:continuous, subbin=true, measure_sng=fout, measure_floor=nfl)),
]

# --- 실행 -------------------------------------------------------------
function run_benchmark(; snrs=[20, 10, 5, 0], nrep::Int=10, seed::Int=1,
                       sdts::Bool=true, verbose::Bool=true)
    rng = MersenneTwister(seed)
    sp = stft_param()
    # rows[(config, call, snr)] = [(Δlo, Δhi, IoU), ...]   Δ = 추정 − 정답 (kHz)
    rows = Dict{Tuple{String,String,Int},Vector{NTuple{3,Float64}}}()
    for snr in snrs, rep in 1:nrep
        audio, segs = synth_recording(CALLS, snr; rng=rng)
        spec = spec_data(audio, sp; fs=FS)
        f = spec.f
        fout = spec.fout
        fin = sdts ? sdts_filter(fout; alpha=SDTS_ALPHA[], m=SDTS_M[], a=SDTS_A[]) : fout
        harm_sng = snr_spectrogram(fin)
        nfl = row_noise_floor(fout)
        fh = Float64(maximum(f))
        for (s_on, s_off, c) in segs
            on, off = sample_to_cols(s_on, s_off, sp, size(fout, 2))
            tlo, thi = true_extent(c)
            for (name, kw) in bench_configs(fout, nfl)
                a = usv_harmonic_analysis(harm_sng, f, on, off; fhigh=fh, BENCH_BASE..., kw...)
                inter = max(0.0, min(a.f_hi, thi) - max(a.f_lo, tlo))
                uni = max(a.f_hi, thi) - min(a.f_lo, tlo)
                iou = uni > 0 ? inter / uni : 0.0
                push!(get!(rows, (name, c.name, snr), NTuple{3,Float64}[]),
                      ((a.f_lo - tlo) / 1e3, (a.f_hi - thi) / 1e3, iou))
            end
        end
    end
    verbose && print_table(rows, snrs)
    return rows
end

function print_table(rows, snrs)
    names = first.(bench_configs(nothing, nothing))
    for snr in snrs
        @printf("\n=== per-bin SNR %d dB ===   Δ = 추정 − 정답 (kHz, 평균 ± sd),  IoU 평균\n", snr)
        @printf("%-20s %-6s %16s %16s %7s\n", "call", "config", "Δf_lo", "Δf_hi", "IoU")
        for c in CALLS, nm in names
            v = get(rows, (nm, c.name, snr), nothing)
            v === nothing && continue
            dl = getindex.(v, 1); dh = getindex.(v, 2); io = getindex.(v, 3)
            @printf("%-20s %-6s %8.2f ± %5.2f %8.2f ± %5.2f %7.3f\n",
                    c.name, nm, mean(dl), std(dl), mean(dh), std(dh), mean(io))
        end
        for nm in names   # 전체 요약
            v = reduce(vcat, [get(rows, (nm, c.name, snr), NTuple{3,Float64}[]) for c in CALLS])
            @printf("%-20s %-6s |Δlo| %6.2f  |Δhi| %6.2f  IoU %6.3f\n", "  (all calls)", nm,
                    mean(abs.(getindex.(v, 1))), mean(abs.(getindex.(v, 2))), mean(getindex.(v, 3)))
        end
    end
end

# --- 수동 box 와 비교 ------------------------------------------------
# pred, ref : [(t1, t2, f_lo, f_hi), ...]  (초, Hz)
#   ref 마다 시간 IoU 가 가장 큰 pred 를 짝지음 (시간 IoU < min_tiou 면 miss).
#   반환: (; n_ref, n_matched, dlo_khz, dhi_khz, freq_iou)  — 짝지어진 것들의 벡터
#
# GUI 에서 쓰는 예 (Jwi-eak.jl 실행 후 REPL):
#   to_hz(l) = (Float64(l[1].x.val), Float64(l[2].x.val),
#               Float64(spec_obs[].f[userunit_to_freq_bin(Float64(l[2].y.val), size(spec_obs[].fout, 1))]),
#               Float64(spec_obs[].f[userunit_to_freq_bin(Float64(l[1].y.val), size(spec_obs[].fout, 1))]))
#   manual = to_hz.(liness[])            # 수동 box 를 불러온 상태에서 저장
#   (detection 버튼 실행 후)  auto = to_hz.(liness[])
#   r = compare_boxes(auto, manual); mean(abs.(r.dlo_khz)), mean(abs.(r.dhi_khz))
function compare_boxes(pred, ref; min_tiou::Float64=0.3)
    dlo = Float64[]; dhi = Float64[]; fiou = Float64[]
    for (r1, r2, rlo, rhi) in ref
        best, bk = 0.0, 0
        for (k, (p1, p2, _, _)) in enumerate(pred)
            inter = max(0.0, min(r2, p2) - max(r1, p1))
            uni = max(r2, p2) - min(r1, p1)
            tiou = uni > 0 ? inter / uni : 0.0
            tiou > best && ((best, bk) = (tiou, k))
        end
        (bk == 0 || best < min_tiou) && continue
        _, _, plo, phi = pred[bk]
        push!(dlo, (plo - rlo) / 1e3); push!(dhi, (phi - rhi) / 1e3)
        inter = max(0.0, min(phi, rhi) - max(plo, rlo))
        uni = max(phi, rhi) - min(plo, rlo)
        push!(fiou, uni > 0 ? inter / uni : 0.0)
    end
    return (; n_ref=length(ref), n_matched=length(dlo), dlo_khz=dlo, dhi_khz=dhi, freq_iou=fiou)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
