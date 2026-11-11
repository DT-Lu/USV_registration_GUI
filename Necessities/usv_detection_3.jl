# =====================================================================
# usv_detection.jl
#   USV detection (시간축) + 주파수(세로) 경계 탐지 통합 파일
#
#   구성
#     1. 공통 유틸 / 전처리 (denoise, feature, running median)
#     2. 시간축 detection (원본 usv_whole + 개선판 usv_whole_2d)
#     3. 주파수 경계 탐지
#          - :ridge     → usv_freq_bounds        (ridge argmax 분위수, 기존 방식)
#          - :threshold → detect_frequency_bounds (time-bin 별 noise/peak 임계, 구 TimeBinDetector)
#        두 방식 모두 usv_segment_freq_bounds 로 같은 인터페이스를 가짐
#          - harmonic   → usv_harmonic_analysis / usv_segment_harmonics
#                         (:ridge 에 n_peaks > 1 을 주면 box 가 harmonic 까지 포함)
#                         약한/고차 harmonic: snr_spectrogram(raw) + HARM_KW_MORE,
#                         Nyquist 위에서 접힌 harmonic: fs 지정
#     4. GUI 변환 (detection_to_liness / detection_to_liness_2d)
#     5. 한 번에 돌리는 pipeline (usv_detect_2d)
#
#   GUI 의존: XY, UserUnit, sequence_check! 는 이 파일 밖(GUI 코드)에 정의되어 있어야 함.
#             detection_to_liness* 를 쓰지 않으면 GUI 없이도 동작.
#
#   통합하면서 고친 점
#     - usv_preprocess 가 specdiscont 에서 sng 를 in-place 로 column 정규화하던 문제 수정
#       (copy 해서 계산). 이전에는 usv_whole_2d 의 band power / noise floor 가
#       정규화된 값(모든 컬럼 ≈ 1)으로 계산되어 amp-gate, confident-FP,
#       extend_boundaries 가 사실상 동작하지 않았음.
#     - StepRangeLen{Float64,TwicePrecision...} 로 묶인 타입 제한을 AbstractVector 로 완화
#       (usv_whole_2d 가 받는 일반 StepRangeLen 이 내부 함수에서 MethodError 나던 문제).
#     - ON/OFF 를 detect_frequency_bounds 에 넘기기 전에 [1, ncol] 로 clamp + 정렬
#       (putative_usv 는 ON=0 을 낼 수 있어 그대로 넘기면 ArgumentError).
#     - (2026-10) ridge box 를 peak 위치가 아니라 ridge 폭(edge)으로 계산, 약한 FM 꼬리는
#       hysteresis tracking 으로 채택, 연속성 필터는 고립점 제거로 (강한 ridge 위/아래 잘림 수정)
#     - (2026-10) box y 좌표를 1 - f/fmax 대신 GUI 의 bin 규약
#       y = (nfreq+1-bin)/nfreq 로 통일 (freq_bin_to_userunit / userunit_to_freq_bin).
#     - (2026-11) 주파수 경계 정밀화 (usv_harmonic_analysis):
#         measure_sng/measure_floor : ridge 는 SDTS 맵에서 찾고 폭은 원본 맵에서 측정
#         subbin                    : edge 를 dB 보간한 서브빈 위치로, contour 는 포물선 보간
#         bound_mode=:continuous    : 이웃 컬럼과 이어진 극단 edge 만 box 경계로 (튄 컬럼 1개 무시)
#       bound_mode=:quantile, subbin=false, measure_sng=nothing 이면 temp9 결과와 같음.
# =====================================================================

using SparseArrays
using Statistics

# =====================================================================
# 1. 공통 유틸 / 전처리
# =====================================================================

function running_median(x::AbstractVector, window::Int)
    window = max(window, 1)
    iseven(window) && (window += 1)   # odd num window

    half = window ÷ 2
    out = similar(x, Float64)

    @inbounds for i in eachindex(x)
        lo = max(firstindex(x), i - half)
        hi = min(lastindex(x), i + half)
        out[i] = median(@view x[lo:hi])
    end
    return out
end

# flow/fhigh 밖은 0 으로 지움. harmonic 을 잡으려면 fhigh 를 올릴 것 (예: Nyquist).
# noise_lo~noise_hi 대역의 95% 분위수를 잡음 임계로 사용.
function denoise_usv(sng::Matrix{Float64}, f::AbstractVector{<:Real};
                     flow::Real=15_000, fhigh::Real=110_000,
                     noise_lo::Real=15_000, noise_hi::Real=40_000)
    usv2 = deepcopy(sng)

    tind1 = findlast(x -> x < noise_lo, f)
    tind2 = findfirst(x -> x > noise_hi, f)

    noise_vals = usv2[tind1:tind2, :]
    noise_sorted = sort(vec(noise_vals))
    thresh_val = noise_sorted[Int(round(length(noise_sorted) * 0.95))]

    usv2[findall(x -> x < thresh_val, usv2)] .= 0

    # USV 주파수 대역 밖 에너지 제거
    usv2[findall(x -> x < flow, f), :] .= 0
    usv2[findall(x -> x > fhigh, f), :] .= 0
    return usv2
end

# normalize a 2d array in column; sum of each column is one  (※ in-place)
function normalize_sng!(sng::Matrix{Float64}, pow::BitArray{1})
    for i in 1:size(sng, 2)
        s = 0.0
        for j in 1:size(sng, 1)
            @inbounds s += sng[j, i]
        end
        div, id = !iszero(s) ? (s, false) : (1.0, true)
        for j in 1:size(sng, 1)
            @inbounds sng[j, i] /= div
        end
        @inbounds pow[i] = id # zero column
    end
    nothing
end

# ith offset, jth column sdsng
function dense_sng_diff(sng::Matrix{Float64}, frange::UnitRange{Int64},
                        offset::UnitRange{Int64}, i::Int64, j::Int64)
    sdsng = 0.0
    for n = 1:length(frange)
        sdsng += abs(sng[frange[n], j+1] - sng[frange[n]+offset[i], j])
    end
    return sdsng
end

# ※ 인자를 in-place 로 정규화함. 원본을 보존하려면 copy 를 넘길 것 (usv_preprocess 참고).
function specdiscont(sng::Matrix{Float64})
    zero_ind = falses(size(sng, 2))
    normalize_sng!(sng, zero_ind)

    moff = 3
    offset = -moff:moff
    frange = 1+moff:size(sng, 1)-moff

    sdsng = zeros(length(offset))
    dch = zeros(Float64, size(sng, 2))
    dch[zero_ind] .= 2.0
    dch[end] = 2.0

    zero_ind .= .!zero_ind   # non-zero inds
    zero_ind[end] = 0        # 배열 밖 계산 방지
    zz = sparse(zero_ind)

    @inbounds for j in zz.nzind
        for i in 1:length(offset)
            sdsng[i] = dense_sng_diff(sng, frange, offset, i, j)
        end
        dch[j] = minimum(sdsng)
    end
    return dch
end

function spectralpurity(sng::AbstractMatrix{<:Real})
    specpur = zeros(size(sng, 2))
    for col = 1:size(sng, 2)
        temp = @view sng[:, col]
        specpur[col] = maximum(temp) / sum(temp)
    end
    specpur[isnan.(specpur)] .= 0
    return specpur
end

function meanfrequency(sng::AbstractMatrix{<:Real}, f::AbstractVector{<:Real})
    mf = zeros(size(sng, 2))
    @inbounds for col = 1:size(sng, 2)
        up = 0.0
        down = 0.0
        for row = 1:size(sng, 1)
            up += f[row] * sng[row, col]
            down += sng[row, col]
        end
        mf[col] = up / down
    end
    mf[isnan.(mf)] .= 0
    return mf
end

function usv_preprocess(sng::Matrix{Float64}, dt_sng::Float64, f::AbstractVector{<:Real})
    mf = meanfrequency(sng, f)
    sp = spectralpurity(sng)
    sd = specdiscont(copy(sng))   # ← copy: 호출자의 sng 가 정규화되지 않도록

    w = Int(round(0.005 / dt_sng))
    mf[2:end] .= running_median(mf, w)[2:end]
    sp[2:end] .= running_median(sp, w)[2:end]
    sd[2:end] .= running_median(sd, w)[2:end]

    return mf, sp, sd
end

# band index 범위를 구하는 helper (flow~fhigh Hz -> row bin 범위)
function _band_bins(f, flow::Real, fhigh::Real, nrow::Int)
    lo = findfirst(x -> x >= flow, f)
    hi = findlast(x -> x <= fhigh, f)
    lo === nothing && (lo = 1)
    hi === nothing && (hi = nrow)
    lo, hi = min(lo, hi), max(lo, hi)
    return lo, hi
end

# flow~fhigh 대역의 컬럼별 총 파워
function usv_band_power(sng::AbstractMatrix{<:Real}, f;
                        flow::Float64=15_000.0, fhigh::Float64=110_000.0)
    lo, hi = _band_bins(f, flow, fhigh, size(sng, 1))
    return vec(sum(@view(sng[lo:hi, :]), dims=1))
end

# robust noise floor: nonzero 컬럼 파워의 낮은 분위수(q)
#   (median(nonzero) 는 denoise 된 sparse 데이터에서 신호 레벨이 되어버림)
function _noise_floor(colpow::AbstractVector{<:Real}; q::Float64=0.25)
    nz = colpow[colpow .> 0]
    isempty(nz) && return 0.0
    return quantile(nz, q)
end

# 한 컬럼(벡터)의 nonzero 중앙값 (per-column 잡음 추정용)
function _median_nonzero(col::AbstractVector{<:Real})
    nz = [v_ for v_ in col if v_ > 0]
    isempty(nz) && return 0.0
    return median(nz)
end

# ON/OFF 를 [1, ncol] 로 clamp 하고 on <= off 로 정렬
function _sanitize_onoff(ON::AbstractVector{<:Integer}, OFF::AbstractVector{<:Integer}, ncol::Int)
    length(ON) == length(OFF) || throw(DimensionMismatch("ON and OFF must have equal length"))
    on  = clamp.(Int.(ON),  1, ncol)
    off = clamp.(Int.(OFF), 1, ncol)
    return min.(on, off), max.(on, off)
end

# =====================================================================
# 2. 시간축 detection
# =====================================================================

function usv_ind(mf::Vector{Float64}, sp::Vector{Float64}, sd::Vector{Float64};
                 mfthresh::Real=20_000, spthresh::Real=0.3, sdthresh::Real=0.85)
    spind = findall(x -> x > spthresh, sp)
    mfind = findall(x -> x > mfthresh, mf)
    sdind = findall(x -> x < sdthresh, sd)
    return intersect(mfind, spind, sdind)
end

# ※ 원본 MATLAB 포팅 규약 유지: ON/OFF 가 실제 전이 지점보다 1~2 bin 앞섬, ON=0 가능.
#   하위 함수들은 _sanitize_onoff / clamp 로 처리함.
function putative_usv(uind::AbstractVector{<:Integer}, t::AbstractVector)
    usv_putative = zeros(length(t))
    usv_putative[uind] .= 1
    usv_on_off = diff(usv_putative)
    usvON = findall(x -> x == 1, usv_on_off) .- 1
    usv_putative[1] == 1 && prepend!(usvON, [1])
    usvOFF = findall(x -> x == -1, usv_on_off) .- 1
    usv_putative[end] == 1 && append!(usvOFF, [length(t)])
    return usvON, usvOFF
end

function usv_duration_check(usvON::Vector{Int64}, usvOFF::Vector{Int64}, dt_sng)
    minlen = Int(round(0.005 / dt_sng))
    keepind = [z for z in 1:length(usvON) if usvOFF[z] > usvON[z] + minlen]
    return usvON[keepind], usvOFF[keepind]
end

function usv_merge(usvON::Vector{Int64}, usvOFF::Vector{Int64}, dt_sng)
    gap = Int(round(0.03 / dt_sng))
    mergeind = [p for p in 1:length(usvOFF)-1 if usvON[p+1] - usvOFF[p] < gap]
    deleteat!(usvON, mergeind .+ 1)
    deleteat!(usvOFF, mergeind)
    return usvON, usvOFF
end

function false_positive_usv(usvON::Vector{Int64}, usvOFF::Vector{Int64}, usv_putative, dt_sng)
    gapThreshold = Int(round(0.25 / dt_sng))
    if length(usvON) > 1
        indicesToThrowOut = Int64[]
        for z = 1:length(usvON)
            distanceBefore = z > 1 ? usvON[z] - usvOFF[z-1] : usvON[z]
            distanceAfter  = z < length(usvON) ? usvON[z+1] - usvOFF[z] :
                                                 length(usv_putative) - usvOFF[z]
            if distanceAfter > gapThreshold && distanceBefore > gapThreshold
                push!(indicesToThrowOut, z)
            end
        end
        deleteat!(usvON, indicesToThrowOut)
        deleteat!(usvOFF, indicesToThrowOut)
    end
    return usvON, usvOFF
end

function usv_processing(uind::AbstractVector{<:Integer}, t::AbstractVector, dt_sng::Float64)
    on, off = putative_usv(uind, t)
    on, off = usv_duration_check(on, off, dt_sng)
    on, off = usv_merge(on, off, dt_sng)
    on, off = false_positive_usv(on, off, t, dt_sng)
    return on, off
end

# 원본 detection (비교용으로 유지)
function usv_whole(sng::Matrix{Float64}, dt_sng::Float64,
                   t_sng::AbstractVector, f::AbstractVector{<:Real})
    mf2, sp2, sd2 = usv_preprocess(sng, dt_sng, f)
    ui = usv_ind(mf2, sp2, sd2)
    return usv_processing(ui, t_sng, dt_sng)
end

# (A) amplitude / SNR gate : 컬럼 파워가 noise floor 의 snr_k배 초과인 컬럼만 true.
#   ※ 조용한 onset/offset 을 잘라 syllable 을 truncate 할 수 있어 usv_whole_2d 에서 기본 off.
function usv_amplitude_gate(sng::AbstractMatrix{<:Real}, f;
                            flow::Float64=15_000.0, fhigh::Float64=110_000.0,
                            snr_k::Float64=3.0, floor_q::Float64=0.25)
    p = usv_band_power(sng, f; flow=flow, fhigh=fhigh)
    floor_val = _noise_floor(p; q=floor_q)
    return p .> (snr_k * floor_val)
end

# (A) confident false-positive:
#   고립 구간을 그 자체가 충분히 강할 때만 유지. 전부 지워질 상황이면 아무것도 안 지움.
function false_positive_usv_confident(usvON::Vector{Int64}, usvOFF::Vector{Int64},
                                      n::Int, dt_sng::Float64,
                                      sng::AbstractMatrix{<:Real}, f;
                                      gapThreshold_s::Float64=0.25,
                                      keep_snr::Float64=1.5, keep_pur::Float64=0.25,
                                      floor_q::Float64=0.25,
                                      flow::Float64=15_000.0, fhigh::Float64=110_000.0)
    length(usvON) <= 1 && return usvON, usvOFF
    gap = Int(round(gapThreshold_s / dt_sng))

    p = usv_band_power(sng, f; flow=flow, fhigh=fhigh)
    floor_val = _noise_floor(p; q=floor_q)

    lo, hi = _band_bins(f, flow, fhigh, size(sng, 1))
    ncol = size(sng, 2)

    throwout = Int64[]
    for z in 1:length(usvON)
        db = z > 1 ? usvON[z] - usvOFF[z-1] : usvON[z]
        da = z < length(usvON) ? usvON[z+1] - usvOFF[z] : n - usvOFF[z]
        ((da > gap) && (db > gap)) || continue   # 고립된 구간만 판정

        a = clamp(usvON[z], 1, ncol)
        b = clamp(usvOFF[z], 1, ncol)
        a, b = min(a, b), max(a, b)
        seg  = @view sng[lo:hi, a:b]
        colp = vec(sum(seg, dims=1))
        amp  = any(colp .> 0) ? median(colp[colp .> 0]) : 0.0
        pur  = mean(vec(maximum(seg, dims=1)) ./ max.(colp, eps()))

        strong = (pur >= keep_pur) && (amp >= keep_snr * floor_val)
        strong || push!(throwout, z)
    end

    length(throwout) == length(usvON) && return usvON, usvOFF   # fail-safe

    keep = setdiff(1:length(usvON), throwout)
    return usvON[keep], usvOFF[keep]
end

# (A) 히스테리시스 경계 확장 (dual-threshold):
#   core 구간을 신호가 noise floor 근처로 떨어질 때까지 좌/우로 확장.
#   core 에서만 자라므로 순수 잡음 구간은 생성되지 않음.
function extend_boundaries!(ON::Vector{Int64}, OFF::Vector{Int64},
                            sp::Vector{Float64}, colpow::Vector{Float64},
                            floor_val::Float64;
                            snr_ext::Float64=1.2, sp_lo::Float64=0.15)
    ncol = length(colpow)
    n = length(ON)
    thr = snr_ext * floor_val
    @inbounds for k in 1:n
        # 시작점을 유효 범위로 (putative_usv 는 ON=0 을 낼 수 있음)
        ON[k]  = clamp(ON[k],  1, ncol)
        OFF[k] = clamp(OFF[k], 1, ncol)

        left_bound = k > 1 ? OFF[k-1] : 0
        i = ON[k]
        while i > 1 && (i - 1) > left_bound && colpow[i-1] > thr && sp[i-1] > sp_lo
            i -= 1
        end
        ON[k] = i

        right_bound = k < n ? ON[k+1] : ncol + 1
        j = OFF[k]
        while j < ncol && (j + 1) < right_bound && colpow[j+1] > thr && sp[j+1] > sp_lo
            j += 1
        end
        OFF[k] = j
    end
    return ON, OFF
end

# 개선된 전체 detection (시간축). 세로 경계는 usv_segment_freq_bounds 로 붙임.
#   snr_k   : amp-gate core seed 용 (엄격)
#   snr_ext : 경계 확장용 (느슨)
function usv_whole_2d(sng::Matrix{Float64}, dt_sng::Float64,
                      t_sng::AbstractVector, f::AbstractVector{<:Real};
                      use_amp_gate::Bool=false,
                      snr_k::Float64=0.8,
                      fp_mode::Symbol=:none,      # :confident | :original | :none
                      extend::Bool=true,
                      snr_ext::Float64=1.3,
                      sp_lo::Float64=0.15,
                      floor_q::Float64=0.15,
                      mfthresh::Float64=20_000.0,
                      spthresh::Float64=0.1,
                      sdthresh::Float64=0.99,
                      verbose::Bool=true)
    fp_mode in (:confident, :original, :none) ||
        error("fp_mode must be :confident, :original, or :none (got $fp_mode)")

    mf2, sp2, sd2 = usv_preprocess(sng, dt_sng, f)
    uind = usv_ind(mf2, sp2, sd2; mfthresh=mfthresh, spthresh=spthresh, sdthresh=sdthresh)
    verbose && println("  [stage] core columns (sp&mf&sd): ", length(uind))

    if use_amp_gate
        gate = usv_amplitude_gate(sng, f; snr_k=snr_k, floor_q=floor_q)
        uind = filter(i -> gate[i], uind)
        verbose && println("  [stage] after amp-gate:        ", length(uind))
    end

    on, off = putative_usv(uind, t_sng)
    on, off = usv_duration_check(on, off, dt_sng)
    on, off = usv_merge(on, off, dt_sng)
    verbose && println("  [stage] segments after merge:  ", length(on))

    if fp_mode === :confident
        on, off = false_positive_usv_confident(on, off, length(t_sng), dt_sng, sng, f;
                                               keep_snr=1.5, keep_pur=0.25, floor_q=floor_q)
    elseif fp_mode === :original
        on, off = false_positive_usv(on, off, t_sng, dt_sng)
    end
    verbose && println("  [stage] segments after FP ($(fp_mode)): ", length(on))

    if extend && !isempty(on)
        colpow = usv_band_power(sng, f)
        floor_val = _noise_floor(colpow; q=floor_q)
        extend_boundaries!(on, off, sp2, colpow, floor_val; snr_ext=snr_ext, sp_lo=sp_lo)
        on, off = usv_merge(on, off, dt_sng)
    end
    verbose && println("  [stage] final segments:        ", length(on))

    return on, off
end

# =====================================================================
# 3. 주파수(세로) 경계 탐지
# =====================================================================

# --- 3-1. :ridge 방식 ------------------------------------------------
# 한 syllable(on:off)의 주파수 경계(Hz). 각 컬럼 ridge(argmax)를 모아 분위수로 대역 결정.
#   select = :local  (기본) 컬럼 peak 가 그 컬럼 자체 잡음의 local_k 배 초과면 포함
#            :global 세그먼트 전체에서 상위 (1-col_energy_q) 컬럼만 포함
#   select=:local (기본)은 usv_harmonic_analysis 로 계산 (n_peaks=1 이면 harmonic 없이 단일 ridge).
#     → 단일 ridge 가 flow~fhigh 전체로 잡히던 문제의 수정(잡음 추정, col_rel, 연속성, fallback)이
#       harmonic 모드와 똑같이 적용됨. analysis_kw 로 col_rel, max_jump_hz 등을 넘길 수 있음.
#   n_peaks > 1 이면 harmonic 탐지 모드 (섹션 3-4).
#   select=:global 은 옛 구현 그대로 (비교용).
function usv_freq_bounds(sng::AbstractMatrix{<:Real}, f, on::Integer, off::Integer;
                         flow::Float64=15_000.0, fhigh::Float64=110_000.0,
                         select::Symbol=:local, local_k::Float64=1.5,
                         col_energy_q::Float64=0.60,
                         ridge_q_lo::Float64=0.02, ridge_q_hi::Float64=0.99,
                         margin_bins::Int=4, min_height_bins::Int=6,
                         n_peaks::Int=1, analysis_kw...)
    select in (:local, :global) || throw(ArgumentError("select must be :local or :global"))
    if select === :local || n_peaks > 1
        a = usv_harmonic_analysis(sng, f, on, off; flow=flow, fhigh=fhigh,
                                  n_peaks=n_peaks, local_k=local_k,
                                  ridge_q_lo=ridge_q_lo, ridge_q_hi=ridge_q_hi,
                                  margin_bins=margin_bins, min_height_bins=min_height_bins,
                                  analysis_kw...)
        return a.f_lo, a.f_hi
    end
    isempty(analysis_kw) ||
        throw(ArgumentError("options $(keys(analysis_kw)) are not used with select=:global"))

    ncol = size(sng, 2)
    on  = clamp(Int(on), 1, ncol)
    off = clamp(Int(off), 1, ncol)
    on, off = min(on, off), max(on, off)

    lo, hi = _band_bins(f, flow, fhigh, size(sng, 1))

    seg    = @view sng[lo:hi, on:off]
    coltot = vec(sum(seg, dims=1))
    pos    = coltot[coltot .> 0]
    isempty(pos) && return flow, fhigh

    gthr = quantile(pos, col_energy_q)   # select==:global 용

    peaks = Int[]
    @inbounds for c in 1:size(seg, 2)
        col = @view seg[:, c]
        pk  = maximum(col)
        pk <= 0 && continue
        coltot[c] >= gthr || continue                # (:global 전용 경로)
        push!(peaks, argmax(col) + lo - 1)   # 전체 스펙트럼 bin 으로 환산
    end
    isempty(peaks) && return flow, fhigh

    b_lo = floor(Int, quantile(peaks, ridge_q_lo)) - margin_bins
    b_hi = ceil(Int, quantile(peaks, ridge_q_hi)) + margin_bins
    b_lo, b_hi = _ensure_min_height(b_lo, b_hi, min_height_bins, lo, hi)

    return Float64(f[b_lo]), Float64(f[b_hi])
end

# 최소 높이 보장 후 band 로 clamp (두 방식 공통)
function _ensure_min_height(b_lo::Int, b_hi::Int, min_height_bins::Int, lo::Int, hi::Int)
    if b_hi - b_lo < min_height_bins
        c = (b_hi + b_lo) ÷ 2
        b_lo = c - min_height_bins ÷ 2
        b_hi = c + min_height_bins ÷ 2
    end
    return clamp(b_lo, lo, hi), clamp(b_hi, lo, hi)
end

# --- 3-2. :threshold 방식 (구 TimeBinDetector) ------------------------
# 각 time bin 마다 threshold = max(absolute_threshold,
#                                  median + noise_k·1.4826·MAD,
#                                  peak_fraction·peak)
# 을 넘는 최저/최고 bin 을 찾고, 세그먼트 전체의 min/max 를 경계로 사용.
# 반환: 세그먼트별 NamedTuple
#   (on, off, time_bins, bin_min, bin_max, freq_min, freq_max,   ← time bin 별 contour
#    min_bin, max_bin, min_freq, max_freq, valid)                ← 세그먼트 전체 경계
function detect_frequency_bounds(
    sng::AbstractMatrix{<:Real}, f::AbstractVector{<:Real},
    ON::AbstractVector{<:Integer}, OFF::AbstractVector{<:Integer};
    flow::Real=15_000, fhigh::Real=110_000, noise_k::Real=3.0,
    peak_fraction::Real=0.05, absolute_threshold::Real=0.0,
    margin_bins::Integer=0,
)
    Base.require_one_based_indexing(sng, f, ON, OFF)
    length(f) == size(sng, 1) || throw(DimensionMismatch("f must match spectrogram rows"))
    length(ON) == length(OFF) || throw(DimensionMismatch("ON and OFF must have equal length"))
    isempty(f) && throw(ArgumentError("frequency axis must not be empty"))
    all(isfinite, f) && all(diff(f) .> 0) ||
        throw(ArgumentError("frequencies must be finite and strictly increasing"))
    isfinite(flow) && isfinite(fhigh) && 0 <= flow <= fhigh ||
        throw(ArgumentError("require 0 <= flow <= fhigh, both finite"))
    isfinite(noise_k) && noise_k >= 0 || throw(ArgumentError("noise_k must be finite and nonnegative"))
    isfinite(peak_fraction) && 0 <= peak_fraction < 1 ||
        throw(ArgumentError("peak_fraction must be in [0, 1)"))
    isfinite(absolute_threshold) && absolute_threshold >= 0 ||
        throw(ArgumentError("absolute_threshold must be finite and nonnegative"))
    margin_bins >= 0 || throw(ArgumentError("margin_bins must be nonnegative"))
    lo, hi = searchsortedfirst(f, flow), searchsortedlast(f, fhigh)
    lo <= hi || throw(ArgumentError("no frequency bins in the requested band"))
    for (on, off) in zip(ON, OFF)
        1 <= on <= off <= size(sng, 2) ||
            throw(ArgumentError("require 1 <= ON <= OFF <= number of time bins"))
    end

    results = NamedTuple[]
    for (on, off) in zip(ON, OFF)
        time_bins = Int(on):Int(off)
        bin_min = Vector{Union{Missing,Int}}(missing, length(time_bins))
        bin_max = copy(bin_min)
        freq_min = Vector{Union{Missing,Float64}}(missing, length(time_bins))
        freq_max = copy(freq_min)
        overall_lo, overall_hi = hi + 1, lo - 1
        for (j, t) in enumerate(time_bins)
            col = @view sng[lo:hi, t]
            values = Float64[v for v in col if isfinite(v)]
            isempty(values) && continue
            minimum(values) >= 0 || throw(ArgumentError("sng must contain nonnegative linear values"))
            peak = maximum(values)
            peak > 0 || continue
            baseline = median(values)
            mad = median(abs.(values .- baseline))
            # noise_k=0 explicitly disables the noise-floor criterion.
            noise_threshold = iszero(noise_k) ? 0.0 : baseline + noise_k * 1.4826 * mad
            threshold = max(absolute_threshold, noise_threshold, peak_fraction * peak)
            first_bin, last_bin = hi + 1, lo - 1
            for r in lo:hi
                v = sng[r, t]
                if isfinite(v) && v > threshold
                    first_bin = min(first_bin, r)
                    last_bin = max(last_bin, r)
                end
            end
            first_bin <= last_bin || continue
            bin_min[j], bin_max[j] = first_bin, last_bin
            freq_min[j], freq_max[j] = Float64(f[first_bin]), Float64(f[last_bin])
            overall_lo = min(overall_lo, first_bin)
            overall_hi = max(overall_hi, last_bin)
        end
        valid = overall_lo <= overall_hi
        min_bin = valid ? overall_lo - min(margin_bins, overall_lo - lo) : missing
        max_bin = valid ? overall_hi + min(margin_bins, hi - overall_hi) : missing
        push!(results, (; on=Int(on), off=Int(off), time_bins, bin_min, bin_max,
                          freq_min, freq_max, min_bin, max_bin,
                          min_freq=valid ? Float64(f[min_bin]) : missing,
                          max_freq=valid ? Float64(f[max_bin]) : missing, valid))
    end
    return results
end

# detect_frequency_bounds 를 detection 출력(ON/OFF)에 바로 쓰는 adapter.
#   - ON/OFF 를 sanitize 해서 넘김
#   - invalid 세그먼트는 (flow, fhigh) 로 fallback (ridge 방식과 동일 규약)
#   - min_height_bins 보장 (ridge 방식과 동일)
#   - min_col_peak_frac: 세그먼트 최대 peak 대비 이 비율 미만인 time bin 은 경계 계산에서 제외.
#       detect_frequency_bounds 는 모든 time bin 의 min/max 를 합치므로, 경계 확장으로
#       섞여 들어온 약한 잡음 컬럼 하나가 box 를 전체 대역으로 벌릴 수 있음 → 이를 방지.
#       0.0 이면 원래 detect_frequency_bounds 결과와 동일.
# 반환: Vector{Tuple{Float64,Float64}}  (f_lo, f_hi) per segment
function usv_freq_bounds_threshold(sng::AbstractMatrix{<:Real}, f::AbstractVector{<:Real},
                                   ON::AbstractVector{<:Integer}, OFF::AbstractVector{<:Integer};
                                   flow::Float64=15_000.0, fhigh::Float64=110_000.0,
                                   margin_bins::Int=4, min_height_bins::Int=6,
                                   noise_k::Real=3.0, peak_fraction::Real=0.05,
                                   absolute_threshold::Real=0.0,
                                   min_col_peak_frac::Real=0.2)
    isempty(ON) && return Tuple{Float64,Float64}[]
    on, off = _sanitize_onoff(ON, OFF, size(sng, 2))
    res = detect_frequency_bounds(sng, f, on, off;
                                  flow=flow, fhigh=fhigh, noise_k=noise_k,
                                  peak_fraction=peak_fraction,
                                  absolute_threshold=absolute_threshold,
                                  margin_bins=0)          # margin 은 아래에서 직접 적용
    lo, hi = searchsortedfirst(f, flow), searchsortedlast(f, fhigh)
    out = Vector{Tuple{Float64,Float64}}(undef, length(res))
    for (k, r) in enumerate(res)
        if !r.valid
            out[k] = (flow, fhigh)
            continue
        end
        # time bin 별 band 내 peak
        colpeak = [maximum(x -> isfinite(x) ? Float64(x) : 0.0, @view sng[lo:hi, t])
                   for t in r.time_bins]
        cut = min_col_peak_frac * maximum(colpeak)
        b_lo, b_hi = hi + 1, lo - 1
        for j in eachindex(r.time_bins)
            (ismissing(r.bin_min[j]) || colpeak[j] < cut) && continue
            b_lo = min(b_lo, r.bin_min[j])
            b_hi = max(b_hi, r.bin_max[j])
        end
        if b_lo > b_hi                                    # 강한 컬럼이 임계를 못 넘은 경우 fallback
            b_lo, b_hi = r.min_bin, r.max_bin
        end
        b_lo, b_hi = _ensure_min_height(b_lo - margin_bins, b_hi + margin_bins,
                                        min_height_bins, lo, hi)
        out[k] = (Float64(f[b_lo]), Float64(f[b_hi]))
    end
    return out
end

# --- 3-3. 공통 인터페이스 ---------------------------------------------
# fmethod = :ridge | :threshold. kwargs 는 선택한 방식의 함수로 그대로 전달.
# 반환: Vector{Tuple{Float64,Float64}}  (f_lo, f_hi) per segment
function usv_segment_freq_bounds(sng::AbstractMatrix{<:Real}, f::AbstractVector{<:Real},
                                 ON::AbstractVector{<:Integer}, OFF::AbstractVector{<:Integer};
                                 fmethod::Symbol=:ridge,
                                 flow::Float64=15_000.0, fhigh::Float64=110_000.0,
                                 kwargs...)
    if fmethod === :ridge
        on, off = _sanitize_onoff(ON, OFF, size(sng, 2))
        return Tuple{Float64,Float64}[usv_freq_bounds(sng, f, a, b; flow=flow, fhigh=fhigh, kwargs...)
                                      for (a, b) in zip(on, off)]
    elseif fmethod === :threshold
        return usv_freq_bounds_threshold(sng, f, ON, OFF; flow=flow, fhigh=fhigh, kwargs...)
    else
        error("fmethod must be :ridge or :threshold (got $fmethod)")
    end
end

# --- 3-4. Harmonic 탐지 (multi-peak ridge) ----------------------------
# 아이디어
#   1) 컬럼마다 local maximum 을 진폭 순으로 최대 n_peaks 개 고름
#      (가장 강한 peak 의 peak_rel 배 이상, min_sep_hz 이상 떨어진 것만 → 같은 ridge 중복 방지)
#   2) 그 중 "가장 강한 peak 를 포함하면서 정수배(2..max_harmonic, 허용오차 harm_tol)로
#      설명되는 가장 큰 집합"만 남김 → 동시에 겹친 다른 개체의 콜/잡음 blob 은 harmonic 으로 취급 안 함
#   3) harmonic 이 있는 컬럼 비율이 min_harm_frac 이상일 때만 세그먼트를 harmonic 으로 판정
#   4) fundamental 이 약해서 harmonic 만 보이는 컬럼은, harmonic 세그먼트에서
#      중앙 fundamental 의 정수배이면 harmonic 으로 재분류
#   5) (선택) fs 를 주면 Nyquist(fs/2) 위의 harmonic 이 접혀(alias) 들어온 위치
#      |fs·m − k·f0| 도 harmonic 으로 인정. anti-alias 필터가 없는 DAQ 에서만 의미 있음.
#
# 더 많은 / 더 약한 harmonic 을 잡으려면
#   - denoise 된 sng 대신 snr_spectrogram(raw) 에서 분석 (denoise 의 95% 임계가 약한 harmonic 을 지움)
#   - HARM_KW_MORE 설정 사용 (n_peaks↑, peak_rel↓, min_harm_frac↓, max_harmonic↑)

# 많은 / 약한 harmonic 용 권장 설정. snr_spectrogram 입력과 함께 쓸 것.
#   peak_rel=0.01 은 fout 이 power 일 때 −20 dB (magnitude 라면 0.1 이 −20 dB).
const HARM_KW_MORE = (; n_peaks=6, peak_rel=0.01, local_k=3.0,
                        max_harmonic=6, min_harm_frac=0.1, min_sep_hz=8_000.0)

# 행(주파수)별 잡음 바닥으로 나눈 SNR spectrogram.
#   마이크/DAQ 감도가 고주파에서 떨어져도 그 대역의 잡음도 같이 떨어지므로,
#   Nyquist 근처의 약한 harmonic 이 상대적으로 드러남. 원본(denoise 전) fout 을 넣을 것.
#   q: 행별 잡음 추정 분위수 (USV 는 시간상 sparse 하므로 median 이면 충분)
function snr_spectrogram(sng::AbstractMatrix{<:Real}; q::Float64=0.5)
    nfl = row_noise_floor(sng; q=q)
    out = Matrix{Float64}(undef, size(sng))
    @inbounds for c in axes(sng, 2), r in axes(sng, 1)
        v = sng[r, c]
        out[r, c] = isfinite(v) ? max(Float64(v), 0.0) / nfl[r] : 0.0
    end
    return out
end

# 행(주파수)별 잡음 바닥 벡터 (snr_spectrogram 의 분모). 0 이하이면 eps().
#   SNR 행렬 전체를 만들지 않고 컬럼 단위로 나눠 쓰고 싶을 때 사용
#   (usv_harmonic_analysis 의 measure_sng / measure_floor).
function row_noise_floor(sng::AbstractMatrix{<:Real}; q::Float64=0.5)
    nfl = Vector{Float64}(undef, size(sng, 1))
    for r in axes(sng, 1)
        vals = Float64[v for v in @view(sng[r, :]) if isfinite(v)]
        x = isempty(vals) ? 0.0 : quantile(vals, q)
        nfl[r] = x > 0 ? x : eps()
    end
    return nfl
end

# --- 주파수 경계 정밀화 helper (2026-11) -------------------------------

# 실수 bin 위치 → Hz (f 는 등간격이 아니어도 됨: 인접 bin 사이 선형 보간)
function _bin_to_hz(f::AbstractVector{<:Real}, x::Real)
    n = length(f)
    n == 1 && return Float64(f[1])
    i = clamp(floor(Int, x), 1, n - 1)
    return Float64(f[i]) + (x - i) * Float64(f[i+1] - f[i])
end

# 포물선 보간 peak 위치 (log 값에서). v(r) 는 bin r 의 선형 파워. 반환: 실수 bin.
#   Hann 창 + log 파워에서 bin 양자화 오차(±0.5 bin)를 ~0.1 bin 이하로 줄임.
function _parabolic_peak(v, b::Int, lo::Int, hi::Int)
    (b <= lo || b >= hi) && return float(b)
    a, c, d = log(max(v(b - 1), floatmin())), log(max(v(b), floatmin())), log(max(v(b + 1), floatmin()))
    den = a - 2c + d
    den < 0 || return float(b)                    # 볼록하지 않으면(평평/이상) 보간 안 함
    return b + clamp(0.5 * (a - d) / den, -0.5, 0.5)
end

# 임계 thr 를 안쪽 값 vin(≥thr) 과 바깥 값 vout(<thr) 사이에서 지나는 위치 (0..1, 안쪽 기준).
#   dB 선형 보간. 0 = 안쪽 bin 위, 1 = 바깥 bin 위.
function _cross_frac(vin::Real, vout::Real, thr::Real)
    vin > thr || return 0.0
    num = log(vin / thr)
    den = log(vin / max(vout, floatmin()))
    return den > 0 ? clamp(num / den, 0.0, 1.0) : 0.0
end

# 최소 높이 보장 + band clamp (실수 bin 버전)
function _ensure_min_height_f(a::Real, c::Real, min_h::Real, lo::Int, hi::Int)
    if c - a < min_h
        m = (a + c) / 2
        a, c = m - min_h / 2, m + min_h / 2
    end
    return clamp(Float64(a), lo, hi), clamp(Float64(c), lo, hi)
end

# 컬럼별 edge 목록에서 box 경계 (bound_mode=:continuous).
#   cols : 각 edge 가 속한 컬럼 번호 (시간 순서, 같은 컬럼 여러 개 가능 = harmonic)
#   e    : (아래 edge, 위 edge) 실수 bin
#   한 컬럼의 극단값은 시간상 이웃 컬럼(앞/뒤 중 하나)의 같은 쪽 edge 와 jump bin 이내일 때만
#   인정 → 잡음으로 한 컬럼만 튄 edge 는 버리고, sweep 끝처럼 이웃과 이어진 극단은 유지.
#   (예전 quantile(…, 0.99) 는 컬럼 < 100 개면 사실상 최댓값이라 튄 컬럼 하나가 box 를 결정)
function _continuous_extent(cols::AbstractVector{<:Integer},
                            e::AbstractVector{<:Tuple{<:Real,<:Real}}, jump::Real)
    u = unique(cols)
    elo = [minimum(first(e[k]) for k in eachindex(cols) if cols[k] == c) for c in u]
    ehi = [maximum(last(e[k])  for k in eachindex(cols) if cols[k] == c) for c in u]
    n = length(u)
    n <= 2 && return minimum(elo), maximum(ehi)
    supported(x, i) = (i > 1 && abs(x[i] - x[i-1]) <= jump) || (i < n && abs(x[i] - x[i+1]) <= jump)
    okl = [supported(elo, i) for i in 1:n]
    okh = [supported(ehi, i) for i in 1:n]
    a = any(okl) ? minimum(elo[okl]) : median(elo)
    c = any(okh) ? maximum(ehi[okh]) : median(ehi)
    return a, c
end

# 한 컬럼의 peak 후보 (band-local index, 진폭 내림차순)
function _column_peaks(col::AbstractVector{<:Real}, max_peaks::Int, min_rel::Real,
                       min_sep::Int, noise_floor::Real)
    n = length(col)
    cand = Int[]
    @inbounds for r in 1:n
        v = col[r]
        (v > 0 && v > noise_floor) || continue
        left  = r == 1 ? -Inf : col[r-1]
        right = r == n ? -Inf : col[r+1]
        (v >= left && v > right) && push!(cand, r)     # plateau 는 한 점만
    end
    isempty(cand) && return cand
    sort!(cand; by=r -> col[r], rev=true)
    top = col[cand[1]]
    picked = Int[]
    for r in cand
        col[r] < min_rel * top && break
        all(p -> abs(p - r) >= min_sep, picked) || continue   # non-max suppression
        push!(picked, r)
        length(picked) >= max_peaks && break
    end
    return picked
end

# fp 가 f0 의 몇 번째 harmonic 인지. 반환 (k, aliased). 아니면 (0, false).
#   fs === nothing : 직접 정수배만
#   fs 지정        : Nyquist 위 harmonic 이 접힌 위치도 검사
#   alias_tol: 접힌 위치 허용오차 (f0 대비 비율). 접힌 harmonic 은 후보 차수가 많아
#              우연히 맞을 확률이 높으므로 직접 harmonic(tol)보다 엄격하게 둠.
function _harmonic_order(fp::Real, f0::Real; tol::Real=0.06, max_harmonic::Int=4,
                         fs::Union{Nothing,Real}=nothing, alias_tol::Real=0.03)
    f0 > 0 || return (0, false)
    nyq = fs === nothing ? Inf : fs / 2
    r = fp / f0
    k = round(Int, r)
    if 2 <= k <= max_harmonic && abs(r / k - 1) <= tol && k * f0 <= nyq * (1 + tol)
        return (k, false)
    end
    fs === nothing && return (0, false)
    for k in 2:max_harmonic
        fk = k * f0
        fk > nyq || continue
        fa = mod(fk, fs)
        fa > nyq && (fa = fs - fa)                     # 접힌 위치
        abs(fp - fa) <= alias_tol * f0 && return (k, true)
    end
    return (0, false)
end

_is_harmonic_of(fp::Real, f0::Real; kw...) = first(_harmonic_order(fp, f0; kw...)) > 0

# peaks (전체 bin, 진폭 내림차순) → (fund, harms, orders, aliased)
#   가장 강한 peak 를 포함하면서 가장 많은 peak 를 설명하는 fundamental 을 선택.
#   aliased harmonic 은 fundamental 보다 낮은 주파수에 있을 수 있으므로 fund 를 따로 반환.
function _harmonic_set(peaks::Vector{Int}, f; tol::Real, max_harmonic::Int,
                       fs::Union{Nothing,Real}=nothing, alias_tol::Real=0.03)
    strongest = peaks[1]
    best = (; fund=strongest, harms=Int[], orders=Int[], aliased=Bool[])
    length(peaks) == 1 && return best
    for c in sort(peaks; by=p -> f[p])            # 낮은 주파수부터 fundamental 후보
        harms = Int[]; orders = Int[]; aliased = Bool[]
        for p in peaks
            p == c && continue
            k, al = _harmonic_order(f[p], f[c]; tol=tol, max_harmonic=max_harmonic,
                                    fs=fs, alias_tol=alias_tol)
            k > 0 || continue
            k in orders && continue               # 같은 차수 중복 방지 (먼저 = 더 강한 peak 유지)
            push!(harms, p); push!(orders, k); push!(aliased, al)
        end
        (c == strongest || strongest in harms) || continue
        length(harms) > length(best.harms) && (best = (; fund=c, harms, orders, aliased))
    end
    return best
end

# 한 세그먼트의 harmonic 분석.
# 반환 NamedTuple:
#   f_lo, f_hi      : box 경계 (Hz). harmonic_box=:all 이면 harmonic 포함, :fundamental 이면 기본음만
#   is_harmonic     : harmonic 세그먼트 판정
#   harm_frac       : harmonic 이 보인 컬럼 비율 (활성 컬럼 기준)
#   n_components    : 한 컬럼에서 동시에 보인 최대 성분 수 (기본음 포함)
#   orders          : 2개 이상 컬럼에서 확인된 harmonic 차수 (예: [2, 3])
#   max_order       : orders 의 최댓값 (없으면 1)
#   n_aliased       : 접힌(aliased) harmonic 이 보인 컬럼 수 (fs 를 준 경우만 > 0 가능)
#   ratio           : 직접(비-alias) harmonic / fundamental 주파수 비의 중앙값
#   fund_lo, fund_hi: fundamental 대역 (Hz)
#   harm_lo, harm_hi: harmonic 대역 (Hz, 스펙트로그램 상 위치 — alias 면 접힌 위치), 없으면 missing
#   valid           : false 면 세그먼트에서 신호를 못 찾음 (이때만 box = flow~fhigh)
#   n_cols          : 경계 계산에 실제로 쓰인 컬럼 수
#   fallback        : true 면 ridge 대신 세그먼트 에너지 profile 로 경계를 잡음
#
# 단일 ridge 가 설정 대역 전체로 잡히던 문제 대응 (2026-10)
#   - 컬럼 잡음 추정: _median_nonzero → median(col) (0 포함).
#       denoise 된 sparse 컬럼은 nonzero 가 거의 ridge 자체라 "잡음" median 이 peak 에 가까움
#       → 단일 ridge 컬럼이 모두 탈락 → 빈 결과 → flow~fhigh 로 떨어졌음.
#       (harmonic 콜은 약한 bin 이 많아 median 이 낮아져서 통과했기 때문에 잘 잡혔던 것)
#   - col_rel   : 세그먼트의 강한 컬럼(peak 90% 분위) 대비 col_rel 미만인 컬럼은 제외.
#       경계 확장으로 들어온 조용한 컬럼의 무작위 argmax 가 분위수를 벌리던 문제.
#   - 연속성    : fundamental 궤적의 running median(jump_win 컬럼)에서 max_jump_hz 이상 튄
#       컬럼 제외. 실제 frequency jump 는 여러 컬럼 지속되므로 median 이 따라가서 유지됨.
#   - 쓸 컬럼이 없으면 flow~fhigh 대신 세그먼트 에너지 profile 로 경계 (fallback=true).
#   - (짧은 콜) 에너지 mask: 컬럼의 가장 강한 peak 가 세그먼트 시간합 profile 의 유의 대역
#       밖이면 그 컬럼을 버림. 짧은 콜은 앞뒤 무음/잡음 컬럼 비율이 커서, 잡음 컬럼의
#       무작위 argmax 가 분위수 경계를 대역 끝까지 벌리던 문제 (col_rel/연속성으로는 못 막음:
#       SNR 입력에서 잡음 최대값이 median 의 ~8배라 local_k=3 을 통과하고, 잡음 컬럼이
#       여러 개 연달아 있으면 running median 도 잡음을 따라감).
#
# 강한 ridge 의 위/아래가 잘리던 문제 대응 (2026-10, 2차)
#   - box 를 ridge 의 "peak 위치"가 아니라 "ridge 의 실제 폭(edge)"으로 계산:
#       각 컬럼에서 peak 로부터 위/아래로 값이 max(edge_rel·peak, local_k·잡음) 이상인 동안 확장
#       (최대 max_edge_hz). 강한 ridge 는 leakage/FM 으로 peak 주변까지 에너지가 넓게 퍼지는데
#       예전엔 peak 위치 ± margin(4 bin) 만 써서 보이는 ridge 보다 box 가 좁았음.
#   - 약한 꼬리 컬럼 hysteresis (track_hz, track_gap):
#       강한 컬럼(seed: mask 안 & col_rel 이상)에서 시간상 이웃하고 주파수가 track_hz 이내로
#       이어지는 약한 컬럼은 살림. 예전 col_rel/mask 는 "가장 강한 컬럼 대비" 상대 기준이라
#       강한 콜일수록 FM 의 약한 시작/끝(= 주파수 극단)을 더 많이 잘랐음.
#   - 연속성 필터를 running median → "양쪽 이웃 모두와 max_jump_hz 이상 차이 나는 고립점" 제거로.
#       running median 은 빠른 FM sweep 끝을 따라가지 못해 sweep 의 끝 주파수를 잘랐음.
function usv_harmonic_analysis(sng::AbstractMatrix{<:Real}, f, on::Integer, off::Integer;
                               flow::Float64=15_000.0, fhigh::Float64=125_000.0,
                               n_peaks::Int=5, local_k::Float64=1.5,
                               peak_rel::Float64=0.1, min_sep_hz::Float64=3_000.0,
                               harm_tol::Float64=0.05, max_harmonic::Int=10,
                               min_harm_frac::Float64=0.05,
                               fs::Union{Nothing,Real}=nothing, alias_tol::Float64=0.03,
                               col_rel::Float64=0.1,
                               track_hz::Float64=10_000.0, track_gap::Int=3,
                               max_jump_hz::Float64=20_000.0, jump_win::Int=5,
                               edge_rel::Float64=0.01, max_edge_hz::Float64=10_000.0,
                               energy_rel::Float64=0.1,
                               use_mask::Bool=true, mask_rel::Float64=0.02,
                               mask_z::Float64=3.0, mask_dilate::Int=5,
                               harmonic_box::Symbol=:all,
                               ridge_q_lo::Float64=0.02, ridge_q_hi::Float64=0.99,
                               margin_bins::Int=4, min_height_bins::Int=6,
                               bound_mode::Symbol=:continuous, edge_jump_hz::Float64=4_000.0,
                               subbin::Bool=true,
                               measure_sng::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                               measure_floor::Union{Nothing,AbstractVector{<:Real}}=nothing,
                               peak_search_bins::Int=2)
    harmonic_box in (:all, :fundamental) ||
        throw(ArgumentError("harmonic_box must be :all or :fundamental"))
    bound_mode in (:continuous, :quantile) ||
        throw(ArgumentError("bound_mode must be :continuous or :quantile (got $bound_mode)"))
    measure_sng === nothing || size(measure_sng) == size(sng) ||
        throw(DimensionMismatch("measure_sng must have the same size as sng"))
    measure_floor === nothing || length(measure_floor) == size(sng, 1) ||
        throw(DimensionMismatch("measure_floor must have length size(sng, 1)"))
    n_peaks >= 1 || throw(ArgumentError("n_peaks must be >= 1"))
    max_harmonic >= 2 || throw(ArgumentError("max_harmonic must be >= 2"))
    # jump_win 은 예전 running-median 연속성 필터용 (호환을 위해 인자만 유지)

    ncol = size(sng, 2)
    on, off = minmax(clamp(Int(on), 1, ncol), clamp(Int(off), 1, ncol))
    lo, hi = _band_bins(f, flow, fhigh, size(sng, 1))
    df = length(f) >= 2 ? abs(Float64(f[2] - f[1])) : 1.0
    min_sep = max(1, round(Int, min_sep_hz / df))
    max_edge = max(0, round(Int, max_edge_hz / df))
    hkw = (; tol=harm_tol, max_harmonic=max_harmonic, fs=fs, alias_tol=alias_tol)
    pad(a, b) = _ensure_min_height(a - margin_bins, b + margin_bins, min_height_bins, lo, hi)
    padf(a, b) = _ensure_min_height_f(a - margin_bins, b + margin_bins, min_height_bins, lo, hi)
    edge_jump = edge_jump_hz / df

    # 경계 측정용 값 (A: "검출은 필터 맵, 측정은 원본 맵")
    #   measure_sng === nothing → sng 그대로 (예전 동작)
    #   measure_sng 지정        → measure_sng[r,t] / measure_floor[r]  (SNR 행렬을 따로 만들지 않음)
    measuring = measure_sng !== nothing
    mval(r::Int, t::Int) = !measuring ? Float64(sng[r, t]) :
        (x = Float64(measure_sng[r, t]); x = isfinite(x) ? max(x, 0.0) : 0.0;
         measure_floor === nothing ? x : x / Float64(measure_floor[r]))

    # 결과 생성 (모든 return 이 같은 필드 순서). 경계 인자는 정수/실수 bin 모두 가능.
    #   contour_t / contour_hz : 경계 계산에 쓴 fundamental ridge (컬럼, 서브빈 Hz)
    function result(b_lo, b_hi, fb_lo, fb_hi; is_harmonic=false, harm_frac=0.0,
                    n_components=1, orders=Int[], n_aliased=0, ratio=missing,
                    harm_lo=missing, harm_hi=missing, valid=true, n_cols=0, fallback=false,
                    contour_t=Int[], contour_hz=Float64[])
        return (; f_lo=_bin_to_hz(f, b_lo), f_hi=_bin_to_hz(f, b_hi),
                  is_harmonic, harm_frac, n_components,
                  orders, max_order=isempty(orders) ? 1 : maximum(orders),
                  n_aliased, ratio,
                  fund_lo=_bin_to_hz(f, fb_lo), fund_hi=_bin_to_hz(f, fb_hi),
                  harm_lo, harm_hi, valid, n_cols, fallback, contour_t, contour_hz)
    end
    # 쓸 컬럼이 없을 때: 세그먼트 에너지 profile → 그것도 없으면 전체 대역 (valid=false)
    function fallback_result()
        band = _energy_band(sng, lo, hi, on, off, energy_rel)
        band === nothing && return result(lo, hi, lo, hi; n_components=0, valid=false)
        a, b = pad(band...)
        return result(a, b, a, b; fallback=true)
    end
    # 측정 맵에서 peak 재정렬: 검출 맵(필터)의 peak b0 주변 ±peak_search_bins 안의 최대 bin
    function refine_peak(b0::Int, t::Int)
        measuring || return b0
        b = b0; best = mval(b0, t)
        for r in max(lo, b0 - peak_search_bins):min(hi, b0 + peak_search_bins)
            v = mval(r, t)
            v > best && ((b, best) = (r, v))
        end
        return b
    end
    # 측정 맵 컬럼 잡음 (band 내 median). 측정 맵이 없으면 검출 단계의 noise 를 그대로 씀.
    meas_noise(t::Int, noise::Float64) =
        measuring ? median(Float64[mval(r, t) for r in lo:hi]) : noise

    # 컬럼 t 에서 peak bin b 의 ridge 폭: 값이 thr 이상으로 이어지는 [a, c] (최대 max_edge bin)
    #   subbin=true 면 임계를 지나는 위치를 dB 보간한 실수 bin 으로 반환 (B)
    function ridge_edges(b0::Int, t::Int, noise::Float64)
        b = refine_peak(b0, t)
        v(r) = mval(r, t)
        pk = v(b)
        thr = max(edge_rel * pk, local_k * noise)
        pk >= thr || return float(b), float(b)       # 측정 맵에서 임계 미만 → peak 한 점
        a = b
        while a > lo && b - (a - 1) <= max_edge && v(a - 1) >= thr
            a -= 1
        end
        c = b
        while c < hi && (c + 1) - b <= max_edge && v(c + 1) >= thr
            c += 1
        end
        subbin || return float(a), float(c)
        ea = (a > lo && v(a - 1) < thr) ? a - _cross_frac(v(a), v(a - 1), thr) : float(a)
        ec = (c < hi && v(c + 1) < thr) ? c + _cross_frac(v(c), v(c + 1), thr) : float(c)
        return ea, ec
    end

    # 0) 세그먼트 에너지 mask (짧은 콜에서 잡음 컬럼 seed 차단용)
    mask = use_mask ? _energy_mask(sng, lo, hi, on, off; rel=mask_rel, z=mask_z,
                                   dilate=mask_dilate) : trues(hi - lo + 1)
    mask === nothing && return fallback_result()

    # 1) 후보 컬럼: peak / harmonic 집합
    #    harmonic 후보 peak 는 mask 안의 것만, 가장 강한 peak 는 mask 밖이어도 일단 보관
    #    (mask 밖 컬럼은 seed 는 못 되고, 2) 의 tracking 으로 이어질 때만 채택)
    tcols = Int[]; colpk = Float64[]; colnoise = Float64[]; inmask = Bool[]
    sets = Vector{NamedTuple{(:fund, :harms, :orders, :aliased),
                             Tuple{Int,Vector{Int},Vector{Int},Vector{Bool}}}}()
    for t in on:off
        col = @view sng[lo:hi, t]
        pk = maximum(col)
        (isfinite(pk) && pk > 0) || continue
        noise = median(col)                           # 0 포함 (denoise 된 컬럼이면 보통 0)
        isfinite(noise) || continue
        pk > local_k * noise || continue
        pks = _column_peaks(col, n_peaks, peak_rel, min_sep, local_k * noise)
        isempty(pks) && continue
        im = mask[pks[1]]
        pks = vcat(pks[1:1], filter(r -> mask[r], pks[2:end])) .+ (lo - 1)
        push!(tcols, t); push!(colpk, pk); push!(colnoise, Float64(noise)); push!(inmask, im)
        push!(sets, _harmonic_set(pks, f; hkw...))
    end
    any(inmask) || return fallback_result()

    # 2) seed (mask 안 & 강한 컬럼) + hysteresis tracking 으로 약한 꼬리 컬럼 채택
    ref = quantile(colpk[inmask], 0.9)
    acc = inmask .& (colpk .>= col_rel * ref)
    if track_hz > 0
        changed = true
        while changed
            changed = false
            for i in eachindex(sets)
                acc[i] && continue
                for j in (i - 1, i + 1)
                    (1 <= j <= length(sets) && acc[j]) || continue
                    abs(tcols[i] - tcols[j]) <= track_gap || continue
                    abs(f[sets[i].fund] - f[sets[j].fund]) <= track_hz || continue
                    acc[i] = true
                    changed = true
                    break
                end
            end
        end
    end
    any(acc) || return fallback_result()
    sets = sets[acc]; tcols = tcols[acc]; colnoise = colnoise[acc]

    n_harm = count(s -> !isempty(s.harms), sets)
    harm_frac = n_harm / length(sets)
    is_harm = n_harm > 0 && harm_frac >= min_harm_frac

    # 3) fundamental / harmonic 분리 (시간 순서 유지, 컬럼 index 같이 보관)
    fund_bins = Int[]; fund_i = Int[]
    harm_bins = Int[]; harm_i = Int[]
    ratios = Float64[]
    order_count = Dict{Int,Int}()
    n_aliased = 0
    f0_med = is_harm ? median([Float64(f[s.fund]) for s in sets if !isempty(s.harms)]) : 0.0
    loose = merge(hkw, (; tol=2harm_tol, alias_tol=2alias_tol))
    for (i, s) in enumerate(sets)
        if is_harm && !isempty(s.harms)
            push!(fund_bins, s.fund); push!(fund_i, i)
            append!(harm_bins, s.harms); append!(harm_i, fill(i, length(s.harms)))
            for (p, k, al) in zip(s.harms, s.orders, s.aliased)
                order_count[k] = get(order_count, k, 0) + 1
                al || push!(ratios, f[p] / f[s.fund])
            end
            any(s.aliased) && (n_aliased += 1)
        elseif is_harm && first(_harmonic_order(f[s.fund], f0_med; loose...)) > 0
            push!(harm_bins, s.fund); push!(harm_i, i)   # 기본음이 사라지고 harmonic 만 남은 컬럼
        else
            push!(fund_bins, s.fund); push!(fund_i, i)   # 비-harmonic: 가장 강한 peak
        end
    end
    if isempty(fund_bins)
        fund_bins = [s.fund for s in sets]; fund_i = collect(eachindex(sets))
    end

    # 4) 고립점 제거: 앞/뒤 이웃 모두와 max_jump_hz 넘게 떨어진 fundamental 만 버림
    #    (sweep 은 이웃끼리 가깝고, 실제 frequency jump 는 양쪽 중 한쪽과는 가까움)
    if length(fund_bins) >= 3 && isfinite(max_jump_hz)
        ff = Float64[f[b] for b in fund_bins]
        n = length(ff)
        ok = [min(i > 1 ? abs(ff[i] - ff[i-1]) : Inf,
                  i < n ? abs(ff[i] - ff[i+1]) : Inf) <= max_jump_hz for i in 1:n]
        if any(ok)
            fund_bins = fund_bins[ok]; fund_i = fund_i[ok]
        end
    end
    orders = sort([k for (k, n) in order_count if n >= 2])   # 한 컬럼에서만 보인 차수는 무시

    # 5) ridge 폭(edge) 으로 경계
    #    bound_mode=:continuous (기본) 이웃 컬럼과 이어진 극단 edge 의 min/max   (E)
    #    bound_mode=:quantile           컬럼별 edge 의 분위수 (ridge_q_lo / ridge_q_hi, 예전 동작)
    #    subbin=false 이고 :quantile 이면 예전 결과와 같음 (정수 bin 으로 floor/ceil)
    noise_cache = Dict{Int,Float64}()
    noise_of(i) = get!(() -> meas_noise(tcols[i], colnoise[i]), noise_cache, i)
    function band_from(bins, idx)
        e = [ridge_edges(b, tcols[i], noise_of(i)) for (b, i) in zip(bins, idx)]
        if bound_mode === :quantile
            a = quantile(first.(e), ridge_q_lo)
            c = quantile(last.(e), ridge_q_hi)
        else
            a, c = _continuous_extent(idx, e, edge_jump)
        end
        subbin && return padf(a, c)
        return pad(floor(Int, a), ceil(Int, c))
    end

    # fundamental ridge contour (서브빈). feature 계산/검증용으로 반환.
    contour_t = [tcols[i] for i in fund_i]
    contour_hz = map(zip(fund_bins, fund_i)) do (b0, i)
        t = tcols[i]
        b = refine_peak(b0, t)
        _bin_to_hz(f, subbin ? _parabolic_peak(r -> mval(r, t), b, lo, hi) : b)
    end

    fb_lo, fb_hi = band_from(fund_bins, fund_i)
    if is_harm && !isempty(harm_bins)
        hb_lo, hb_hi = band_from(harm_bins, harm_i)
        harm_lo, harm_hi = _bin_to_hz(f, hb_lo), _bin_to_hz(f, hb_hi)
    else
        hb_lo, hb_hi = fb_lo, fb_hi
        harm_lo, harm_hi = missing, missing
    end

    use_all = is_harm && harmonic_box === :all
    b_lo = use_all ? min(fb_lo, hb_lo) : fb_lo
    b_hi = use_all ? max(fb_hi, hb_hi) : fb_hi

    return result(b_lo, b_hi, fb_lo, fb_hi;
                  is_harmonic=is_harm, harm_frac=harm_frac,
                  n_components=is_harm ? 1 + maximum(s -> length(s.harms), sets) : 1,
                  orders=orders, n_aliased=n_aliased,
                  ratio=isempty(ratios) ? missing : median(ratios),
                  harm_lo=harm_lo, harm_hi=harm_hi, n_cols=length(sets),
                  contour_t=contour_t, contour_hz=contour_hz)
end

# 세그먼트(on:off)의 시간 합 주파수 profile 에서 배경보다 유의하게 높은 행 mask (band-local).
#   threshold = max(rel·최대, z·1.4826·MAD)  (배경 = profile 중앙값)
#   denoise 된 입력이면 MAD≈0 이라 rel 기준만, SNR/raw 입력이면 잡음 합의 흔들림 기준이 같이 작동.
#   유의한 행이 없으면 nothing.
function _energy_mask(sng::AbstractMatrix{<:Real}, lo::Int, hi::Int, on::Int, off::Int;
                      rel::Real=0.02, z::Real=4.0, dilate::Int=5)
    prof = vec(sum(x -> isfinite(x) ? max(Float64(x), 0.0) : 0.0, @view(sng[lo:hi, on:off]); dims=2))
    d = prof .- median(prof)
    m = maximum(d)
    m > 0 || return nothing
    thr = max(rel * m, z * 1.4826 * median(abs.(d)))
    core = findall(>=(thr), d)
    isempty(core) && return nothing
    mask = falses(length(d))
    for i in core
        mask[max(1, i - dilate):min(length(d), i + dilate)] .= true
    end
    return mask
end

# 세그먼트(on:off)의 주파수 에너지 profile 에서 최대점 주변 대역 (band-local → 전체 bin).
#   profile 의 중앙값(배경)을 빼고, 최대값의 rel 배 이상으로 이어진 구간. 신호가 없으면 nothing.
function _energy_band(sng::AbstractMatrix{<:Real}, lo::Int, hi::Int, on::Int, off::Int, rel::Real)
    prof = vec(sum(x -> isfinite(x) ? max(Float64(x), 0.0) : 0.0, @view(sng[lo:hi, on:off]); dims=2))
    prof .-= median(prof)
    m = argmax(prof)
    prof[m] > 0 || return nothing
    thr = rel * prof[m]
    a, b = m, m
    while a > 1 && prof[a-1] >= thr
        a -= 1
    end
    while b < length(prof) && prof[b+1] >= thr
        b += 1
    end
    return a + lo - 1, b + lo - 1
end

# 여러 세그먼트에 대해 harmonic 분석
function usv_segment_harmonics(sng::AbstractMatrix{<:Real}, f::AbstractVector{<:Real},
                               ON::AbstractVector{<:Integer}, OFF::AbstractVector{<:Integer};
                               kwargs...)
    on, off = _sanitize_onoff(ON, OFF, size(sng, 2))
    return [usv_harmonic_analysis(sng, f, a, b; kwargs...) for (a, b) in zip(on, off)]
end

# =====================================================================
# 4. GUI 변환  (XY, UserUnit, sequence_check! 는 GUI 쪽 정의 필요)
#   좌표계: y = 1 - f/fmax  (y 작을수록 고주파)
# =====================================================================

# --- GUI 주파수 좌표 규약 (gui_function_temp.jl 의 userunit_to_freq 와 공유) ---
#   y 는 0(위, 고주파) ~ 1(아래, 저주파). nfreq = size(fout, 1).
#   y(bin) = (nfreq + 1 - bin)/nfreq  ↔  bin(y) = nfreq + 1 - round(y·nfreq)  (정확히 역함수)
#   기존 userunit_to_freq(257 고정) 공식과 같아서 저장된 수동 box 와 호환됨.
freq_bin_to_userunit(bin::Integer, nfreq::Integer) = (nfreq + 1 - clamp(bin, 1, nfreq)) / nfreq
function userunit_to_freq_bin(y::Real, nfreq::Integer)
    isnan(y) && return 1                          # NaN → 저주파 끝 (box 가 이상하면 drawing 단계에서 보임)
    isinf(y) && return y > 0 ? 1 : nfreq
    return clamp(nfreq + 1 - round(Int, y * nfreq), 1, nfreq)
end
_hz_to_bin(f::AbstractVector{<:Real}, hz::Real) = argmin(abs.(f .- hz))
_hz_to_userunit(f::AbstractVector{<:Real}, hz::Real) = freq_bin_to_userunit(_hz_to_bin(f, hz), length(f))

# 고정 대역 box (기존)
function detection_to_liness(ON, OFF, spec;
                             flow::Float64=15_000.0, fhigh::Float64=125_000.0)
    out = Vector{Vector{XY{UserUnit}}}()
    y_low  = _hz_to_userunit(spec.f, fhigh)   # 작은 y = 고주파
    y_high = _hz_to_userunit(spec.f, flow)

    for (on, off) in zip(ON, OFF)
        on  = clamp(on, 1, length(spec.t_sng))
        off = clamp(off, 1, length(spec.t_sng))
        t1 = Float64(spec.t_sng[on])
        t2 = Float64(spec.t_sng[off])
        push!(out, [XY{UserUnit}(t1, y_low), XY{UserUnit}(t2, y_high)])
    end

    sequence_check!(out)
    return out
end

# 이미 계산된 (f_lo, f_hi) 로 box 생성
function bounds_to_liness(ON, OFF, bounds::AbstractVector{<:Tuple}, spec)
    length(bounds) == length(ON) == length(OFF) ||
        throw(DimensionMismatch("ON, OFF, bounds must have equal length"))
    out  = Vector{Vector{XY{UserUnit}}}()
    nt   = length(spec.t_sng)

    for ((on, off), (f_lo, f_hi)) in zip(zip(ON, OFF), bounds)
        a, b = minmax(clamp(Int(on), 1, nt), clamp(Int(off), 1, nt))
        t1 = Float64(spec.t_sng[a])
        t2 = Float64(spec.t_sng[b])
        y_hi = _hz_to_userunit(spec.f, f_hi)   # 작은 y = 고주파
        y_lo = _hz_to_userunit(spec.f, f_lo)   # 큰 y  = 저주파
        push!(out, [XY{UserUnit}(t1, y_hi), XY{UserUnit}(t2, y_lo)])
    end

    sequence_check!(out)
    return out
end

# 시간 ON/OFF + 개별 주파수 경계 box. fmethod 로 :ridge / :threshold 선택.
function detection_to_liness_2d(ON, OFF, spec, sng::AbstractMatrix{<:Real};
                                fmethod::Symbol=:ridge,
                                flow::Float64=15_000.0, fhigh::Float64=110_000.0,
                                kwargs...)
    bounds = usv_segment_freq_bounds(sng, spec.f, ON, OFF;
                                     fmethod=fmethod, flow=flow, fhigh=fhigh, kwargs...)
    return bounds_to_liness(ON, OFF, bounds, spec)
end

# =====================================================================
# 5. Pipeline: 시간축 detection + 주파수 경계를 한 번에 (GUI 불필요)
#   detect_kw : usv_whole_2d 에 전달할 NamedTuple   예) (; fp_mode=:confident)
#   freq_kw   : 주파수 방식에 전달할 NamedTuple      예) (; noise_k=2.0)
#   harmonics : true 면 harmonic 분석 결과(harm)도 반환.
#               fmethod=:ridge 이면 box 도 harmonic 분석에서 가져옴 (harm_kw 우선, freq_kw 와 병합).
#               fmethod=:threshold 이면 box 는 threshold 방식, harm 은 정보로만 제공.
#   harm_kw   : usv_harmonic_analysis 옵션  예) (; n_peaks=3, harmonic_box=:fundamental)
#   harm_sng  : harmonic 분석에 쓸 별도 spectrogram (예: snr_spectrogram(spec.fout)).
#               nothing 이면 sng(denoise 된 것) 사용 — 약한 harmonic 이 denoise 로 지워질 수 있음.
#   measure_sng / measure_floor : 경계(edge) 측정용 원본 spectrogram 과 행별 잡음 바닥
#               (예: spec.fout, row_noise_floor(spec.fout)). harm_sng(필터 맵)로 ridge 를 찾고
#               폭은 이 맵에서 잼 — SDTS 가 깎은 ridge 옆 에너지 때문에 box 가 좁아지는 것 방지.
#               nothing 이면 harm_sng 에서 측정 (예전 동작).
#   반환: (ON, OFF, bounds, harm)   bounds[k] = (f_lo, f_hi) Hz, harm 은 harmonics=false 면 nothing
#   ※ harmonic 이 110 kHz 위에 있으면 fhigh 를 올리고 denoise_usv 의 fhigh 도 같이 올릴 것.
# =====================================================================
function usv_detect_2d(sng::Matrix{Float64}, dt_sng::Float64,
                       t_sng::AbstractVector, f::AbstractVector{<:Real};
                       fmethod::Symbol=:ridge,
                       flow::Float64=15_000.0, fhigh::Float64=125_000.0,
                       detect_kw::NamedTuple=(;), freq_kw::NamedTuple=(;),
                       harmonics::Bool=false, harm_kw::NamedTuple=(;),
                       harm_sng::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                       measure_sng::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                       measure_floor::Union{Nothing,AbstractVector{<:Real}}=nothing)
    ON, OFF = usv_whole_2d(sng, dt_sng, t_sng, f; detect_kw...)

    harm = nothing
    if harmonics
        hs = harm_sng === nothing ? sng : harm_sng
        size(hs) == size(sng) || throw(DimensionMismatch("harm_sng must have the same size as sng"))
        hkw = fmethod === :ridge ? merge(freq_kw, harm_kw) : harm_kw
        measure_sng === nothing ||
            (hkw = merge(hkw, (; measure_sng=measure_sng, measure_floor=measure_floor)))
        harm = usv_segment_harmonics(hs, f, ON, OFF; flow=flow, fhigh=fhigh, hkw...)
    end

    bounds = if harmonics && fmethod === :ridge
        Tuple{Float64,Float64}[(h.f_lo, h.f_hi) for h in harm]
    else
        usv_segment_freq_bounds(sng, f, ON, OFF;
                                fmethod=fmethod, flow=flow, fhigh=fhigh, freq_kw...)
    end
    return (; ON, OFF, bounds, harm)
end