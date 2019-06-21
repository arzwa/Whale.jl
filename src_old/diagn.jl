# MCMC diagnostics (and plots)
# In general I don't think we plan to run multiple chains
function diagnostics(fname::String; burnin=1000)
    post = CSV.read(fname)
    post = post[burnin:end, :]
    return diagnostics(post)
end

function diagnostics(post::DataFrame)
    ar = colwise(acceptance_proportion, post)
    @info "Acceptance proportions"
    println(ar)
    @info "Summary"
    sumry = describe(
        post[2:end], stats=[:mean, :std, :q25, :median, :q75])
    sumry[:ESS] = colwise(effective_sample_size, post[2:end])
    print(sumry)
    println()
    @info "Harmonic mean estimator of marginal likelihood"
    @show harmonicmean(post[:lhood])
    return sumry
end

function harmonicmean(lhood)
    m = minimum(lhood)
    lhood_ = lhood .- m
    return log(1.0/(sum([1.0/exp(x) for x in lhood_])/length(lhood_))) + m
end

"""
Compute acceptance ratio for a sample of one parameter.
"""
function acceptance_proportion(x::AbstractVector)
    return length([x[i] for i in 2:length(x) if x[i] != x[i-1]])/length(x)
end

# NOTE: Code below from https://github.com/tpapp/MCMCDiagnostics.jl (MIT)
"""
    autocorrelation(x, k, v = var(x))
Estimate of lag-`k` autocorrelation of `x` from a variogram. `v` is the variance
of `x`, used when supplied. See Gelman et al (2013), section 11.4.
"""
function autocorrelation(x::AbstractVector, k::Integer, v = var(x))
    x1 = @view(x[1:(end-k)])
    x2 = @view(x[(1+k):end])
    V = sum((x1 .- x2).^2) / length(x1)
    1 - V / (2*v)
end

"""
    ess_factor_estimate(x, v = var(x))
Estimate for effective sample size factor. Return `τ, K` where `τ` is estimated
effective sample size / sample size, and `K` is the last lag used for autocorrelation
estimation.

# Notes
See Gelman et al (2013), section 11.4.

`τ` is capped at 1, this is relevant when the sample has large negative
autocorrelation (happens with HMC/NUTS). Some implementations (eg Stan) use FFT
for autocorrelations, which yields the whole spectrum. In practice, a <50-100
lags are usually sufficient for reasonable samplers, so the "naive" version may
be more efficient.
"""
function ess_factor_estimate(x::AbstractVector, v = var(x))
    N = length(x)
    τ_inv = 1 + 2 * autocorrelation(x, 1, v)
    K = 2
    while K < N - 2
        Δ = autocorrelation(x, K, v) + autocorrelation(x, K + 1, v)
        if Δ < 0
            break
        else
            τ_inv += 2*Δ
            K += 2
        end
    end
    min(1 / τ_inv, one(τ_inv)), K
end


"""
    effective_sample_size(x, v = var(x))
Effective sample size of vector `x`.
Estimated from autocorrelations. See Gelman et al (2013), section 11.4.
When the variance `v` is supplied, it saves some calculation time.
"""
function effective_sample_size(x::AbstractVector, v = var(x))
    τ, _ = ess_factor_estimate(x, v)
    τ * length(x)
end


"""
    potential_scale_reduction(chains...)
Potential scale reduction factor (for possibly ragged chains).
Also known as R̂. Always ≥ 1 by construction, but values much larger than 1 (say
1.05) indicate poor mixing.
Uses formula from Stan Development Team (2017), section 28.3.
"""
function potential_scale_reduction(chains::AbstractVector...)
    mvs = mean_and_var.(chains)
    W = mean(last.(mvs))
    B = var(first.(mvs))
    √(1 + B / W)
end


# NOTE: MCSE related functions - code from Mamba (MIT)
"""
Compute the Markov Chain/Monte Carlo standard error.
"""
function mcse(x::Vector{T}, method::Symbol=:imse; args...) where {T<:Real}
    method == :bm ? mcse_bm(x; args...) :
    method == :imse ? mcse_imse(x) :
    method == :ipse ? mcse_ipse(x) :
        throw(ArgumentError("unsupported mcse method $method"))
end

# using batch means
function mcse_bm(x::Vector{T}; size::Integer=100) where {T<:Real}
    n = length(x)
    m = div(n, size)
    m >= 2 ||
    throw(ArgumentError(
        "iterations are < $(2 * size) and batch size is > $(div(n, 2))"
    ))
    mbar = [mean(x[i * size .+ (1:size)]) for i in 0:(m - 1)]
    sem(mbar)
end

# using initial monotone sequence (see HB of MCMC Geyer p. 16)?
function mcse_imse(x::Vector{T}) where {T<:Real}
    n = length(x)
    m = div(n - 2, 2)
    ghat = autocov(x, [0, 1])
    Ghat = sum(ghat)
    value = -ghat[1] + 2 * Ghat
    for i in 1:m
        Ghat = min(Ghat, sum(autocov(x, [2 * i, 2 * i + 1])))
        Ghat > 0 || break
        value += 2 * Ghat
    end
    sqrt(value / n)
end

# using initial positive sequence (see HB of MCMC Geyer p. 16)
function mcse_ipse(x::Vector{T}) where {T<:Real}
    n = length(x)
    m = div(n - 2, 2)
    ghat = autocov(x, [0, 1])
    value = ghat[1] + 2 * ghat[2]
    for i in 1:m
        Ghat = sum(autocov(x, [2 * i, 2 * i + 1]))
        Ghat > 0 || break
        value += 2 * Ghat
    end
    sqrt(value / n)
end
