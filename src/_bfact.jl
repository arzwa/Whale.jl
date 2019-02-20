# Arthur Zwaenepoel - 2018
# Functions to calculate the Bayes factor (B) to do model selection
# NOTE: Still not 100% sure about the calculation of p(q₀) = ∫p(q₀, ψ)dψ
# TODO: Add an option in the config whether to calculate the bayes factor for a particular WGD
# or not. Could be in an [bayesfactor] section, with `wgd id` = true or something
"""
    bayesfactor_sd(df::DataFrame, i::Int64, ccd, chain, prior; burnin=1000)
Compute `B` for the hypothesis that WGD `i` has qᵢ ≠ 0 (H1) vs. the hypothesis
that qᵢ = 0 (H0). This uses the Savage-Dickey density ratio (Verdinelli & Wasserman
1995; Suchard et al. 2001). Calculation amounts to ∫p(q₀, ψ|x)dψ / ∫p(q₀, ψ)dψ.
"""
function bayesfactor_sd(df::DataFrame, i::Int64, ccd::DArray{CCD}, chain::ChainSettings,
        prset::PriorSettings; burnin::Int64=1)
    n = 0.
    d = 0.
    @show nrow(df)
    for j in 1:nrow(df)
        post, prior = marginal_posterior(df[j, :], i, ccd, chain, prset)
        n += post
        d += prior
    end
    @info "∫p(q₀, ψ|x)dψ ≈ $n"
    @info "  ∫p(q₀, ψ)dψ ≈ $d"
    # note that the averaging term 1/N is canceled since it appears both in the
    # prior and marginal posterior calculations.
    return n / d
end

"""
    marginal_posterior(row, i, ccd, chain, prior)
Compute the marginal posterior density for qᵢ = 0. Equation (2) of Verdinelli & Wasserman (1995),
operates on a row from the posterior data frame. Computes a 'brute force' estimator of
p(q₀|x) = ∫p(q₀, ψ|x)dψ.
"""
function marginal_posterior(row, i::Int64, ccd::DArray{CCD}, chain::ChainSettings,
        prset::PriorSettings)
    # get rates
    λ, μ, q = get_rates_dfrow(row)
    q[i] = 0.0  # qᵢ = 0
    η = chain.state["η"][1]

    # Compute numerator
    @show lhood = evaluate_lhood!(ccd, λ, μ, q, chain, prset)           # (a) Compute p(x|q0, ψ)
    @show prior = evaluate_prior(chain.S, λ, μ, q, chain.state, prset)  # (b) Compute p(q0, ψ)
    Whale.set_recmat!(ccd)

    # Compute denominator
    denominator = approximate_integral_midpoint(λ, μ, q, η, i, lhood+prior, ccd, chain, prset)
    return lhood + prior - denominator, prior
end

# Can I just integrate over the loglikelihood ??
# Nope: https://www.johndcook.com/blog/2012/07/26/avoiding-underflow-in-bayesian-computations/
# - Find m, the maximum of the log of the integrand.
# - Let I be the integral of exp( log of the integrand – m ).
# - Keep track that your actual integral is exp(m) I, or that its log is m + log I.
"""
    approximate_integral_midpoint(λ, μ, q, η, i, m, ccd, chain, prset; δ=0.1)
Approximate the one-dimensional integral ∫p(q, ψ)p(q, ψ)dq using the trapezoidal rule.
Note that `m` is a correction factor to prevent underflow, it should be on the same order
as the log of the posterior probability, so just provide the already computed posterior
denisty at q₀.
"""
function approximate_integral_midpoint(λ, μ, q, η, i, m, ccd, chain, prset; δ=0.1)
    integral = 0.
    for qᵢ = 0:δ:(1. - δ)
        q = [q[1:i-1] ; qᵢ + δ/2 ; q[i+1:end]]
        node = get_branch(i, chain.wgds)
        lhood = evaluate_lhood!(ccd, λ, μ, q, node, chain, prset)
        prior = evaluate_prior(chain.S, λ, μ, q, chain.state, prset)
        integral += exp(prior + lhood - m) * δ
    end
    return m + log(integral)
end

# get the rates from a row of the posterior, assumes the row is correctly sorted.
function get_rates_dfrow(row)
    λ::Array{Float64} = collect(Iterators.flatten(
        [x[2] for x in eachcol(row) if startswith(string(x[1]), "l")][1:end-1]))
    μ::Array{Float64} = collect(Iterators.flatten(
        [x[2] for x in eachcol(row) if startswith(string(x[1]), "m")]))
    q::Array{Float64} = collect(Iterators.flatten(
        [x[2] for x in eachcol(row) if startswith(string(x[1]), "q")]))
    return λ, μ, q
end

get_branch(i, wgds) = [k for (k, v) in wgds if i in v][1]
