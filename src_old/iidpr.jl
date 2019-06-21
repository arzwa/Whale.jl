# Independent and identically distributed rates prior
abstract type PriorSettings end

"""
    IidRates(θ, q, η)
Parametrization of the uncorrelated i.i.d. prior. We have a hyperprior on the mean
of the distribution and a fixed variance.
"""
struct IidRates <: PriorSettings
    prior_λ::Distribution{Univariate,Continuous}  # hyperprior on mean of λ
    prior_μ::Distribution{Univariate,Continuous}  # hyperprior on mean of μ
    prior_q::Distributions.Beta{Float64}          # retention rates ∼ Beta
    prior_η::Distributions.Beta{Float64}          # geometric prior on root ∼ Beta
    fixed_η::Bool
    ν::Float64                                    # fixed variance of lognormal on rates

    function IidRates(λ::Tuple{Float64,Float64}, μ::Tuple{Float64,Float64},
            q::Tuple{Float64,Float64}, η::Tuple, ν::Tuple)
        prior_λ = LogNormal(log(λ[1]), λ[2])
        prior_μ = LogNormal(log(μ[1]), μ[2])
        prior_q = Beta(q[1], q[2])
        length(η) == 1 ? prior_η = Beta(η[1], 1.) : prior_η = Beta(η[1], η[2])
        new(prior_λ, prior_μ, prior_q, prior_η, length(η) == 1, ν[1])
    end
end

# iid Rates prior; draw and evaluation. Note that the output of the draw_from_prior
# function defines the `state` dictionary used in the MCMC.
"""
    draw_from_prior(S::SpeciesTree, prset::IidRates)
Take a draw from the iid rates prior, return **branch**-wise rates (λ, μ) and q.
"""
function draw_from_prior(S::SpeciesTree, prset::IidRates, nrates::Int64)
    λ1 = rand(prset.prior_λ)
    μ1 = rand(prset.prior_μ)
    λ = rand(LogNormal(log(λ1), prset.ν), nrates)
    μ = rand(LogNormal(log(μ1), prset.ν), nrates)
    λ[1] = λ1
    μ[1] = μ1
    q = rand(prset.prior_q, length(S.wgd_index))
    η = rand(prset.prior_η)
    return Dict("λ" => λ, "μ" => μ, "q" => q, "η" => [η], "ν" => [prset.ν])
end

"""
    evaluate_prior!(chain, iid::IidRates)
Evaluate the prior for the i.i.d. rates prior and the chain. Modifies the
chain.prior field.
"""
function evaluate_prior!(chain, iid::IidRates)
    chain.prior = evaluate_prior(chain.S, chain.state["λ"], chain.state["μ"],
        chain.state["q"], chain.state["η"][1], iid)
end

"""
    evaluate_prior(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, iid::IidRates)
Evaluate the prior for the i.i.d. rates prior.
"""
function evaluate_prior(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, iid::IidRates)
    # the iid rates prior is evaluated as follows: the `mean` of the iid rate distribution
    # has a hyperprior specified in iid.prior_λ (_μ), this mean is the first entry of the rate
    # array  λ[1] (μ[1]). The iid rate distribution has a fixed variance ν. The prior for the
    # rates for every branch λ[2:end] (μ[2:end]) are then lognormal with mean λ[1] (μ[1]) and
    # fixed variance ν. The other priors are straightforward.
    # hyperpriors
    logp  = logpdf(iid.prior_η, η) + logpdf(iid.prior_λ, λ[1]) + logpdf(iid.prior_μ, μ[1])
    logp += sum(logpdf.(LogNormal(log(λ[1]), iid.ν), λ[2:end]))  # λ prior
    logp += sum(logpdf.(LogNormal(log(μ[1]), iid.ν), μ[2:end]))  # μ prior
    logp += sum(logpdf.(iid.prior_q, q))                         # q prior
    return logp
end

function evaluate_prior(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        state::Dict, iid::IidRates)
    evaluate_prior(S, λ, μ, state["q"], state["η"][1], iid)
end

function evaluate_prior(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, state::Dict, iid::IidRates)
    evaluate_prior(S, λ, μ, q, state["η"][1], iid)
end

function evaluate_prior(S::SpeciesTree, q::Array{Float64}, state::Dict,
        iid::IidRates)
    evaluate_prior(S, state["λ"], state["μ"], q, state["η"][1], iid)
end
