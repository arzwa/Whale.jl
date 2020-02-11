# Potentially worth to open issues in DynamicHMC:
# - save intermediate chains (so that we can run a chain indefinitely, and stop
#   it when we're happy)
# - flush monitoring information (can't see monitoring on cluster)
# NOTE: currently this does not work (well) in the case where there would be
# hyperparameters that are sampled but do not end up in the RatesModel, however
# the RatesModel layer in between the prior and the model provides an opportunity
# to handle this.
# NOTE: these prior definitions and transformations can be used for other,
# custom, MCMC samplers as well. Best to keep it as generic as possible
# NOTE: keep an eye on Bijectors.jl (pot. replacement of TransformVariables.jl)
"""
    WhaleProblem

A generic Whale 'problem' interface ̀a la LogDensityProblems.jl. This holds
a TransformVariables transformation, prior, ratesmodel, data and WhaleModel.
This struct defines all DynamicHMC related functionalities and can be
constructed from a WhaleModel instance, data set and prior struct (the rationale
is that the prior struct full defines the problem).
"""
struct WhaleProblem{V<:Prior,R,T,I,U}
    data ::CCDArray{I,U}  # AbstractVector ?
    model::WhaleModel{I,U}
    prior::V
    rates::R
    trans::T
end

function WhaleProblem(wm::WhaleModel, data::CCDArray, prior::P) where P<:Prior
    rates = RatesModel(prior)
    WhaleProblem(data, wm, prior, rates, trans(prior, wm))
end

function gradient(prior::Prior, t, x)
    gradfun = (x) -> logpdf(prior, t(x))
    ForwardDiff.gradient(gradfun, x)
end

function gradient(trans::TransformTuple, x)
    gradfun = (x) -> transform_and_logjac(trans, x)[2]
    ForwardDiff.gradient(gradfun, x)
end

gradient(p::WhaleProblem, x) = gradient(p.model, p.rates, p.data, p.trans, x)

function gradient(wm::WhaleModel, r, data::CCDArray, t, x)
    y = [@spawnat i _grad(wm, r, localpart(data), t, x) for i in procs(data)]
    sum(fetch.(y))
end

function _grad(wm::WhaleModel, r, data::Vector, t, x)
    function gradfun(x)
        model = wm(r(t(x)))  # sets the model
        mapreduce(u->logpdf(model, u), +, data)
    end
    cfg = ForwardDiff.GradientConfig(gradfun, x, ForwardDiff.Chunk{length(x)}())
    ForwardDiff.gradient(gradfun, x, cfg)
end

function LogDensityProblems.logdensity_and_gradient(p::WhaleProblem, x)
    @unpack model, prior, data, trans, rates = p
    v, J = transform_and_logjac(trans, x)
    π = logpdf(prior, v)
    ℓ = logpdf!(model(rates(v)), data)
    ∇ℓ = gradient(model, rates, data, trans, x)
    ∇π = gradient(prior, trans, x)
    ∇J = gradient(trans, x)
    # @show ∇ℓ, ∇π, ∇J
    return (ℓ + π + J)::Float64, (@. ∇ℓ + ∇π + ∇J)::Vector{Float64}
end

LogDensityProblems.capabilities(::Type{<:WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(p::WhaleProblem) = dimension(p.trans)

backtrack(p::WhaleProblem, posterior) =
    backtrack(p.model, p.data, posterior, p.rates)
