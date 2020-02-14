# Potentially worth to open issues in DynamicHMC:
# - save intermediate chains (so that we can run a chain indefinitely, and stop
#   it when we're happy)
# NOTE: currently this does not work (well) in the case where there would be
# hyperparameters that are sampled but do not end up in the RatesModel, however
# the RatesModel layer in between the prior and the model provides an opportunity
# to handle this.
"""
    WhaleProblem

A generic Whale 'problem' interface ̀a la LogDensityProblems.jl. This holds
a TransformVariables transformation, prior, ratesmodel, data and WhaleModel.
This struct defines all DynamicHMC related functionalities and can be
constructed from a WhaleModel instance, data set and prior struct (the rationale
is that the prior struct full defines the problem).
"""
struct WhaleProblem{V<:Prior,R,T,I,U}
    data ::CCDArray{I,U}  # NOTE require DArray
    model::WhaleModel{I,U}
    prior::V
    rates::R
    trans::T
end

function WhaleProblem(wm::WhaleModel, data::CCDArray, prior::P) where P<:Prior
    rates = RatesModel(prior)
    WhaleProblem(data, wm, prior, rates, trans(prior, wm))
end

function fand∇f(f::Function, x, cfg)
    result = DiffResults.GradientResult(x)
    result = ForwardDiff.gradient!(result, f, x, cfg)
    DiffResults.value(result), DiffResults.gradient(result)
end

function fand∇f(prior::Prior, t, x)
    fun = (x) -> logpdf(prior, t(x))
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    return fand∇f(fun, x, cfg)
end

function fand∇f(trans::TransformTuple, x)
    fun = (x) -> transform_and_logjac(trans, x)[2]
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    return fand∇f(fun, x, cfg)
end

# parallel computation of ℓ and ∇ℓ, all derivation is within parallel processes
# and the partial values are accumulated on the main process
function fand∇f(wm::WhaleModel, r, data::CCDArray, t, x)
    y = [@spawnat i _grad(wm, r, localpart(data), t, x) for i in procs(data)]
    result = fetch.(y)
    acc = foldl(+, result)
    acc[1], acc[2:end]
end

function _grad(wm::WhaleModel, r, data::Vector, t, x)
    function fun(x)
        model = wm(r(t(x)))  # sets the model
        mapreduce(u->logpdf(model, u), +, data)
    end
    cfg = ForwardDiff.GradientConfig(fun, x, ForwardDiff.Chunk{length(x)}())
    vcat(fand∇f(fun, x, cfg)...)
end

ℓand∇ℓ(p::WhaleProblem, x) = fand∇f(p.model, p.rates, p.data, p.trans, x)

function LogDensityProblems.logdensity_and_gradient(p::WhaleProblem, x)
    @unpack model, prior, data, trans, rates = p
    ℓ, ∇ℓ = fand∇f(model, rates, data, trans, x)
    π, ∇π = fand∇f(prior, trans, x)
    J, ∇J = fand∇f(trans, x)
    return (ℓ + π + J)::Float64, (@. ∇ℓ + ∇π + ∇J)::Vector{Float64}
end

LogDensityProblems.capabilities(::Type{<:WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(p::WhaleProblem) = dimension(p.trans)

backtrack(p::WhaleProblem, posterior) =
    backtrack(p.model, p.data, posterior, p.rates)
