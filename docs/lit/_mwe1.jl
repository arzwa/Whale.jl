using DynamicHMC, Whale, DistributedArrays, Distributions, Random
using DynamicHMC.Diagnostics, Test

# ## MWE
using TransformVariables, LogDensityProblems, Parameters, ForwardDiff
import TransformVariables: TransformTuple
import Distributions: logpdf

struct TheProblem
    prior
    trans
end

abstract type Prior end
@with_kw struct ThePrior <: Prior
    r::MvNormal = MvNormal(ones(2))
    q::Beta = Beta(3,1)
end

logpdf(prior::ThePrior, θ) = sum(logpdf(prior.r, θ.r)) + logpdf(prior.q, θ.q)
# logpdf(prior::ThePrior, θ) = sum(logpdf(prior.r, log.(θ.r))) + logpdf(prior.q, θ.q)

trans(p::ThePrior) = as((r=as(Array, asℝ, 2), q=as𝕀))
# trans(p::ThePrior) = as((r=as(Array, asℝ₊, 2), q=as𝕀))

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

function LogDensityProblems.logdensity_and_gradient(p::TheProblem, x)
    @unpack prior, trans = p
    # here comes a complicated ℓ, ∇ℓ = fand∇f(model, data, ...)
    π, ∇π = fand∇f(prior, trans, x)
    J, ∇J = fand∇f(trans, x)
    return (π + J)::Float64, (@. ∇π + ∇J)::Vector{Float64}
end

LogDensityProblems.capabilities(::Type{<:TheProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(p::TheProblem) = dimension(p.trans)

# example
prior = ThePrior()
problem = TheProblem(prior, trans(prior))
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 1000)
posterior = transform.(problem.trans, results.chain)
# x = log.([x.r[1] for x in posterior])
x = [x.r[1] for x in posterior]
@show mean(x)

# OK, turns out the problem was with the transformations after all...
