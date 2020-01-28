using Pkg; Pkg.activate("./test")
using Parameters
using NewickTree
using BenchmarkTools
using Distributions
using DistributedArrays
using LogDensityProblems
using TransformVariables
using Random
using DynamicHMC
using ForwardDiff
import Distributions: logpdf
import LogDensityProblems: logdensity_and_gradient
# using ForwardDiff
# using Optim
include("_model.jl")
include("_ccd.jl")
include("_core.jl")
include("_grad.jl")


# https://tamaspapp.eu/LogDensityProblems.jl/dev/#Manually-calculated-derivatives-1
# idea, keep the whole log density interface for prior, use custom for ℓhood?
struct CRPrior
    wm ::WhaleModel
    πr ::MvNormal
    πq ::Beta
    πη ::Beta
end

function (problem::CRPrior)(θ)
    @unpack λ, μ, η, q = θ
    @unpack πr, πη, πq = problem
    logpdf(πη, η) + logpdf(πr, log.([λ, μ])) + sum(logpdf.(πq, q))
end

struct WhaleProblem{T,V}
    model::WhaleModel
    data ::CCDArray
    prior::V
end

function WhaleProblem(model, data, prior::CRPrior)
    t = as((λ = asℝ₊, μ = asℝ₊, q=as(Array, as𝕀, nwgd(model)), η = as𝕀))
    P = TransformedLogDensity(t, prior)
    ∇ = ADgradient(:ForwardDiff, P)
    WhaleProblem{CRPrior,typeof(∇)}(model, data, ∇)
end

function LogDensityProblems.logdensity_and_gradient(p::WhaleProblem, x)
    @unpack model, prior, data = p
    π, ∇π = logdensity_and_gradient(prior, x)
    v = prior.ℓ.transformation(x)
    r = ConstantRates(v...)
    # compute gradient and logpdf of ℓ using dedicated functions
    ℓ  = logpdf(model(r), data)
    ∇ℓ = gradient(model, r, data)
    return ℓ + π, ∇ℓ .+ ∇π
end

LogDensityProblems.capabilities(::Type{<:WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(wp::WhaleProblem{CRPrior}) = 3 + nwgd(wp.model)

# test it
wm = WhaleModel(extree, Δt=0.1)
addwgd!(wm, 5, 0.25, rand())
ccd = CCD("./example/example-ale/OG0004533.fasta.nex.treesample.ale", wm)
D = distribute(read_ale("./example/example-ale", wm))

prior = CRPrior(wm, MvNormal(ones(2)), Beta(3,1), Beta(1,1))
problem = WhaleProblem(wm, D, prior)
logdensity_and_gradient(problem, randn(4))

progress = LogProgressReport(step_interval=100, time_interval_s=10)
@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 2000,
    reporter = progress,
    initialization = (ϵ=0.5, ),
    warmup_stages = fixed_stepsize_warmup_stages())

posterior = transform.(problem.prior.ℓ.transformation, results.chain)
λ = [x.λ for x in posterior]
μ = [x.μ for x in posterior]
q = [x.q[1] for x in posterior]
η = [x.η for x in posterior]
@show mean(λ), std(λ)
@show mean(μ), std(μ)
@show mean(η), std(η)
