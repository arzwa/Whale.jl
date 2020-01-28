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

# single CCD, constant rates, this works
struct CRWhaleProblem
    ccd::CCD
    wm ::WhaleModel
    πr ::MvNormal
    πη ::Beta
end

function (problem::CRWhaleProblem)(θ)
    @unpack λ, μ, η = θ
    @unpack ccd, πr, πη = problem
    logpdf(wm(ConstantRates(θ...)), ccd) + logpdf(πη, η) + logpdf(πr, log.([λ, μ]))
end

p = CRWhaleProblem(ccd, wm, MvNormal(ones(2)), Beta(3,1))
p((λ=0.5, μ=0.2, η=0.9))

trans = as((λ = asℝ₊, μ = asℝ₊, η = as𝕀))
P = TransformedLogDensity(trans, p)
∇P = ADgradient(:ForwardDiff, P)

results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000)

posterior = transform.(trans, results.chain)
λ = first.(posterior)
@show mean(λ), std(λ)
