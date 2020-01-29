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
include("_model.jl")
include("_ccd.jl")
include("_core.jl")
include("_grad.jl")

wm = WhaleModel(extree)
addwgd!(wm, 5, 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm)[1:2])

struct CRWhaleProblem
    ccd::CCDArray
    wm ::WhaleModel
    πr ::MvNormal
    πq ::Beta
    πη ::Beta
end

function (problem::CRWhaleProblem)(θ) where T
    @unpack λ, μ, η, q = θ
    @unpack ccd, πr, πη, πq = problem
    logpdf(wm(ConstantRates(λ=λ, μ=μ, η=η, q=q)), ccd) +
        logpdf(πη, η) + logpdf(πr, log.([λ, μ])) + sum(logpdf.(πq, q))
    logpdf(wm(ConstantRates(λ=λ, μ=μ, η=η, q=q)), ccd)
end

p = CRWhaleProblem(D, wm, MvNormal(ones(2)), Beta(3,1), Beta())
p((λ=0.5, μ=0.2, q=Float64[0.2], η=0.9))

trans = as((λ = asℝ₊, μ = asℝ₊, q = as(Array, as𝕀, 1), η = as𝕀))
P = TransformedLogDensity(trans, p)
∇P = ADgradient(:ForwardDiff, P)

results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000)

posterior = transform.(trans, results.chain) |> display
# this works great!

logdensity_and_gradient(∇P, zeros(4))
# (-76.77150695086583, [15.017958801444323, -35.95111615485038, -0.11144403675049236, 0.6181645286506255])
