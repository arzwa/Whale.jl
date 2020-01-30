using Distributed
# addprocs(2)
@everywhere using Pkg
@everywhere Pkg.activate("./test")
@everywhere using Whale, ForwardDiff
using Parameters
using LogDensityProblems
using TransformVariables
using Random
using Distributions
using DistributedArrays
using DynamicHMC
@everywhere import Distributions: logpdf
import LogDensityProblems: logdensity_and_gradient, transform_and_logjac
@everywhere import Whale: ConstantRates, RatesModel, CCDArray, nwgd, nnonwgd

# https://tamaspapp.eu/LogDensityProblems.jl/dev/#Manually-calculated-derivatives-1
# idea, keep the whole log density interface for prior, use custom for ℓhood?
# Actually, we only keep the transform, and implement the rest ourselves (this)
# also makes us less dependent on the Papp packages
struct _CRPrior
    wm ::WhaleModel
    πr ::MvNormal
    πq ::Beta
    πη ::Beta
end

function (problem::_CRPrior)(θ)
    @unpack λ, μ, η, q = θ
    @unpack πr, πη, πq = problem
    logpdf(πη, η) + logpdf(πr, log.([λ, μ])) + sum(logpdf.(πq, q))
end

function gradient(problem::_CRPrior, x, t)
    function gradfun(x)
        u = t(x)
        problem(u)
    end
    ForwardDiff.gradient(gradfun, x)
end

function gradient(wm::WhaleModel, r::RatesModel, data::CCDArray, x, t)
     mapreduce((ccd)->gradient(wm, r, ccd, x, t), +, data)
end

@everywhere function gradient(wm::WhaleModel, r::RatesModel, ccd::CCD, x, t)
    function gradfun(x)
        u = t(x)
        logpdf(wm(ConstantRates(λ=u.λ, μ=u.μ, η=u.η, q=u.q)), ccd)
    end
    ForwardDiff.gradient(gradfun, x)
end


function gradjac(jac, x)
    f = (u) -> transform_and_logjac(jac, u)[2]
    ForwardDiff.gradient(f, x)
end

struct _WhaleProblem{T,V}
    model::WhaleModel
    data ::CCDArray
    prior::V
end

function _WhaleProblem(model, data, prior::_CRPrior)
    t = as((λ = asℝ₊, μ = asℝ₊, q=as(Array, as𝕀, nwgd(model)), η = as𝕀))
    P = TransformedLogDensity(t, prior)
    _WhaleProblem{_CRPrior,typeof(P)}(model, data, P)
end

function LogDensityProblems.logdensity_and_gradient(p::_WhaleProblem, x)
    @unpack model, prior, data = p
    v, J = transform_and_logjac(prior.transformation, x)
    r = ConstantRates(v...)
    π = prior.log_density_function(prior.transformation(x))
    ℓ = logpdf(model(r), data)

    ∇ℓ = gradient(model, r, data, x, prior.transformation)
    ∇π = gradient(prior.log_density_function, x, prior.transformation)
    ∇J = gradjac(prior.transformation, x)
    @show ∇ℓ, ∇π, ∇J
    return ℓ + π + J, ∇ℓ .+ ∇π .+ ∇J
end

LogDensityProblems.capabilities(::Type{<:_WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(wp::_WhaleProblem{_CRPrior}) = 3 + nwgd(wp.model)


# test it
wm = WhaleModel(Whale.extree)
Whale.addwgd!(wm, 5, 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm))

_prior = _CRPrior(wm, MvNormal(ones(2)), Beta(3,1), Beta())
_problem = _WhaleProblem(wm, D, _prior)
@show logdensity_and_gradient(_problem, zeros(4))

progress = LogProgressReport(step_interval=100, time_interval_s=10)
@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 2000,
    reporter = progress)
    # initialization = (ϵ=0.5, ),
    # warmup_stages = fixed_stepsize_warmup_stages())

posterior = transform.(problem.prior.transformation, results.chain)
l = [x.λ for x in posterior]
m = [x.μ for x in posterior]
q = [x.q[1] for x in posterior]
e = [x.η for x in posterior]

plot(plot(l), plot(m), plot(q), plot(e), legend=false, grid=false)

# now take time to design a robust and flexible API
