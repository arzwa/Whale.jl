# NOTE: currently this does not work (well) in the case where there would be
# hyperparameters that are sampled but do not end up in the RatesModel, however
# the RatesModel layer in between the prior and the model provides an opportunity
# to handle this.

"""
    Prior

A prior subtype implements a logpdf function, a ratesmodel and a transformation.
"""
abstract type Prior end

"""
    WhaleProblem

A generic Whale 'problem' interface ̀a la LogDensityProblems.jl. This holds
a TransformVariables transformation, prior, ratesmodel, data and WhaleModel.
This struct defines all DynamicHMC related functionalities and can be
constructed from a WhaleModel instance, data set and prior struct (the rationale
is that the prior struct full defines the problem).
"""
struct WhaleProblem{V<:Prior,R,T}
    data ::CCDArray
    model::WhaleModel
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

function gradient(wm::WhaleModel, r, data::CCDArray, t, x)
     mapreduce((ccd) -> gradient(wm, r, ccd, t, x), +, data)
end

function gradient(wm::WhaleModel, r, ccd::CCD, t, x)
    gradfun = (x) -> logpdf(wm(r(t(x))), ccd)
    ForwardDiff.gradient(gradfun, x)
end

function LogDensityProblems.logdensity_and_gradient(p::WhaleProblem, x)
    @unpack model, prior, data, trans, rates = p
    v, J = transform_and_logjac(trans, x)
    π = logpdf(prior, v)
    ℓ = logpdf(model(rates(v)), data)
    ∇ℓ = gradient(model, rates, data, trans, x)
    ∇π = gradient(prior, trans, x)
    ∇J = gradient(trans, x)
    # @show ∇ℓ, ∇π, ∇J
    return ℓ + π + J, ∇ℓ .+ ∇π .+ ∇J
end

LogDensityProblems.capabilities(::Type{<:WhaleProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

LogDensityProblems.dimension(p::WhaleProblem) = dimension(p.trans)

"""
    CRPrior

Prior for constant rates model (i.e. one duplication rate and one loss rate
for the entire tree). Supports arbitrary, but fixed number of WGDs.
"""
struct CRPrior <: Prior
    πr::MvNormal
    πq::Beta
    πη::Beta
end

function logpdf(prior::CRPrior, θ)
    @unpack λ, μ, η, q = θ
    @unpack πr, πη, πq = prior
    logpdf(πη, η) + logpdf(πr, log.([λ, μ])) + sum(logpdf.(πq, q))
end

RatesModel(prior::CRPrior) = ConstantRates
trans(::CRPrior, model::WhaleModel) =
    as((λ=asℝ₊, μ=asℝ₊, q=as(Array, as𝕀, nwgd(model)), η=as𝕀))

"""
    IRPrior

Bivariate independent rates prior.
"""
@with_kw struct IRPrior <: Prior
    Ψ ::Matrix{Float64}
    πr::MvNormal = MvNormal([10.,10.])
    πq::Beta = Beta()
    πη::Beta = Beta(3,1)
    πE::Union{Nothing,Tuple{Normal,Vector{Float64}}} = nothing
end

function logpdf(prior::IRPrior, θ)
    @unpack Ψ, πr, πq, πη, πE = prior
    @unpack r, q, η = θ
    X₀ = log.(r[:,1])
    Y = log.(r[:,2:end]) .- X₀  # centered rate vectors prior ~ MvNormal(0, Ψ)
    p = logpdf_pics(Ψ, Y, 3) + logpdf_evalue(πE, r)
    p + logpdf(πη, η) + logpdf(πr, X₀) + sum(logpdf.(πq, q))
end

logpdf_pics(Ψ, Y, ν) = log(det(Ψ)) - ((ν+size(Y)[2])/2)*log(det(Ψ + Y*Y'))
logpdf_evalue(d, r) = isnothing(d) ? 0. : logpdf(d[1], exp.(d[2].*(r[1,:].-r[2,:])))

RatesModel(prior::IRPrior) = BranchRates
trans(::IRPrior, model::WhaleModel) = as((r=as(Array, asℝ₊, 2, nnonwgd(model)),
        q=as(Array, as𝕀, nwgd(model)), η=as𝕀))


# ConstantRates
wm = WhaleModel(Whale.extree)
Whale.addwgd!(wm, 5, 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm))

prior = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(wm, D, prior)
@show logdensity_and_gradient(problem, zeros(4))

# BranchRates
prior = IRPrior(Ψ=[1. 0.; 0. 1.])
problem = WhaleProblem(wm, D, prior)
@show logdensity_and_gradient(problem, zeros(36))

progress = LogProgressReport(step_interval=100, time_interval_s=10)
@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 2000,
    reporter = progress)
