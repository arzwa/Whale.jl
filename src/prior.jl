"""
    Prior

A prior subtype implements a logpdf function, a ratesmodel and a transformation.
"""
abstract type Prior end

"""
    CRPrior

Prior for constant rates model (i.e. one duplication rate and one loss rate
for the entire tree). Supports arbitrary, but fixed number of WGDs.
"""
@with_kw struct CRPrior <: Prior
    πr::MvNormal = MvNormal(ones(2))
    πq::Beta = Beta()
    πη::Beta = Beta(1,3)
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
    Ψ ::Matrix{Float64} = [10. 0.; 0. 10.]
    πr::MvNormal = MvNormal([10.,10.])
    πq::Beta = Beta()
    πη::Beta = Beta(3,1)
    πE::Union{Nothing,Tuple{Normal,Vector{Float64}}} = nothing
end

function Base.rand(prior::IRPrior, wm::WhaleModel)
    @unpack Ψ, πr, πq, πη, πE = prior
    Σ = rand(InverseWishart(3, Ψ))
    v = rand(πr)
    r = exp.(rand(MvNormal(v, Σ), nnonwgd(wm)))
    q = rand(πq, nwgd(wm))
    η = rand(πη)
    BranchRates(r=r, q=q, η=η)
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
logpdf_evalue(d, r) = isnothing(d) ? 0. :
    sum(logpdf.(d[1], @. exp(d[2]*(r[1,:]-r[2,:]))))

RatesModel(prior::IRPrior) = BranchRates
trans(::IRPrior, model::WhaleModel) = as((r=as(Array, asℝ₊, 2, nnonwgd(model)),
        q=as(Array, as𝕀, nwgd(model)), η=as𝕀))
