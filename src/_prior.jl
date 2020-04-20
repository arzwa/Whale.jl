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
    πr::MvLogNormal = MvLogNormal(ones(2))
    πq::Beta = Beta()
    πη::Union{Beta,Normal} = Beta(1,3)
end

function logpdf(prior::CRPrior, θ::T) where T
    @unpack λ, μ, η = θ.params
    q = hasfield(T, :q) ? θ.q : Float64[]
    @unpack πr, πη, πq = prior
    logpdf(πη, η) + logpdf(πr, [λ, μ]) + sum(logpdf.(πq, q))
end

"""
    IRPrior

Bivariate independent rates prior with fixed covariance matrix.
"""
@with_kw struct IRPrior <: Prior
    πr::MvNormal = MvNormal([10.,10.])
    πq::Beta = Beta()
    πη::Union{Beta,Normal} = Beta(3,1)
end

function logpdf(prior::IRPrior, θ)
    @unpack πr, πq, πη, πE = prior
    @unpack λ, μ, η = θ
    q = hasfield(T, :q) ? θ.q : Float64[]
    p  = typeof(πη)<:Normal && πη.σ == zero(πη.σ) ? 0. : logpdf(πη, η)
    p += sum(logpdf(πr, r)) + sum(logpdf.(πq, q))
    isfinite(p) ? p : -Inf
end

"""
    IWIRPrior

Bivariate independent rates prior with Inverse-Wishart prior on
the covariance matrix.
"""
@with_kw struct IWIRPrior <: Prior
    Ψ ::Matrix{Float64} = [10. 0.; 0. 10.]
    πr::MvNormal = MvNormal([10.,10.])
    πq::Beta = Beta()
    πη::Union{Beta,Normal} = Beta(3,1)
    πE::Union{Nothing,Tuple{Normal,Vector{Float64}}} = nothing
end

function logpdf(prior::IWIRPrior, θ::T) where T
    @unpack Ψ, πr, πq, πη, πE = prior
    @unpack λ, μ, η = θ.params
    q = hasfield(T, :q) ? θ.q : Float64[]
    r = log.(permutedims([λ μ]))
    X₀ = r[:,1]
    Y = r[:,2:end] .- X₀  # centered rate vectors prior ~ MvNormal(0, Ψ)
    # Y*Y' is the sample covariance matrix
    p = logpdf_pics(Ψ, Y, 3) + logpdf_evalue(πE, r)
    p += typeof(πη)<:Normal && πη.σ == zero(πη.σ) ? 0. : logpdf(πη, η)
    p += logpdf(πr, X₀) + sum(logpdf.(πq, q))
    isfinite(p) ? p : -Inf
end

# Following Lartillot & Poujol
logpdf_pics(Ψ, Y, ν) =
    (ν/2)*log(det(Ψ)) - ((ν+size(Y)[2])/2)*log(det(Ψ + Y*Y'))

logpdf_evalue(d, r) = isnothing(d) ? 0. :
    sum(logpdf.(d[1], @. exp(d[2]*(exp.(r[1,:])-exp.(r[2,:])))))

"""
    LKJCorr

**Unnormalized** LKJ correlation matrix prior.
"""
struct LKJCorr{T}
    η::T
end

# unnormalized logpdf
# https://github.com/pymc-devs/pymc3/blob/433c693104e32b267e8ccdbab6affddf37b83b65/pymc3/distributions/multivariate.py#L1398
logpdf(d::LKJCorr, R::Matrix) = (d.η - one(d.η))*log(det(R))

# Introduces a hyperparameter...
# """
#     LKJIRPrior
#
# Independent rates prior with LKJCorr prior on the covariance structure.
# https://mc-stan.org/docs/2_22/stan-users-guide/multivariate-hierarchical-priors-section.html
# """
# @with_kw struct LKJIRPrior <: Prior
#     πR::LKJCorr
#     πτ::UnivariateDistribution
#     πr::MvNormal = MvNormal([10.,10.])
#     πq::Beta = Beta()
#     πη::Union{Beta,Normal} = Beta(3,1)
# end
#
# function logpdf(prior::LKJIRPrior, θ)
#     @unpack πR, πτ, πr, πq, πη = prior
#     @unpack U, τ, r, q, η = θ
#     R = [1. U[1,2]; U[1,2] 1.]  # U'U
#     Σ = [τ^2 U[1,2]*τ^2; U[1,2]*τ^2 τ^2]  # τI*U'U*τI
#     if !isposdef(Σ)
#         return -Inf
#     end
#     p = logpdf(πR, R) + logpdf(πτ, τ)
#     p += sum(logpdf(MvNormal(r[:,1], Σ), r[:,2:end]))
#     p += typeof(πη)<:Normal && πη.σ == zero(πη.σ) ? 0. : logpdf(πη, η)
#     p += logpdf(πr, r[:,1]) + sum(logpdf.(πq, q))
#     isfinite(p) ? p : -Inf
# end
#
# RatesModel(prior::LKJIRPrior) = BranchRates
# trans(::LKJIRPrior, model::WhaleModel) = as((
#     τ=asℝ₊, U=CorrCholeskyFactor(2),
#     r=as(Array, asℝ, 2, nnonwgd(model)),
#     q=as(Array, as𝕀, nwgd(model)), η=as𝕀))
