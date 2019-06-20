"""
    $(TYPEDEF)
"""
struct GeometricBrownianMotion{T<:Real} <: ContinuousMultivariateDistribution
    T::SlicedTree
    r::T  # rate at root
    ν::T  # autocorrelation strength

    GeometricBrownianMotion{T}(s, r::T, v::T) where T = new(s, r, v)
end

const GBM = GeometricBrownianMotion

Base.length(d::GBM) = nrates(d.T)

# should work but it doesn't!
function _rand!(d::GBM, r::AbstractVector{T}) where T<:Real
    s = d.T
    @show r[findroot(s)] = d.r
    function walk(n::Int64)
        if !isroot(s.tree, n) && !(haskey(s.qindex, n))
            p = non_wgd_parent(s, n)
            t = distance(s.tree, n, p)
            r[s.rindex[n]] = exp(rand(Normal(log(r[p]) - d.ν^2*t/2, √t*d.ν)))
        end
        isleaf(s.tree, n) ? (return) : [walk(c) for c in childnodes(s, n)]
    end
    walk(findroot(s))
    return r
end

"""
    logpdf(d::GBM, s::SlicedTree, x::Array{Real})

Compute the log density for the GBM prior on rates. Implementation of the
GBM prior on rates based on Ziheng Yang's MCMCTree. Described in Rannala & Yang
(2007) (syst. biol.). This uses the approach whereby rates are defined for
midpoints of branches, and where a correction is performed to ensure that the
correlation is proper (in contrast with Thorne et al. 1998). See Rannala & Yang
2007 for detailed information.
"""
function logpdf(d::GBM{T}, x::Vector{T}, ν=d.ν, r=d.r) where T<:Real
    s = d.T
    logp = -log(2π)/2.0*(2*ntaxa(s)-2)  # factor from the Normal (every branch).
    for n in preorder(s)
        (isleaf(s.tree, n) || haskey(s.qindex, n)) ? continue : nothing
        babies = non_wgd_children(s, n)  # should be non-wgd children!
        n == 1 ? ta = 0. : ta = distance(s.tree, parentnode(s.tree, n), n) / 2
        t1 = distance(s.tree, n, babies[1])/2
        t2 = distance(s.tree, n, babies[2])/2
        # determinant of the var-covar matrix Σ up to factor σ^2
        dett = t1*t2 + ta*(t1+t2)
        # correction terms for correlation given rate at ancestral b
        tinv0 = (ta + t2) / dett
        tinv1 = tinv2 = -ta/dett
        tinv3 = (ta + t1) / dett
        ra = x[s.rindex[n]]
        r1 = x[s.rindex[babies[1]]]
        r2 = x[s.rindex[babies[2]]]
        y1 = log(r1/ra) + (ta + t1)*d.ν^2/2  # η matrix
        y2 = log(r2/ra) + (ta + t2)*d.ν^2/2
        zz = (y1*y1*tinv0 + 2*y1*y2*tinv1 + y2*y2*tinv3)
        logp -= zz/(2*d.ν^2) + log(dett*d.ν^4)/2 + log(r1*r2);
        # power 4 is from determinant (which is computed up to the factor from
        # the variance) i.e. Σ = [ta+t1, ta ; ta, ta + t2] × ν^2, so the
        # determinant is: |Σ| = (ta + t1)ν^2 × (ta + t2)ν^2 - ta ν^2 × ta ν^2 =
        # ν^4[ta × (t1 + t2) + t1 × t2] =#
    end
    return logp
end

#=
abstract type RatePrior end

"""
    $(TYPEDEF)

Geometric brownian motion prior on rates
"""
struct GBMRatePrior{T<:Real} <: RatePrior
    ν::ContinuousUnivariateDistribution
    θ::ContinuousMultivariateDistribution
    q::Beta{T}
    η::Beta{T}
end

const HyperParams = Dict{Symbol,Real}

function rand(d::GBMRatePrior, s::SlicedTree, ν::Float64=-1., η::Float64=-1.)
    ν = ν < 0. ? rand(d.ν) : ν   # we often want to sample under fixed values
    η = η < 0. ? rand(d.η) : η   # we often want to sample under fixed values
    λ0, μ0 = rand(d.θ)
    λ = rand(GBM(ν, λ0), s)
    μ = rand(GBM(ν, μ0), s)
    q = rand(d.q, nwgd(s))
    return HyperParams(:ν=>ν), WhaleParams(λ, μ, q, η)
end
=#
