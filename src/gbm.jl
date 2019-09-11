"""
    GeometricBrownianMotion{T<:Real}
    GBM(t::SlicedTree, r::T, v::T) where {T<:Real}

Distribution induced by a Geometric Brownian Motion (GBM) over the SlicedTree.
`r` is the value at the root of the tree, `ν` is the variance (autocorrelation
strength).

The log density for the GBM distribution is computed with an implementation of
the GBM prior on rates based on Ziheng Yang's MCMCTree, described in [Rannala &
Yang (2007)](https://academic.oup.com/sysbio/article/56/3/453/1657118). This
uses the approach whereby rates are defined for midpoints of branches, and where
a correction is performed to ensure that the correlation is proper (in contrast
with Thorne et al. 1998).
"""
struct GeometricBrownianMotion{T<:Real,Ψ<:Arboreal} <:
        ContinuousMultivariateDistribution
    t::Ψ
    r::T  # rate at root
    ν::T  # autocorrelation strength
end

const GBM = GeometricBrownianMotion
GBM(t::Ψ, r::Real, v::Real) where Ψ<:Arboreal = GBM(t, promote(r, v)...)

# Base extensions
Base.length(d::GBM) = nrates(d.t)
Base.convert(::Type{GBM{T}}, s::SlicedTree, r::S,
    v::S) where {T<:Real,S<:Real} = GBM(s, convert(T, r), convert(T, v))
Base.convert(::Type{GBM{T}}, d::GBM{S}) where {T<:Real,S<:Real} = GBM(d.t, r, v)

# Distributions interface
function Distributions.insupport(d::GBM, x::AbstractVector{T}) where {T<:Real}
    for i=1:length(x)
        @inbounds 0.0 < x[i] < Inf ? continue : (return false)
    end
    true
end

Distributions.assertinsupport(::Type{D}, m::AbstractVector) where {D<:GBM} =
    @assert insupport(D, m) "[GBM] rates should be positive"

Distributions.sampler(d::GBM) = d

function Distributions._rand!(rng::AbstractRNG, d::GBM, r::AbstractVector)
    s = d.t
    r[findroot(s)] = d.r
    function walk(n::Int64)
        if !isroot(s.tree, n) && !(haskey(s.qindex, n))
            p = non_wgd_parent(s, n)
            t = distance(s.tree, n, p)
            r[s.rindex[n]] = exp(rand(Normal(log(r[p]) - d.ν^2*t/2, √t*d.ν)))
        end
        isleaf(s.tree, n) ? (return) : [walk(c) for c in childnodes(s, n)]
    end
    walk(findroot(s))
    return float(r)
end

function Distributions._logpdf(d::GBM, x::AbstractVector{T}) where T<:Real
    if !insupport(d, x)
        return -Inf
    end
    s = d.t
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
