# The RatesModel is obsolete if we no longer use DynamicHMC
function iswgd end
function wgdid end
function nonwgdchild end

abstract type Params{T} end

function (m::Params)(θ)
    x = NamedTuple([k=>getfield(m, k) for k in propertynames(m)])
    return updated(m, merge(x, θ))
end

# sampling probability
getp(m, n) = length(m.p) > 0 && isleaf(n) ? m.p[id(n)] : 0.

"""
    ConstantDLWGD{T,V}

Constant rates of duplication and loss, with WGD nodes. Assumes a
geometric distribution with parameter `η` on the number of lineages
at the root.
"""
@with_kw struct ConstantDLWGD{T,V} <: Params{T}
    λ::T
    μ::T
    q::Vector{T} = Float64[]
    p::Vector{V} = Float64[]
    η::T = 0.66
end

getθ(m::ConstantDLWGD, n) = (
    λ=m.λ, μ=m.μ, η=m.η, p=getp(m, n),
    q=iswgd(n) ? m.q[wgdid(n)] : NaN)

function updated(::ConstantDLWGD, θ)
    T = eltype(θ.q)  # XXX q as reference, should be a promotion?
    ConstantDLWGD(; λ=T(θ.λ), μ=T(θ.μ), q=θ.q, η=T(θ.η), p=θ.p)
end

"""
    DLWGD{T,V}

Branch-specific duplication and loss rates, with WGD events. Assumes
a geometric distribution with parameter `η` on the number of lineages
at the root.
"""
@with_kw struct DLWGD{T,V} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T} = Float64[]
    p::Vector{V} = Float64[]
    η::T = 0.66
end

function getθ(m::DLWGD, n)
    return if iswgd(n)
        c = nonwgdchild(n)
        (λ=exp(m.λ[id(c)]), μ=exp(m.μ[id(c)]), q=m.q[wgdid(n)])
    else
        id(n) > length(m.λ) ?
            (λ=NaN, μ=NaN, p=getp(m, n), η=m.η) :
            (λ=exp(m.λ[id(n)]), μ=exp(m.μ[id(n)]), p=getp(m, n), η=m.η)
    end
end

updated(::DLWGD, θ) = DLWGD(;λ=θ.λ, μ=θ.μ, q=θ.q, η=eltype(θ.λ)(θ.η), p=θ.p)
