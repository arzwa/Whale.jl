# For convenience, I re-inserted the rate models in the Whale library.
function iswgd end
function wgdid end
function nonwgdchild end

abstract type Params{T} end

struct RatesModel{T,M<:Params{T},V}
    params::M
    fixed ::Tuple
    trans ::V
end

RatesModel(θ; fixed=()) = RatesModel(θ, fixed, gettrans(θ, fixed))

Base.eltype(m::RatesModel{T}) where T = T
Base.show(io::IO, m::RatesModel) = write(io,
    "RatesModel with $(m.fixed) fixed\n$(m.params)")

getθ(m::RatesModel, node) = getθ(m.params, node)
getp(m::P, n) where {T,P<:Params{T}} = hasfield(P, :p) &&
    length(m.p) > 0 && isleaf(n) ? m.p[id(n)] : 0.

# HACK: a little bit of metaprogramming to allow fixed parameters, necessary?
function gettrans(p::P, fixed) where P<:Params
    inner = join(["$k=$v," for (k,v) in pairs(trans(p)) if k ∉ fixed])
    expr  = Meta.parse("as(($inner))")
    eval(expr)
end

(m::RatesModel)(x::Vector) = m(m.trans(x))
function (m::RatesModel)(θ)
    θ′ = merge(θ, [k=>getfield(m.params, k) for k in m.fixed])
    RatesModel(m.params(θ′), m.fixed, m.trans)
end

Base.rand(m::M) where M<:RatesModel = m(m.trans(randn(dimension(m.trans))))

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

trans(m::ConstantDLWGD) = (
    λ=asℝ₊, μ=asℝ₊, η=as𝕀,
    q=as(Array, as𝕀, length(m.q)),
    p=as(Array, as𝕀, length(m.p)))

function (::ConstantDLWGD)(θ)
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
        (λ=exp(m.λ[id(n)]), μ=exp(m.μ[id(n)]), p=getp(m, n), η=m.η)
    end
end

trans(m::DLWGD) = (
    λ=as(Array, asℝ, length(m.λ)),
    μ=as(Array, asℝ, length(m.λ)),
    q=as(Array, as𝕀, length(m.q)),
    p=as(Array, as𝕀, length(m.p)),
    η=as𝕀)

(::DLWGD)(θ) = DLWGD(;λ=θ.λ, μ=θ.μ, q=θ.q, η=eltype(θ.λ)(θ.η), p=θ.p)

"""
    Critical{T,V}

Branch-specific duplication and loss rates, with WGD events. Assumes
a geometric distribution with parameter `η` on the number of lineages
at the root.
"""
@with_kw struct Critical{T,V} <: Params{T}
    λ::Vector{T}
    q::Vector{T} = Float64[]
    p::Vector{V} = Float64[]
    η::T = 0.66
end

function getθ(m::Critical, n)
    return if iswgd(n)
        c = nonwgdchild(n)
        (λ=m.λ[id(c)], μ=m.λ[id(c)], q=m.q[wgdid(n)])
    else
        (λ=m.λ[id(n)], μ=m.λ[id(n)], p=getp(m, n), η=m.η)
    end
end

trans(m::Critical) = (
    λ=as(Array, asℝ₊, length(m.λ)),
    q=as(Array, as𝕀, length(m.q)),
    p=as(Array, as𝕀, length(m.p)),
    η=as𝕀)

(::Critical)(θ) = Critical(;λ=θ.λ, q=θ.q, η=eltype(θ.λ)(θ.η), p=θ.p)
