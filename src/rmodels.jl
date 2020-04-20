module RatesModels

# The RatesModel provides an interface between parameter values and a general
# phylogenetic model, so that we can use the same algorithm routines
# irrespective of how parameters are shared across nodes/branches/families,...
# IDEA: define a WGD model as a wrapper around a RatesModel. No WGD models could be traits
# XXX: should we have Gamma mixtures baked in? Or should that be another
# wrapper around the ratesmodel?
# There is some room for metaprogramming hacks here

abstract type Params{T} end

struct RatesModel{T,M<:Params{T},V}
    params::M
    fixed ::Tuple
    trans ::V
end

Base.eltype(m::RatesModel{T}) where T = T
RatesModel(θ; fixed=()) = RatesModel(θ, fixed, gettrans(θ, fixed))
Base.show(io::IO, m::RatesModel) = write(io,
    "RatesModel with $(m.fixed) fixed\n$(m.params)")
getθ(m::RatesModel, node) = getθ(m.params, node)

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
    ConstantDLG{T}

Simple constant rates duplication-loss and gain model. All nodes of
the tree are associated with the same parameters (duplication rate λ,
loss rate μ, gain rate κ). This assumes a shifted geometric distribution
on the family size at the root with mean 1/η.
"""
@with_kw struct ConstantDLG{T} <: Params{T}
    λ::T
    μ::T
    κ::T = 0.
    η::T = 0.66
end

getθ(m::ConstantDLG, node) = m
trans(::ConstantDLG) = (λ=asℝ₊, μ=asℝ₊, κ=asℝ₊, η=as𝕀)
# the zip is a bit slow...
(::ConstantDLG)(θ) = ConstantDLG(; zip(keys(θ), promote(θ...))...)

@with_kw struct ConstantDLGWGD{T} <: Params{T}
    λ::T
    μ::T
    q::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

getθ(m::ConstantDLGWGD, node) = iswgd(node) ?
    (λ=m.λ, μ=m.μ, q=m.q[wgdid(node)], κ=m.κ) : (λ=m.λ, μ=m.μ, κ=m.κ, η=m.η)
trans(m::ConstantDLGWGD) = (
    λ=asℝ₊, μ=asℝ₊,
    q=as(Array, as𝕀, length(m.q)),
    κ=asℝ₊, η=as𝕀)
function (::ConstantDLGWGD)(θ)
    T = eltype(θ.q)
    ConstantDLGWGD(;λ=T(θ.λ), μ=T(θ.μ), q=θ.q, κ=T(θ.κ), η=T(θ.η))
end

@with_kw struct DLG{T} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

getθ(m::DLG, node) = (λ=m.λ[id(node)], μ=m.μ[id(node)], κ=m.κ, η=m.η)
trans(m::DLG) = (λ=as(Array, asℝ₊, length(m.λ)),
    μ=as(Array, asℝ₊, length(m.λ)), κ=asℝ₊, η=as𝕀)
(::DLG)(θ) = DLG(; λ=θ.λ, μ=θ.μ, κ=eltype(θ.λ)(θ.κ), η=eltype(θ.λ)(θ.η))

@with_kw struct DLGWGD{T} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}
    κ::T = 0.
    η::T = 0.66
end

# TODO: find a proper way to infer `wgdid`
function getθ(m::DLGWGD, node)
    return if iswgd(node)
        c = nonwgdchild(node)
        (λ=m.λ[id(c)], μ=m.μ[id(c)], q=m.q[wgdid(node)], κ=m.κ)
    else
        (λ=m.λ[id(node)], μ=m.μ[id(node)], κ=m.κ, η=m.η)
    end
end

trans(m::DLGWGD) = (
    λ=as(Array, asℝ₊, length(m.λ)),
    μ=as(Array, asℝ₊, length(m.λ)),
    q=as(Array, as𝕀, length(m.q)),
    κ=asℝ₊, η=as𝕀)

(::DLGWGD)(θ) = DLGWGD(;
    λ=θ.λ, μ=θ.μ, q=θ.q, κ=eltype(θ.λ)(θ.κ), η=eltype(θ.λ)(θ.η))

end
