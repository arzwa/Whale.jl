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

mergetup(t1, t2) = tuple(union(t1, t2)...)

# ------------------------------------------------------------------------------
@with_kw struct ConstantDLWGD{T,V} <: Params{T}
    λ::T
    μ::T
    q::Vector{T}
    p::Vector{V}
    η::T = 0.66
end

getθ(m::ConstantDLWGD, node) = iswgd(node) ?
    (λ=m.λ, μ=m.μ, q=m.q[wgdid(node)]) : m
trans(m::ConstantDLWGD) = (
    λ=asℝ₊, μ=asℝ₊, η=as𝕀,
    q=as(Array, as𝕀, length(m.q)),
    p=as(Array, as𝕀, length(m.p)))
function (::ConstantDLWGD)(θ)
    T = eltype(θ.q)
    ConstantDLWGD(; λ=T(θ.λ), μ=T(θ.μ), q=θ.q, η=T(θ.η), p=θ.p)
end

# ------------------------------------------------------------------------------
@with_kw struct DLWGD{T,V} <: Params{T}
    λ::Vector{T}
    μ::Vector{T}
    q::Vector{T}
    p::Vector{V}
    η::T = 0.66
end

function getθ(m::DLWGD, node)
    return if iswgd(node)
        c = nonwgdchild(node)
        (λ=exp(m.λ[id(c)]), μ=exp(m.μ[id(c)]), q=m.q[wgdid(node)])
    else
        (λ=exp(m.λ[id(node)]), μ=exp(m.μ[id(node)]), p=m.p, η=m.η)
    end
end

trans(m::DLWGD) = (
    λ=as(Array, asℝ, length(m.λ)),
    μ=as(Array, asℝ, length(m.λ)),
    q=as(Array, as𝕀, length(m.q)),
    p=as(Array, as𝕀, length(m.p)),
    η=as𝕀)

(::DLWGD)(θ) = DLWGD(;λ=θ.λ, μ=θ.μ, q=θ.q, η=eltype(θ.λ)(θ.η), p=θ.p)
