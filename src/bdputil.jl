# define all in terms of α and β!
const ΛMATOL = 1e-6
const MARGIN = 1e-3
getα(λ, μ, t) = isapprox(λ, μ, atol=ΛMATOL) ?
    λ*t/(one(t) + λ*t) : μ*(exp(t*(λ-μ)) - one(t))/(λ*exp(t*(λ-μ)) - μ)
_ϵ(α, β, ϵ) = (α + (one(α)-α-β)*ϵ)/(one(α)-β*ϵ)
_ϕ(α, β, ϵ) = (one(α)-α)*(one(α)-β)/(one(α)-β*ϵ)^2
_ψ(α, β, ϵ) = (one(α)-α)*(one(α)-β)*β/(one(α)-β*ϵ)^3

stir(n) = n*log(n) - n + log(2π*n)/2
_bin(n, k) =
    k == 0 ? 1. :
    n <= 60 ? float(binomial(n, k)) :
    exp(stir(n)-stir(k)-stir(n - k))
_ξ(i, j, k, α, β) = _bin(i, k)*_bin(i+j-k-1,i-1)*α^(i-k)*β^(j-k)*(one(α)-α-β)^k
probify(p) =  p > one(p)  && isapprox(p,  one(p), atol=MARGIN) ?
    one(p)  : p < zero(p) && isapprox(p, zero(p), atol=MARGIN) ?
    zero(p) : !(zero(p) <= p <= one(p)) ? NaN : p

# transition probability for the linear BDP
function tp(a, b, t, λ, μ)
    a == b == zero(a) && return one(λ)
    α = getα(λ, μ, t)
    β = (λ/μ)*α
    return sum([_ξ(a, b, k, α, β) for k=0:min(a,b)])
end

"""
    LinearBDP{T}

A linear birth-death process with rate parameters λ (birth) and μ (death).
"""
struct LinearBDP{T}
    λ::T
    μ::T
    t::T
    ρ::T
    LinearBDP(λ::T, μ::T, t::T) where T = new{T}(λ, μ, t, exp((μ - λ)*t))
end

tp(p::LinearBDP, a, b) = tp(a, b, p.t, p.λ, p.μ)

"""
    pgf(::LinearBDP, s)

The probability generating function for the linear BDP.

Example (probability of extinction given a Geometric distribution on
the number of initial particles):
```
p = LinearBDP(0.2, 0.3, 1.)
d = Geometric(0.66)
pgf(d, pgf(p, 0.))
```
"""
function pgf(p::LinearBDP{T}, s) where T
    @unpack ρ, λ, μ, t = p
    # what about the critical case?
    isapprox(λ, μ, atol=ΛMATOL) ?
        (one(T) - (λ*t - one(T))*(s - one(T)))/(one(T) - λ*t*(s - one(T))) :
        (ρ*(λ*s - μ) - μ*(s-one(T)))/(ρ*(λ*s - μ) - λ*(s-one(T)))
end

pgf(d::Geometric, s) = geompgf(d.p, s)
geompgf(p, s) = p*s/(one(p) - (one(p) - p)*s)

# struct WGDBernouilli{T}
#     q::T
# end
# pgf(d::WGDBernouilli, s) = wgdpgf(d.q, s)
wgdpgf(q, s) = s*(one(q) - q + s*q)

"""
    treepgf(tree, xs::Vector)

Evaluate the joint pgf for the leaves of a tree structure.

!!! note
    Assumes the NewickTree interface for the nodes (i.e. id(⋅), distance(⋅) and
    children(⋅) should be defined). **It is also assumed** that leaf ids are 1,
    2, ..., l
"""
function treepgf(tree, xs::Vector{T}) where T
    function walk(n)
        θ = Whale.getθ(tree.rates, n)
        f = isroot(n) ? Geometric(θ.η) : LinearBDP(θ.λ, θ.μ, distance(n))
        isleaf(n) && return pgf(f, xs[id(n)])
        x = prod([walk(c) for c in children(n)])
        return iswgd(n) ? pgf(f, wgdpgf(θ.q, x)) : pgf(f, x)
    end
    walk(getroot(tree))
end

function extinctionp(tree)  # joint pgf evaluated at 0, 0, 0, ...
    function walk(n)
        θ = Whale.getθ(tree.rates, n)
        f = isroot(n) ? Geometric(θ.η) : LinearBDP(θ.λ, θ.μ, distance(n))
        isleaf(n) && return pgf(f, 0.)
        x = prod([walk(c) for c in children(n)])
        return iswgd(n) ? pgf(f, wgdpgf(θ.q, x)) : pgf(f, x)
    end
    walk(getroot(tree))
end

# This function evaluates the pgf for all binary arguments, i.e. f(0,0,0,...,0)
# f(1,0,0,...,0), f(0,1,0,...,0), ... f(1,1,1,...,1).
function treepgf_allbinary(tree)
    function walk(n)
        θ = Whale.getθ(tree.rates, n)
        f = isroot(n) ? Geometric(θ.η) : LinearBDP(θ.λ, θ.μ, distance(n))
        isleaf(n) && return [pgf(f, 0.), pgf(f, 1.)]
        down = [walk(c) for c in children(n)]
        xs = vec(prod.(Iterators.product(down...)))
        wgd = iswgd(n)
        return [wgd ? pgf(f, wgdpgf(θ.q, x)) : pgf(f, x) for x in xs]
    end
    walk(getroot(tree))
end

# This function gets the sign of each term for if we want to do inclusion-
# exclusion to obtain the probability of being extinct nowhere.
function treepgf_allbinary_sign(tree)
    function walk(n)
        isleaf(n) && return [1., -1.]
        down = [walk(c) for c in children(n)]
        xs = vec(prod.(Iterators.product(down...)))
        return xs
    end
    sign.(walk(getroot(tree)))
end

# Get all binary trees in the order of the above functions
function allbinarytrees(tree)
    function walk(n)
        isleaf(n) && return [0., 1.]
        down = [walk(c) for c in children(n)]
        xs = vcat(collect(Iterators.product(down...))...)
        return [x for x in xs]
    end
    walk(getroot(tree))
end

# Annoyingly, the order is not the one we'd need for inclusion - exclusion...
# (which would be 0,0,...,0; 1,0,...,0; 0,1,...,0; ... ... ; 1,1,...,1), hence
# the `treepgf_allbinary_sign` function.

# In principle possible to use pgf for likelihood?
# https://www.jstor.org/stable/pdf/2245060.pdf?refreqid=excelsior%3A759e9c0a878328f1a759b9f4b7dade56
# describes a numerical inversion algorithm that could be used to obtain the
# probability of a phylogenetic profile from the pgf (if the latter is
# correct)?

# The pgf methods do not work yet with WGD
