# ALE agorithm for DLWGD model
# utilities
# NOTE: in these utils λ and μ should be on 'rate' scale
iscompatible(γ::Clade, n::ModelNode) = γ.species ⊆ n.data.clade
getϵ(n::ModelNode) = n[end,2]
getϕ(n::ModelNode) = n[end,3]
getΔ(n::ModelNode, i::Int) = n[i,1]
getϵ(n::ModelNode, i::Int) = n[i,2]
getϕ(n::ModelNode, i::Int) = n[i,3]
getψ(n::ModelNode, i::Int) = n[i,4]
ℓhood(ℓ) = isfinite(ℓ) ? ℓ : -Inf

# # transition probability under the linear BDP 1 → 2
# # NOTE: this should account for extinction down the tree no??
# function pdup(λ, μ, t)
#     α = getα(λ, μ, t)
#     β = (λ/μ)*α
#     return (one(α) - α)*(one(α) - β)*β
# end

# NOTE conditioning should be done outside the inner loop. If the normalizing
# factor is expensive we don't want to compute it `n` times. The difference is
# really minor when the conditioning factor is cheap (e.g. pbothsides)
"""
    logpdf!(model, ccd)

Compute the log-likelihood of the data (a single CCD or vector of CCDs) given
the parameterized model `model`. Note that this computes the unconditional
likelihood (i.e. not conditional on for instance non-extinction).
"""
logpdf!(wm::WhaleModel, x::CCD) = logpdf!(wm, x.ℓ, x)

@inline function logpdf!(wm::WhaleModel{T}, ℓ::Array, x::CCD{I,V}) where {T,I,V}
    for n in wm.order
        whale!(n, ℓ, x, wm)
    end
    L = ℓ[id(root(wm))][end,1]
    L > zero(L) ? log(L) : -Inf
end

function logpdf(wm::WhaleModel{T}, x::CCD) where T
    # using `similar` did not give a speedup?
    ℓ = [zeros(T, size(xᵢ)) for xᵢ in x.ℓ]
    logpdf!(wm, ℓ, x)
end

# threaded implementation with ordinary vectors
function logpdf!(wm::WhaleModel{T}, xs::Vector{<:CCD}) where T
    #acc = Atomic{T}(0)
    ℓ = Vector{T}(undef, length(xs))
    Threads.@threads for i in 1:length(xs)
        ℓ[i] = logpdf!(wm, xs[i])
        #atomic_add!(acc, ℓ)
    end
    # ℓhood(acc[] - length(xs)*c(wm))
    ℓhood(sum(ℓ) - length(xs)*condition(wm))
    # the atomic thing does not work with Dual types
end

function logpdf(wm::WhaleModel{T}, xs::Vector{<:CCD}) where T
    ℓ = Vector{T}(undef, length(xs))
    Threads.@threads for i in 1:length(xs)
        ℓ[i] = logpdf(wm, xs[i])
    end
    ℓhood(sum(ℓ) - length(xs)*condition(wm))
end

# mapreduce implementations of logpdf for use with DArrays
function logpdf!(wm::WhaleModel{T}, X::CCDArray) where T
    ℓ = mapreduce((x)->logpdf!(wm, x.ℓ, x), +, X)
    ℓhood(ℓ - (length(X))*condition(wm))
end

function logpdf(wm::WhaleModel{T}, X::CCDArray) where T
    ℓ = mapreduce((x)->logpdf(wm, x), +, X)::T
    ℓhood(ℓ - length(X)*condition(wm))
end

function whale!(n::ModelNode{T}, ℓ, x, wm) where T
    iswgd(n)  && return whalewgd!(n, ℓ, x, wm)
    isroot(n) && return whaleroot!(n, ℓ, x, wm)
    e = id(n)
    θ = getθ(wm.rates, n)
    ℓ[e] .= zero(T)
    for γ in x.clades
        !iscompatible(γ, n) && continue
        leaf = isleaf(γ)
        if leaf && isleaf(n)
            ℓ[e][γ.id,1] = n.data.leafℙ
        elseif !isleaf(n)
            p = Πspeciation(γ, ℓ, n) + Πloss(γ, ℓ, n)
            ℓ[e][γ.id,1] += p
        end
        for i=2:length(n)  # iterate over slices
            ℓ[e][γ.id,i] += getϕ(n, i)*ℓ[e][γ.id,i-1]
            if !leaf
                ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i, θ.λ, θ.μ)
            end
        end
    end
end

function whalewgd!(n::ModelNode{T}, ℓ, x, wm) where T
    e = id(n)
    ℓ[e] .= zero(T)
    @unpack q, λ, μ = getθ(wm.rates, n)
    for γ in x.clades
        !iscompatible(γ, n) && continue
        p = zero(T)
        leaf = isleaf(γ)
        if !leaf
            p += Πwgdretention(γ, ℓ, n, q)
        end
        p += Πwgdloss(γ, ℓ, n, q)
        ℓ[e][γ.id,1] = p
        for i=2:length(n)  # iterate over slices
            ℓ[e][γ.id,i] += getϕ(n, i)*ℓ[e][γ.id,i-1]
            if !leaf
                ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i, λ, μ)
            end
        end
    end
end

function whaleroot!(n::ModelNode{T}, ℓ, x, wm) where T
    @unpack η = getθ(wm.rates, n)
    e = id(n)
    ℓ[e] .= zero(T)
    η_ = one(η)/(one(η) - (one(η) - η) * getϵ(n))^2
    for γ in x.clades
        leaf = isleaf(γ)
        p1 = Πloss(γ, ℓ, n)
        p2 = zero(p1)
        if !leaf
            p1 += Πspeciation(γ, ℓ, n)
            p2 += Πroot(γ, ℓ, n, η)
        end
        ℓ[e][γ.id,1] = p1 * η_ + p2
    end
    ℓ[e][end,1] *= η
end

@inline function Πroot(γ, ℓ, n, η)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits  # speciation
        @inbounds p += t.p * ℓ[e][t.γ1,1] * ℓ[e][t.γ2,1]
    end
    return p*(one(p) -η)*(one(p) -(one(p) -η) * getϵ(n))
end

@inline function Πspeciation(γ, ℓ, n)
    f = id(n[1])
    g = id(n[2])
    p = zero(eltype(ℓ[f]))
    for t in γ.splits
        @inbounds p += t.p * (
            ℓ[f][t.γ1,end] * ℓ[g][t.γ2,end] +
            ℓ[g][t.γ1,end] * ℓ[f][t.γ2,end])
    end
    return p
end

@inline function Πloss(γ, ℓ, n)
    f = id(n[1])
    g = id(n[2])
    @inbounds ℓ[f][γ.id,end]*getϵ(n[2]) + ℓ[g][γ.id,end]*getϵ(n[1])
end

# NOTE: λ and μ on 'rate' scale
@inline function Πduplication(γ, ℓ, n, i, λ, μ)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits
        @inbounds p += t.p * ℓ[e][t.γ1,i-1] * ℓ[e][t.γ2,i-1]
    end
    # return p * pdup(λ, μ, n[i,1])
    # return λ*getΔ(n, i)*p
    return getψ(n, i) * p
end

@inline function Πwgdretention(γ, ℓ, n, q)
    f = id(n[1])
    p = zero(eltype(ℓ[f]))
    for t in γ.splits
        @inbounds p += t.p * ℓ[f][t.γ1,end] * ℓ[f][t.γ2,end]
    end
    return p * q
end

@inline function Πwgdloss(γ, ℓ, n, q)
    f = first(children(n))
    @inbounds (one(q) - q)*ℓ[id(f)][γ.id,end] + 2q*getϵ(f)*ℓ[id(f)][γ.id,end]
end
