# ALE agorithm for DLWGD model
# NOTE: This is not compatible with ReverseDiff, but it is not too hard to make
# it so, using the ℓ matrices in the CCD as template and computing the slices
# in the logpdf routine -- at the expense of more allocations (see
# `revdiffsafe` branch)

# utilities
# NOTE: in these utils λ and μ should be on 'rate' scale
getϵ(n::ModelNode) = n[end,2]
getϕ(n::ModelNode) = n[end,3]
getΔ(n::ModelNode, i::Int) = n[i,1]
getϵ(n::ModelNode, i::Int) = n[i,2]
getϕ(n::ModelNode, i::Int) = n[i,3]
getψ(n::ModelNode, i::Int) = n[i,4]
ℓhood(ℓ) = isfinite(ℓ) ? ℓ : -Inf

# # transition probability under the linear BDP 1 → 2
# # NOTE: this should account for extinction down the tree no?? -> we use ψ now
# function pdup(λ, μ, t)
#     α = getα(λ, μ, t)
#     β = (λ/μ)*α
#     return (one(α) - α)*(one(α) - β)*β
# end

Distributions.loglikelihood(model::WhaleModel, x) = logpdf(model, x)

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

Distributions.logpdf(m::ModelArray, xs::Vector{<:CCD}) = 
    sum(tmap(i->logpdf(m.models[i], xs[i]), 1:length(xs)))

Distributions.loglikelihood(m, x) = logpdf(m, x)

function whale!(n::ModelNode{T}, ℓ, x, wm) where T
    iswgd(n)  && return whalewgd!(n, ℓ, x, wm)
    isroot(n) && return whaleroot!(n, ℓ, x, wm)
    e = id(n)
    θ = getθ(wm.rates, n)
    ℓ[e] .= zero(T)
    for c in x.compat[e]
        γ = x[c]
        j = x.index[c,e]
        leaf = isleaf(γ)
        if leaf && isleaf(n)
            ℓ[e][j,1] = one(T)
        elseif !isleaf(n)
            p = Πspeciation(x, γ, ℓ, n) + Πloss(x, γ, ℓ, n)
            ℓ[e][j,1] += p
        end
        within_branch!(n, γ, ℓ, x, e, j, leaf) 
    end
end

function whalewgd!(n::ModelNode{T}, ℓ, x, wm) where T
    e = id(n)
    ℓ[e] .= zero(T)
    @unpack q, λ, μ = getθ(wm.rates, n)
    for c in x.compat[e]
        γ = x[c]
        j = x.index[c,e]
        p = zero(T)
        leaf = isleaf(γ)
        if !leaf
            p += Πwgdretention(x, γ, ℓ, n, q)
        end
        p += Πwgdloss(x, γ, ℓ, n, q)
        ℓ[e][j,1] = p
        within_branch!(n, γ, ℓ, x, e, j, leaf) 
    end
end

@inline function within_branch!(n, γ, ℓ, x, e, j, leaf)
    for i=2:length(n)  # iterate over slices
        @inbounds ℓ[e][j,i] += getϕ(n, i)*ℓ[e][j,i-1]
        if !leaf
            @inbounds ℓ[e][j,i] += Πduplication(x, γ, ℓ, n, i)
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
        p1 = Πloss(x, γ, ℓ, n)
        p2 = zero(p1)
        if !leaf
            p1 += Πspeciation(x, γ, ℓ, n)
            p2 += Πroot(x, γ, ℓ, n, η)
        end
        @inbounds ℓ[e][γ.id,1] = p1 * η_ + p2
    end
    ℓ[e][end,1] *= η
end

@inline function Πroot(x, γ, ℓ, n, η)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits  # speciation
        @inbounds p += t.p * getl(x, ℓ, e, t.γ1, 1) * getl(x, ℓ, e, t.γ2, 1)
    end
    return p*(one(p) -η)*(one(p) -(one(p) -η) * getϵ(n))
end

@inline function Πspeciation(x, γ, ℓ, n)
    f = id(n[1])
    g = id(n[2])
    p = zero(eltype(ℓ[f]))
    for t in γ.splits
        @inbounds p += t.p * (
              getl(x, ℓ, f, t.γ1) * getl(x, ℓ, g, t.γ2) +
              getl(x, ℓ, g, t.γ1) * getl(x, ℓ, f, t.γ2))
    end
    return p
end

@inline function Πloss(x, γ, ℓ, n)
    f = id(n[1])
    g = id(n[2])
    @inbounds getl(x, ℓ, f, γ.id)*getϵ(n[2]) + getl(x, ℓ, g, γ.id)*getϵ(n[1])
end

@inline function Πduplication(x, γ, ℓ, n, i)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits
        @inbounds p += t.p * getl(x, ℓ, e, t.γ1, i-1) * getl(x, ℓ, e, t.γ2, i-1)
    end
    return getψ(n, i) * p
end

@inline function Πwgdretention(x, γ, ℓ, n, q)
    f = id(n[1])
    p = zero(eltype(ℓ[f]))
    for t in γ.splits
        @inbounds p += t.p * getl(x, ℓ, f, t.γ1) * getl(x, ℓ, f, t.γ2)
    end
    return p * q
end

@inline function Πwgdloss(x, γ, ℓ, n, q)
    f = first(children(n))
    return (one(q) - q)*getl(x, ℓ, id(f), γ.id) + 2q*getϵ(f)*getl(x, ℓ, id(f), γ.id)
end
