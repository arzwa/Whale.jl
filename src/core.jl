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
    L = ℓ[id(root(wm))][1,end]
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

function logpdf(mm::MixtureModel{VF,VS,<:WhaleModel{T}}, xs::Vector{<:CCD}) where {VF,VS,T}
    K = length(mm.components)
    ℓ = Matrix{T}(undef, length(xs), K)
    for j in 1:K
        Threads.@threads for i in 1:length(xs)
            ℓ[i,j] = logpdf(mm.components[j], xs[i])
        end
        ℓ[:,j] .+= log(mm.prior.p[j]) .- condition(mm.components[j])
    end
    ℓhood(mapreduce(logsumexp, +, eachrow(ℓ)))::T
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
            ℓ[e][1,j] = n.data.leafℙ
        elseif !isleaf(n)
            p = Πspeciation(x, γ, ℓ, n) + Πloss(x, γ, ℓ, n)
            ℓ[e][1,j] += p
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
        ℓ[e][1,j] = p
        within_branch!(n, γ, ℓ, x, e, j, leaf) 
    end
end

@inline function within_branch!(n, γ, ℓ, x, e, j, leaf)
    for i=2:length(n)  # iterate over slices
        @inbounds ℓ[e][i,j] += getϕ(n, i)*ℓ[e][i-1,j]
        if !leaf
            @inbounds ℓ[e][i,j] += Πduplication(x, γ, ℓ, n, i)
        end
    end
end

function whaleroot!(n::ModelNode{T}, ℓ, x, wm) where T
    @unpack η = getθ(wm.rates, n)
    e = id(n)
    ℓ[e] .= zero(T)
    ϵ = getϵ(n)
    ξ = (1 - (1 - η) * ϵ)
    for γ in x.clades
        leaf = isleaf(γ)
        a = zero(T)
        b = zero(T)
        c = Πloss(x, γ, ℓ, n)
        if !leaf
            a += Πroot(x, γ, ℓ, n, η)
            b += Πspeciation(x, γ, ℓ, n)
        end
        #XXX forgot factor (1-ϵ), or should we not have it?
        #@inbounds ℓ[e][1,γ.id] = (1-η)*ξ*a/η + (η/ξ^2)*(b + c)
        @inbounds ℓ[e][1,γ.id] = (1-η)*ξ*a/η + (η*(1-ϵ)/ξ^2)*(b + c)
    end
end

@inline function Πroot(x, γ, ℓ, n, η)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits  # speciation
        @inbounds p += t.p * getl(x, ℓ, e, t.γ1, 1) * getl(x, ℓ, e, t.γ2, 1)
    end
    return p 
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
    return (1 - q)*getl(x, ℓ, id(f), γ.id) + 2q*getϵ(f)*getl(x, ℓ, id(f), γ.id)
end
