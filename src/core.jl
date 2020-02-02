# ALE agorithm for DLWGD model

# utilities
iscompatible(γ::Clade, n::WhaleNode) = γ.species ⊆ n.clade
getϵ(n::WhaleNode) = n.slices[end,2]
getϕ(n::WhaleNode) = n.slices[end,3]
getϵ(n::WhaleNode, i::Int) = n.slices[i,2]
getϕ(n::WhaleNode, i::Int) = n.slices[i,3]
getα(λ, μ, t) = isapprox(λ, μ) ?
    λ*t/(1. + λ*t) : μ*(exp(t*(λ-μ)) - 1.)/(λ*exp(t*(λ-μ)) - μ)
ℓhood(ℓ) = isfinite(ℓ) ? ℓ : -Inf
integratedϵ(ϵ, η) = η * ϵ / (1. - (1. - η)*ϵ)

# transition probability under the linear BDP 1 → 2
function pdup(λ, μ, t)
    α = getα(λ, μ, t)
    β = (λ/μ)*α
    return (1. - α)*(1. - β)*β
end

logpdf!(wm::WhaleModel, x::CCD, condition::Function=pbothsides) =
    logpdf!(wm, x.ℓtmp, x, condition)

function logpdf!(wm::WhaleModel{T}, ℓ::Array, x::CCD{I,V},
        condition::Function=pbothsides) where {T,I,V}
    for n in wm.order
        whale!(wm[n], ℓ, x, wm)
    end
    nf = condition(wm)
    L = ℓ[1][end,1]
    L = L > zero(L) ? log(L) : -Inf
    ℓhood(L - nf)
end

function logpdf(wm::WhaleModel{T}, x::CCD, condition::Function=pbothsides) where T
    # using `similar` did not give a speedup?
    ℓ = [zeros(T, length(x.clades), length(wm[i])) for i in 1:length(wm)]
    logpdf!(wm, ℓ, x, condition)
end

# mapreduce implementations of logpdf
logpdf(wm::WhaleModel, X::CCDArray, condition::Function=pbothsides) =
    mapreduce((x)->logpdf(wm, x, condition), +, X)

logpdf!(wm::WhaleModel, X::CCDArray, condition::Function=pbothsides) =
    mapreduce((x)->logpdf!(wm, x, condition), +, X)

# log probability of non-extinction
pnonextinct(wm::WhaleModel) = log(1. - integratedϵ(getϵ(wm[1]), wm[1].event.η))

# log probability of non extinction in both clades stemming from the root
function pbothsides(wm::WhaleModel)
    f, g = children(wm[1])
    ϵr = integratedϵ(getϵ(wm[1]), wm[1].event.η)
    ϵf = integratedϵ(getϵ(wm[f]), wm[1].event.η)
    ϵg = integratedϵ(getϵ(wm[g]), wm[1].event.η)
    p = 1. - ϵf - ϵg + ϵr
    p > zero(p) ? log(p) : -Inf
end

function whale!(n::WhaleNode{T,Speciation{T}}, ℓ, x, wm) where T
    set!(n, wm)
    e = n.id
    ℓ[e] .= 0.
    for γ in x.clades
        leaf = isleaf(γ)
        if !iscompatible(γ, n)
            continue
        elseif leaf && isleaf(n)
            ℓ[e][γ.id,1] = 1.0
        elseif !isleaf(n)
            p = Πspeciation(γ, ℓ, n) + Πloss(γ, ℓ, n, wm)
            ℓ[e][γ.id,1] += p
        end
        for i=2:length(n)  # iterate over slices
            ℓ[e][γ.id,i] += getϕ(n, i)*ℓ[e][γ.id,i-1]
            if !leaf
                ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i)
            end
        end
    end
end

function whale!(n::WhaleNode{T,WGD{T}}, ℓ, x, wm) where T
    set!(n, wm)
    e = n.id
    nextsp = nonwgdchild(n, wm)
    ℓ[e] .= 0.
    λ = nextsp.event.λ
    μ = nextsp.event.μ
    for γ in x.clades
        leaf = isleaf(γ)
        p = 0.
        if !iscompatible(γ, n)
            continue
        else
            if !leaf
                p += Πwgdretention(γ, ℓ, n)
            end
            p += Πwgdloss(γ, ℓ, n, wm)
            ℓ[e][γ.id,1] = p
            for i=2:length(n)  # iterate over slices
                ℓ[e][γ.id,i] += getϕ(n, i)*ℓ[e][γ.id,i-1]
                if !leaf
                    ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i, λ, μ)
                end
            end
        end
    end
end

function whale!(n::WhaleNode{T,Root{T}}, ℓ, x, wm) where T
    set!(n, wm)
    ℓ[1] .= 0.
    η_ = 1.0/(1. - (1. - n.event.η) * getϵ(n))^2
    for γ in x.clades
        leaf = isleaf(γ)
        p1 = Πloss(γ, ℓ, n, wm)
        p2 = 0.
        if !leaf
            p1 += Πspeciation(γ, ℓ, n)
            p2 += Πroot(γ, ℓ, n, wm)
        end
        ℓ[1][γ.id,1] = p1 * η_ + p2
    end
    ℓ[1][end,1] *= n.event.η
end

@inline function Πroot(γ, ℓ, n, wm)
    p = 0.0
    for t in γ.splits  # speciation
        @inbounds p += t.p * ℓ[1][t.γ1,1] * ℓ[1][t.γ2,1]
    end
    return p*(1. -n.event.η)*(1. -(1. -n.event.η) * getϵ(n))
end

@inline function Πspeciation(γ, ℓ, n)
    f, g = children(n)
    p = 0.0
    for t in γ.splits
        @inbounds p += t.p * (
            ℓ[f][t.γ1,end] * ℓ[g][t.γ2,end] +
            ℓ[g][t.γ1,end] * ℓ[f][t.γ2,end])
    end
    return p
end

@inline function Πwgdretention(γ, ℓ, n)
    f = first(children(n))
    p = 0.0
    for t in γ.splits
        @inbounds p += t.p * ℓ[f][t.γ1,end] * ℓ[f][t.γ2,end]
    end
    return p * n.event.q
end

@inline function Πwgdloss(γ, ℓ, n, wm)
    f = first(children(n))
    q = n.event.q
    (1.0 - q)*ℓ[f][γ.id,end] + 2*q*getϵ(wm[f])*ℓ[f][γ.id,end]
end

@inline function Πloss(γ, ℓ, n, wm)
    f, g = children(n)
    @inbounds p = ℓ[f][γ.id,end]*getϵ(wm[g]) + ℓ[g][γ.id,end]*getϵ(wm[f])
    return p
end

@inline function Πduplication(γ, ℓ, n, i, λ=n.event.λ, μ=n.event.μ)
    p = 0.
    for t in γ.splits
        @inbounds p += t.p * ℓ[n.id][t.γ1,i-1] * ℓ[n.id][t.γ2,i-1]
    end
    # return p * n.event.λ * n.slices[i,1]
    return p * pdup(λ, μ, n.slices[i,1])
end
