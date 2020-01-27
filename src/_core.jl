
iscompatible(γ::Clade, n::WhaleNode) = γ.species ⊆ n.clade

getϵ(n::WhaleNode) = n.slices[end,2]
getϕ(n::WhaleNode) = n.slices[end,3]
getϵ(n::WhaleNode, i::Int) = n.slices[i,2]
getϕ(n::WhaleNode, i::Int) = n.slices[i,3]
getα(λ, μ, t) = μ*(exp(t*(λ-μ)) - 1.)/(λ*exp(t*(λ-μ)) - μ)
ℓhood(ℓ) = isfinite(ℓ) ? ℓ : -Inf
integratedϵ(ϵ, η) = η * ϵ / (1. - (1. - η)*ϵ)

function pdup(λ, μ, t)
    α = getα(λ, μ, t)
    β = (λ/μ)*α
    return (1. - α)*(1. - β)*β
end

function logpdf!(wm::WhaleModel{T}, x::CCD, condition::Function=pbothsides) where T
    for n in wm.order
        whale!(wm[n], x, wm)
    end
    nf = condition(wm)::T
    ℓhood(log(ccd.ℓtmp[1][end,1]) - nf)::T
end

# log probability of non-extinction
pnonextinct(wm::WhaleModel) = log(1. - integratedϵ(getϵ(wm[1]), wm[1].event.η))

# log probability of non extinction in both clades stemming from the root
function pbothsides(wm::WhaleModel)
    f, g = children(wm[1])
    ϵr = integratedϵ(getϵ(wm[1]), wm[1].event.η)
    ϵf = integratedϵ(getϵ(wm[f]), wm[1].event.η)
    ϵg = integratedϵ(getϵ(wm[g]), wm[1].event.η)
    log(1. - ϵf - ϵg + ϵr)
end

function whale!(n::WhaleNode{T,Speciation{T}}, x::CCD, wm::WhaleModel) where T
    @unpack ℓtmp, clades = x
    e = n.id
    ℓtmp[e] .= 0.
    for γ in clades
        leaf = isleaf(γ)
        if !iscompatible(γ, n)
            continue
        elseif leaf && isleaf(n)
            ℓtmp[e][γ.id,1] = 1.0
        elseif !isleaf(n)
            p = Πspeciation(γ, ℓtmp, n) + Πloss(γ, ℓtmp, n, wm)
            ℓtmp[e][γ.id,1] += p
        end
        for i=2:length(n)  # iterate over slices
            ℓtmp[e][γ.id,i] += getϕ(n, i)*ℓtmp[e][γ.id,i-1]
            if !leaf
                ℓtmp[e][γ.id,i] += Πduplication(γ, ℓtmp, n, i)
            end
        end
    end
end

function whale!(n::WhaleNode{T,Root{T}}, x::CCD, wm::WhaleModel) where T
    @unpack ℓtmp, clades = x
    ℓtmp[1] .= 0.
    η_ = 1.0/(1. - (1. - n.event.η) * getϵ(n))^2
    for γ in clades
        leaf = isleaf(γ)
        p1 = Πloss(γ, ℓtmp, n, wm)
        p2 = 0.
        if !leaf
            p1 += Πspeciation(γ, ℓtmp, n)
            p2 += Πroot(γ, ℓtmp, n, wm)
        end
        ℓtmp[1][γ.id,1] = p1 * η_ + p2
    end
    ℓtmp[1][end,1] *= n.event.η
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
    for t in γ.splits  # speciation
        # maybe use @inbounds
        @inbounds p += t.p * (
            ℓ[f][t.γ1,end] * ℓ[g][t.γ2,end] +
            ℓ[g][t.γ1,end] * ℓ[f][t.γ2,end])
    end
    return p
end

@inline function Πloss(γ, ℓ, n, wm)
    f, g = children(n)
    @inbounds p = ℓ[f][γ.id,end]*getϵ(wm[g]) + ℓ[g][γ.id,end]*getϵ(wm[f])
    return p
end

@inline function Πduplication(γ, ℓ, n, i)
    p = 0.
    for t in γ.splits
        @inbounds p += t.p * ℓ[n.id][t.γ1,i-1] * ℓ[n.id][t.γ2,i-1]
    end
    return p * n.event.λ * n.slices[i,1]
    # return p * pdup(n.event.λ, n.event.μ, n.slices[i,1])
end
