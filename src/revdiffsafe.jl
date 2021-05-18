
# A source of major overhead seems to be in the ccd actually...
# the ℓ matrices are way too big! Often only the root matrix has to contain
# all clades... This would probably result in quite a significant speed-up...

# simply put this is what we need, everything should happen inside
# so that it is autodiff compatible.
# For reversediff: Note that arrays explicitly constructed within the target
# function (e.g. via ones, similar, etc.) can be mutated
function _loglikelihood(data, model, θ)
    T = eltype(θ.λ)
    # 1. set the slice probabilities
    slices = get_slices(model, θ, T)  
    # 2. threaded loop over all families
    ℓ = whale_lhood(model, data, slices, θ, T)
    # 3. condition factor
    c = condition(model, slices, θ.η, model.condition)
    sum(ℓ) - length(data)*c
end

# this is already reversediff safe -- should be cleaned up though
function get_slices(model, θ, T)
    @unpack λ, μ, q = θ
    Xs = map(i->similar(model[i].data.slices, T), 1:length(model))
    for n in model.order
        # XXX somehow ReverseDIff complains with UInt indices!
        i = id(n)
        X = Xs[i]
        X[1,3] = one(T)
        X[1,4] = zero(T)
        if Whale.iswgd(n)
            ϵ = Xs[id(n[1])][end,2]
            r = q[Int(Whale.wgdid(n))]
            X[1,2] = r * ϵ^2 + (1. - r) * ϵ
        else
            X[1,2] = isleaf(n) ? zero(T) : 
                Xs[id(n[1])][end,2] * Xs[id(n[2])][end,2]
        end
        for j=2:size(X, 1)
            α = Whale.getα(λ[i], μ[i], n[j,1])
            β = (λ[i]/μ[i])*α
            ϵ = X[j-1,2]
            X[j,2] = Whale._ϵ(α, β, ϵ)
            X[j,3] = Whale._ϕ(α, β, ϵ)
            X[j,4] = Whale._ψ(α, β, ϵ)
        end
    end
    return Xs
end

function whale_lhood(model, data, slices, θ, T) 
    ℓ = Vector{T}(undef, length(data))
    #Threads.@threads 
    for i in 1:length(data)
        ℓ[i] = _whale_lhood(model, data[i], slices, θ, T)
    end
    return ℓ
end

function _whale_lhood(model, x, slices, θ, T)
    ℓ = [zeros(T, size(xᵢ)) for xᵢ in x.ℓ]
    for n in model.order
        _whale!(n, ℓ, x, slices, θ, T)
    end
    L = ℓ[id(root(model))][end,1]
    L > zero(L) ? log(L) : -Inf 
end

function _whale!(n, ℓ, x, slices, θ, T)
    iswgd(n)  && return _whalewgd!(n, ℓ, x, slices, θ, T)
    isroot(n) && return _whaleroot!(n, ℓ, x, slices, θ.η, T)
    e = id(n)
    ϵf = isleaf(n) ? 0. : slices[id(n[1])][end,2]
    ϵg = isleaf(n) ? 0. : slices[id(n[2])][end,2]
    ℓ[e] .= zero(T)
    for γ in x.clades
        !iscompatible(γ, n) && continue
        leaf = isleaf(γ)
        if leaf && isleaf(n)
            ℓ[e][γ.id,1] = one(T)
        elseif !isleaf(n)
            p = Πspeciation(γ, ℓ, n) + _Πloss(γ, ℓ, n, ϵf, ϵg)
            ℓ[e][γ.id,1] += p
        end
        for i=2:size(slices[e], 1)  # iterate over slices
            ℓ[e][γ.id,i] += slices[e][i,3]*ℓ[e][γ.id,i-1]  # propagation
            if !leaf
                ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i, slices[e][i,4])
            end
        end
    end
end

function _whalewgd!(n, ℓ, x, slices, θ, T)
    e = id(n)
    f = id(n[1])
    ϵf = slices[f][end,2]
    q = θ.q[wgdid(n)]
    for γ in x.clades
        !iscompatible(γ, n) && continue
        p = zero(T)
        leaf = isleaf(γ)
        if !leaf
            p += Πwgdretention(γ, ℓ, n, q)
        end
        p += _Πwgdloss(γ, ℓ, n, q, ϵf)
        ℓ[e][γ.id,1] = p
        for i=2:length(n)  # iterate over slices, should be share with non-wgd...
            ℓ[e][γ.id,i] += slices[e][i,3]*ℓ[e][γ.id,i-1]
            if !leaf
                ℓ[e][γ.id,i] += Πduplication(γ, ℓ, n, i, slices[e][i,4])
            end
        end
    end
end

function _whaleroot!(n, ℓ, x, slices, η, T)
    e = id(n)
    ϵ = slices[id(n)][1,2]
    ϵf = slices[id(n[1])][end,2]
    ϵg = slices[id(n[2])][end,2]
    ℓ[e] .= zero(T)
    η_ = one(η)/(one(η) - (one(η) - η) * ϵ)^2
    for γ in x.clades
        leaf = isleaf(γ)
        p1 = _Πloss(γ, ℓ, n, ϵf, ϵg)
        p2 = zero(p1)
        if !leaf
            p1 += Πspeciation(γ, ℓ, n)
            p2 += _Πroot(γ, ℓ, n, η, ϵ)
        end
        ℓ[e][γ.id,1] = p1 * η_ + p2
    end
    ℓ[e][end,1] *= η
end

function condition(m, slices, η, ::RootCondition)
    e = root(m)
    f, g = children(e)
    ϵr = geompgf(η, slices[id(e)][1,2])
    ϵf = geompgf(η, slices[id(f)][end,2])
    ϵg = geompgf(η, slices[id(g)][end,2])
    p = one(η) - ϵf - ϵg + ϵr
    p > zero(p) ? log(p) : -Inf
end

@inline function _Πroot(γ, ℓ, n, η, ϵ)
    e = id(n)
    p = zero(eltype(ℓ[e]))
    for t in γ.splits  # speciation
        @inbounds p += t.p * ℓ[e][t.γ1,1] * ℓ[e][t.γ2,1]
    end
    return p*(one(p) -η)*(one(p) -(one(p) -η) * ϵ)
end

@inline function _Πloss(γ, ℓ, n, ef, eg)
    f = id(n[1])
    g = id(n[2])
    @inbounds ℓ[f][γ.id,end]*eg + ℓ[g][γ.id,end]*ef
end

@inline function _Πwgdloss(γ, ℓ, n, q, ef)
    f = first(children(n))
    @inbounds (one(q) - q)*ℓ[id(f)][γ.id,end] + 2q*ef*ℓ[id(f)][γ.id,end]
end
