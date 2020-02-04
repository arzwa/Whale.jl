# recursive stochastic backtracking of reconciled trees
# i.e. sampling reconciled trees conditional on the data and parameters
# this corresponds actually to sampling latent states, not posterior prediction
# Posterior prediction would be this:
# R|X ~ p(R|X) = (∫θ) p(R|θ,X) p(θ|X) = (∫θ) p(R|θ) p(θ|X)
# but this is not what we do when backtracking trees.
# the ALE algorithm computes a marginal likelihood, integrating over latent
# states (latent states can be thought of as assignments of clades γ to (e,t))
# i.e. slices of the tree.
# in other words, in the Bayesian setting this does not amount to posterior
# prediction (obviously actually) but to sampling a reconciliation from the
# posterior, which is how I treated them in the past, but I didn't actually
# realize we can look at it like latent states

# Should think about how this can be implemented elegantly
# a node in a reconciled tree always corresponds to some clade, this connects
# the rectree to the CCD
@with_kw struct _RecNode{I}
    γ       ::I
    rec     ::I = UInt16(1)
    kind    ::Symbol = :root
    children::Set{_RecNode{I}} = Set{_RecNode{UInt16}}()
    parent  ::Union{Nothing,_RecNode{I}} = nothing
end

# during the recusions, we return these SliceStates, which tell us from where
# to continue backtracking
struct _SliceState{I}
    e::I
    n::_RecNode{I}
    t::Int64
end

initroot(m, c) = _SliceState(m[1].id, _RecNode(γ=c[end], rec=m[1].id), 1)

# do the backtracking (exported function?)
backtrack(wm::WhaleModel, c) = backtrack!(initroot(wm, ccd), wm, c)

# dispatch on node type
backtrack!(s::SliceState, wm, c) = backtrack!(s, wm[s.e], wm, c)

function backtrack!(ss::Vector{SliceState}, wm, c)
    for s in ss
        backtrack!(s, wm[s.e], wm, c)
    end
end

# root backtracking
function backtrack!(s::SliceState, m::WhaleNode{T,Root{T}}, wm, c) where T
    @unpack e, n, t = s
    @unpack kind, γ = n
    p = c.ℓmat[1][γ,1]
    r = kind == :root ? rand()*p/m.event.η : rand()*p
    η_ = 1.0/(1. - (1. - m.event.η) * getϵ(m))^2
    if !isleaf(γ)  # bifurcating events
        @unpack r, next = root_bifurcation(r, m, c.ℓmat, γ, η, η_)
    end
    if r > 0. # loss events
        @unpack r, next = root_nonbifurcation()
    end
    if r > 0.
        error("Backtracking failed, could not obtain latent state, $s")
    end
    backtrack!(next, wm, ccd)
end

# inter branch backtracking
function backtrack!(n::RecNode, m::WhaleNode{T,Speciation{T}}, ccd, γ) where T
end

function backtrack!(n::RecNode, m::WhaleNode{T,WGD{T}}, ccd, γ) where T
end

# intra branch backtracking
function backtrack!(n::RecNode, m, ccd, γ, t) where T
end


function root_bifurcation(r, m, ℓ, γ, η, η_)
    f, g = m.children
    for t in γ.splits
        @unpack p, γ1, γ2 = t
        # either stay in root and duplicate
        r -= p * ℓ[1,t.γ1,1] * ℓ[1,γ2,1] * √(1.0/η_)*(1.0-η)
        if r < 0.
            n1 = RecNode(γ=, rec=m., kind=, parent=m)
            return (r=r, next=[SliceState(1, γ1, 1), SliceState(1, γ2, 1)])
        end
        # or speciate
        r -= p * ℓ[f][γ1, end] * ℓ[g][γ2, end] * η_
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, 1), SliceState(g, γ2, 1)])
        end
        r -= p * ℓ[g][γ1, end] * ℓ[f][γ2, end] * η_
        if r < 0.
            return (r=r, next=[SliceState(g, γ1, 1), SliceState(f, γ2, 1)])
        end
    end
end

function root_nonbifurcation(r, m, ℓ, γ, η_, wm)
    f, g = m.children
    r -= ℓ[f][γ,end] * getϵ(wm[g]) * η_
    if r < 0.
        return (r=r, next=SliceState(f, γ, 1))
    end
    r -= ℓmat[g][γ,end] * getϵ(wm[f]) * η_
    if r < 0.
        return (r=r, next=SliceState(g, γ, 1))
    end
end
