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

struct RecNode{I}
    rec ::I
    kind::Symbol
end

# during the recusions, we return these SliceStates, which tell us from where
# to continue backtracking
struct SliceState
    n::RecNode
    e::
    γ::
    t::
end

initroot(m, c) = SliceState(RecNode(m[1].id, :root), m[1].id, c.clades[end], 1)

# do the backtracking (exported function?)
backtrack(wm::WhaleModel, ccd) = backtrack!(initroot(wm, ccd), wm, ccd)

# dispatch on node type
backtrack!(s::SliceState, wm, ccd) = backtrack!(s, wm[s.e], wm, ccd)

function backtrack!(ss::Vector{SliceState}, wm, ccd)
    for s in ss
        backtrack!(s, wm[s.e], wm, ccd)
    end
end

# root backtracking
function backtrack!(s, m::WhaleNode{T,Root{T}}, wm, ccd) where T
    p = ccd.ℓmat[1][γ,1]
    r = n.kind == :root ? rand()*p/m.event.η : rand()*p
    η_ = 1.0/(1. - (1. - m.event.η) * getϵ(m))^2

    # bifurcating events
    if !isleaf(γ)
        @unpack r, next = root_bifurcation(r, )
    end

    # loss events
    if r > 0.
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
            return (r=r, next=[SliceState(1, γ1, 1), SliceState(1, γ2, 1)])
        end
        # or speciate
        r -= p * ℓ[f][γ1, end] * ℓ[g][γ2, end] * η_
        r < 0. ? (return r, γ1, γ2, f, g) : nothing

        r -= p * ℓ[g][γ1, end] * ℓ[f][γ2, end] * η_
        r < 0. ? (return r, γ1, γ2, g, f) : nothing
    end
    return r, -1, -1, g, f
end
