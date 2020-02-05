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
@with_kw struct RecNode{I}
    γ       ::I
    rec     ::I = UInt16(1)
    kind    ::Symbol = :root
    children::Set{RecNode{I}} = Set{RecNode{UInt16}}()
    parent  ::Union{Nothing,RecNode{I}} = nothing
end

Base.push!(n::RecNode{I}, m::RecNode{I}) where I = push!(n.children, m)

# during the recusions, we return these SliceStates, which tell us from where
# to continue backtracking
struct SliceState{I}
    e::I
    γ::I
    t::Int64
end

struct BackTracker{I}
    state::SliceState{I}
    node ::RecNode{I}
    model::WhaleModel
    ccd  ::CCD
end

BackTracker(model, ccd) = BackTracker(
    SliceState(model[1].id, ccd[end].id, 1),
    RecNode(γ=ccd[end].id, rec=model[1].id))

# non-bifurcations can be sploss or wgdloss
function update(b::BackTracker, newstate::SliceState)
    @unpack state, node, model, ccd = b
    @unpack e, γ, t = newstate
    if iswgd(model[e])  # newstate is in WGD branch
        # add WGD node
        newnode =
    else
        # add sploss node
        newnode =
    end
    BackTracker(newstate, newnode, model, ccd)
end

# bifurcations
function update!(b::BackTracker, newstates::Vector{SliceState})
    @unpack state, node, model = b
    @unpack e, γ, t = newstate
    # TODO don't like this, should be dispatching on node type?
    if state.e != e && !iswgd(model[e])   # speciation
    elseif state.e != e  # WGD retention
    elseif t == 1 # we stayed in the root
    else  # we duplicated (stayed in same branch, t != 0)
    end
end

# do the backtracking (exported function?)
backtrack(wm::WhaleModel, ccd) = backtrack!(BackTracker(wm, ccd))

# dispatch on node type
backtrack!(b::BackTracker) = backtrack!(b, b.model[b.state.e])

# root backtracking
function backtrack!(b::BackTracker, n::WhaleNode{T,Root{T}}) where T
    @unpack node, state, model, ccd = b
    @unpack e, γ, t = state
    p = ccd.ℓmat[e][γ,t]
    r = node.kind == :root ? rand()*p/n.event.η : rand()*p
    η_ = 1.0/(1. - (1. - n.event.η) * getϵ(m))^2
    if !isleaf(γ)  # bifurcating events
        @unpack r, next = root_bifurcation!(r, state, n, ccd.ℓmat, η, η_)
        if r < 0.0
            bs = update(b, next)  # TODO add new node to rectree and updates state
            return backtrack!.(bs)
        end
    end
    # loss
    @unpack r, next = root_nonbifurcation!(r, state, n, ccd.ℓmat, η_, model)
    if r < 0.0
        update!(b, next)
        return backtrack!(b)
    else
        error("Backtracking failed, could not obtain latent state, $s")
    end
end

# inter branch backtracking
function backtrack!(s::SliceState, m::WhaleNode{T,Speciation{T}}, ccd, γ) where T
    @unpack e, γ, t, n = s
    r = rand()*ccd.ℓmat[e][γ,t]
    @unpack r, next = non_bifurcation!(r, m, s, ccd.ℓmat, wm)
end

function backtrack!(s::SliceState, m::WhaleNode{T,WGD{T}}, ccd, γ) where T
end

# intra branch backtracking
function backtrack!(n::RecNode, m, ccd, γ, t) where T
end


function non_bifurcation!(r, s::SliceState, m::WhaleNode, ℓ, wm)
    @unpack e, γ, t, n = s
    f, g = m.children
    r -= ℓ[f][γ,end]*getϵ(wm[g])
    if r < 0.
        newnode = splossnode!(n, e, γ)
        return (r=r, next=SliceState(f, γ, 1))
    end
    r -= ℓ[g][γ,end]*getϵ(wm[f])
    if r < 0.
        newnode = splossnode!(n, e, γ)
        return (r=r, next=SliceState(g, γ, 1))
    end
end

function root_bifurcation!(r, s::SliceState, m::WhaleNode, ℓ, η, η_)
    @unpack e, γ, t, n = s
    f, g = m.children
    for t in γ.splits
        @unpack p, γ1, γ2 = t
        # either stay in root and duplicate
        r -= p * ℓ[e,t.γ1,t] * ℓ[e,γ2,t] * √(1.0/η_)*(1.0-η)
        if r < 0.
            return (r=r, next=[SliceState(e, γ1, t), SliceState(e, γ2, t)])
        end
        # or speciate
        r -= p * ℓ[f][γ1,end] * ℓ[g][γ2,end] * η_
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, t), SliceState(g, γ2, t)])
        end
        r -= p * ℓ[g][γ1,end] * ℓ[f][γ2,end] * η_
        if r < 0.
            return (r=r, next=[SliceState(g, γ1, t), SliceState(f, γ2, t)])
        end
    end
end

function root_nonbifurcation!(r, s::SliceState, m::WhaleNode, ℓ, η_, wm)
    @unpack e, γ, t, n = s
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

function dupnode!(n::RecNode{I}, γ::I) where I
    newnode = RecNode(γ=γ, rec=n.rec, kind=:duplication, parent=n)
    push!(n, newnode)
    newnode
end

function spnode!(n::RecNode{I}, e::I, γ::I)
    newnode = RecNode(γ=γ, rec=e, kind=:speciation, parent=n)
    push!(n, newnode)
    newnode
end

function splossnode!(n::RecNode{I}, e::I, γ::I)
    newnode = RecNode(γ=γ, rec=e, kind=:sploss, parent=n)
    push!(n, newnode)
    newnode
end
