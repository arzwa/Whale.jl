"""
    RecNode{I}

Reconciled tree node, holds a reference to its corresponding clade (`γ`)
and species tree node. The reference to the species tree node together
with the number of leaves should give sufficient information. Loss nodes
are identified by having `γ == 0`.
"""
@with_kw mutable struct RecData{I}
    γ::I                # clade in CCD
    e::I = UInt16(1)    # edge (in species tree)
    t::Int = 1          # slice index along edge `e`
    name::String = ""
end

const RecNode{I} = Node{I,RecData{I}} where I<:Integer

# NOTE `t` is not inhash, because duplications with different `t` should ==
Base.hash(r::RecNode) = hash((r.data.γ, r.data.e, hash(Set(children(r)))))
Base.show(io::IO, n::RecData) = write(io, "$(n.e), $(n.name)")

getγ(n::RecNode) = n.data.γ
gete(n::RecNode) = n.data.e
gett(n::RecNode) = n.data.t
sister(n::RecNode) = first(setdiff(children(parent(n)), Set([n])))
rectuple(r::RecNode) = (r.data.γ, r.data.e)
cladehash(r::RecNode) = r.data.γ == 0 ?
    hash((r.data.γ, r.data.e, sister(r).data.γ)) :
    hash((r.data.γ, r.data.e, Set(rectuple.(children(r)))))

# NewickTree.isleaf(n::RecNode) = length(n.children) == 0
# NewickTree.isroot(n::RecNode) = isnothing(n.parent)
# NewickTree.children(n::RecNode) = collect(n.children)
# NewickTree.id(n::RecNode) = cladehash(n)
NewickTree.distance(r::RecNode) = 1.
NewickTree.name(r::RecData) = r.name

"""
    SliceState{I}

Should not be exported. This essentially holds the last backtracked state
for the slice (branch=e,time=t). An instance thus represents the observed
(sampled) state for the latent random variable Yₜᵉ. A sequence of these
Yₜᵉ across all (e,t) defines a reconciled tree.
"""
struct SliceState{I}
    e::I
    γ::I
    t::Int64
end

Loss(e) = SliceState(e, zero(e), 0)
Base.show(io::IO, s::SliceState) = write(io, "$(Int64.((s.e, s.γ, s.t)))")

"""
    BackTracker{I}

Should not be exported. This aids in the recursions.

NOTE: Not sure if it is wasteful to hold the model and CCD in here.
I guess since these are just references it is fine?
"""
struct BackTracker{I}
    state::SliceState{I}
    node ::RecNode{I}
    model::WhaleModel
    ccd  ::CCD
end

Base.error(b::BackTracker, r) = error("Backtracking failed, r=$r at Y=$b.state")

function Base.:-(b::BackTracker)
    @unpack e, γ, t = b.state
    BackTracker(SliceState(e, γ, t-1), b.node, b.model, b.ccd)
end

BackTracker(model::WhaleModel, ccd::CCD) = BackTracker(
    SliceState(id(root(model)), ccd[end].id, 1),
    Node(ccd[end].id, RecData(γ=ccd[end].id, e=id(root(model)))),
    model, ccd)

function (b::BackTracker)(newstate::SliceState)
    @unpack γ, e, t = newstate
    if γ != getγ(b.node) || e != gete(b.node)
        newnode = Node(γ, RecData(γ=γ, e=e, t=t), b.node)
    else
        newnode = b.node
    end
    BackTracker(newstate, newnode, b.model, b.ccd)
end

"""
    backtrack(wm::WhaleModel, ccd::CCD)
    backtrack(wm::WhaleModel, ccd::AbstractVector{CCD})

Backtracking function, samples latent states (which corresponds to a
reconciliation) conditional on the current `ℓmat` in the CCD object(s) (i.e.
the dynamic programming matrix containing the fractional likelihoods for
each clade at all internal slices of the species tree). This `ℓmat` is
computed for a given parameterized WhaleModel using `logpdf!(wm, ccd)`.

```julia
ℓhood = logpdf!(wm, ccd)
rtree = backtrack(wm, ccd)
NewickTree.print_tree(rtree)
```
"""
backtrack(wm::WhaleModel, D::AbstractVector) = map((ccd)->backtrack(wm, ccd), D)

function backtrack(wm::WhaleModel, ccd::CCD)
    bt = BackTracker(wm, ccd)
    backtrack!(bt)
    return bt.node
end

backtrack!(b::BackTracker, next::SliceState) = backtrack!(b(next))

function backtrack!(b::BackTracker, next::Vector)
    for nextstate in next
         backtrack!(b, nextstate)
    end
end

# backtracking starts here for real
backtrack!(b::BackTracker) =
    b.state.γ == 0 ? (return addleafname!(b, "loss_$(hash(parent(b.node)))")) :
        b.state.t == 1 ?
            _backtrack!(b, b.model[b.state.e]) :
            _backtrack!(b)

# `n` is a ModelNode!
function _backtrack!(b, n)  # internode backtracking
    isleaf(n) ? (return addleafname!(b)) : nothing  # terminates recursion
    @unpack node, state, ccd = b
    @unpack e, γ, t = state
    r = rand()*ccd.ℓ[e][γ,t]
    if isroot(n)
        return _backtrackroot!(r, b, n)
    elseif iswgd(n)
        return _backtrackwgd!(r, b, n)
    else
        return _backtrack!(r, b, n)
    end
end

addleafname!(b) = b.node.data.name = b.ccd.leaves[getγ(b.node)]
addleafname!(b, s) = b.node.data.name = s

# NOTE: implementation note
# there is a _backtrack!(r, b, n) function for each node kind in the tree
# this should be fairly easy to extend in case we devise models with other
# events (e.g. WGT)
function _backtrack!(r, b, n)
    for f in [sploss, speciation]
        @unpack r, next = f(r, b, n)
        if r < 0.
            return backtrack!(b, next)
        end
    end
    error(b, r)
end

function _backtrackroot!(r, b, n)
    @unpack η = getθ(b.model.rates, n)
    if b.state.γ == length(b.ccd)
        r /= η
    end
    η_ = one(η)/(one(η) - (one(η) - η) * getϵ(n))^2
    for f in [rootbifurcation, rootloss]
        @unpack r, next = f(r, b, n, η_)
        if r < zero(r)
            return backtrack!(b, next)
        end
    end
    error(b, r)
end

function _backtrackwgd!(r, b, n)
    for f in [wgdloss, wgdretention]
        @unpack r, next = f(r, b, n)
        if r < 0.
            return backtrack!(b, next)
        end
    end
    error(b, r)
end

function _backtrack!(b::BackTracker)  # intra branch backtracking
    @unpack state, ccd, model = b
    @unpack e, γ, t = state
    if isleaf(ccd[γ])
        return backtrack!(b, SliceState(e, γ, 1))
    end
    r = rand()*ccd.ℓ[e][γ,t]
    r -= getϕ(model[e], t) * ccd.ℓ[e][γ,t-1]
    if r < 0.
        return backtrack!(-b)
    end
    @unpack r, next = duplication(r, b)
    r < 0. ? backtrack!(b, next) : error(b, r)
end

# NOTE: implementation note
# These are the actual events and this repeats a lot from the core ALE
# algorithm. Would be neat to strip down repetitive code but no priority.
function duplication(r, b)
    @unpack state, ccd, model = b
    @unpack e, γ, t = state
    @unpack ℓ = ccd
    @unpack λ, μ = getθ(model.rates, model[e])
    Δt = model[e][t,1]
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= p * ℓ[e][γ1,t-1] * ℓ[e][γ2,t-1] * pdup(exp(λ), exp(μ), Δt)
        if r < 0.
            return (r=r, next=[SliceState(e, γ1, t-1), SliceState(e, γ2, t-1)])
        end
    end
    return (r=r, next=SliceState[])
end

function speciation(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t = state
    @unpack ℓ = ccd
    f, g = id(m[1]), id(m[2])
    tf = lastslice(m[1])
    tg = lastslice(m[2])
    ℓ = ccd.ℓ
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= p * ℓ[f][γ1,end] * ℓ[g][γ2,end]
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(g, γ2, tg)])
        end
        r -= p * ℓ[g][γ1,end] * ℓ[f][γ2,end]
        if r < 0.
            return (r=r, next=[SliceState(g, γ1, tg), SliceState(f, γ2, tf)])
        end
    end
    return (r=r, next=SliceState[])
end

function sploss(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    @unpack ℓ = ccd
    f, g = id(m[1]), id(m[2])
    r -= ℓ[f][γ,end]*getϵ(m[2])
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(g)])
    end
    r -= ℓ[g][γ,end]*getϵ(m[1])
    if r < 0.
        return (r=r, next=[SliceState(g, γ, lastslice(m[2])), Loss(f)])
    end
    return (r=r, next=SliceState[])
end

function wgdloss(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    @unpack q = getθ(model.rates, m)
    f = id(m[1])
    r -= (1. - q + 2q*getϵ(m[1])) * ccd.ℓ[f][γ,end]
    return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(f)])
end

function wgdretention(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    @unpack q = getθ(model.rates, m)
    f = id(m[1])
    tf = lastslice(m[1])
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= q * p * ccd.ℓ[f][γ1,end] * ccd.ℓ[f][γ2,end]
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(f, γ2, tf)])
        end
    end
    return (r=r, next=SliceState[])
end

function rootbifurcation(r, b, m, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    @unpack ℓ = ccd
    @unpack η = getθ(model.rates, m)
    f, g = id(m[1]), id(m[2])
    tf = lastslice(m[1])
    tg = lastslice(m[2])
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        # either stay in root and duplicate
        r -= p * ℓ[e][γ1,t] * ℓ[e][γ2,t] * √(one(η)/η_)*(one(η)-η)
        if r < zero(r)
            return (r=r, next=[SliceState(e, γ1, t), SliceState(e, γ2, t)])
        end
        # or speciate
        r -= p * ℓ[f][γ1,end] * ℓ[g][γ2,end] * η_
        if r < zero(r)
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(g, γ2, tg)])
        end
        r -= p * ℓ[g][γ1,end] * ℓ[f][γ2,end] * η_
        if r < zero(r)
            return (r=r, next=[SliceState(g, γ1, tg), SliceState(f, γ2, tf)])
        end
    end
    return (r=r, next=SliceState[])
end

function rootloss(r, b, m, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    @unpack ℓ = ccd
    f, g = id(m[1]), id(m[2])
    r -= ℓ[f][γ,end] * getϵ(m[2]) * η_
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(g)])
    end
    r -= ℓ[g][γ,end] * getϵ(m[1]) * η_
    if r < 0.
        # XXX (*):  state.node.kind = :sploss
        return (r=r, next=[SliceState(g, γ, lastslice(m[2])), Loss(f)])
    end
    return (r=r, next=SliceState[])
end

# NOTE: we should set the parent node at the relevant occasion (see e.g. * ↑)
