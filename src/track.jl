"""
    RecNode{I}

Reconciled tree node, holds a reference to its corresponding clade (`γ`)
and species tree node. The reference to the species tree node together
with the number of leaves should give sufficient information. Loss nodes
are identified by having `γ == 0`.
"""
@with_kw struct RecNode{I}
    γ::I                # clade in CCD
    e::I = UInt16(1)    # edge (in species tree)
    t::Int = 1          # slice index along edge `e`
    children::Set{RecNode{I}} = Set{RecNode{UInt16}}()
    parent  ::Union{Nothing,RecNode{I}} = nothing
end

# NOTE `t` is not inhash, because duplications with different `t` should ==
Base.hash(r::RecNode) = hash((r.γ, r.e, hash(r.children)))
Base.push!(n::RecNode{I}, m::RecNode{I}) where I = push!(n.children, m)
Base.show(io::IO, n::RecNode) = write(io, "RecNode(γ=$(n.γ); rec=$(n.e))")

rectuple(r::RecNode) = (r.γ, r.e)
cladehash(r::RecNode) = r.γ == 0 ? hash((r.γ, r.e, sister(r).γ)) :
    hash((r.γ, r.e, Set(rectuple.(r.children))))

NewickTree.isleaf(n::RecNode) = length(n.children) == 0
NewickTree.isroot(n::RecNode) = isnothing(n.parent)
NewickTree.children(n::RecNode) = collect(n.children)
NewickTree.id(n::RecNode) = cladehash(n)
NewickTree.distance(r::RecNode) = 1.

sister(n::RecNode) = first(setdiff(n.parent.children, Set([n])))

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
    SliceState(model[1].id, ccd[end].id, 1),
    RecNode(γ=ccd[end].id, e=model[1].id), model, ccd)

function (b::BackTracker)(newstate::SliceState)
    @unpack γ, e, t = newstate
    if γ != b.node.γ || e != b.node.e
        newnode = RecNode(γ=γ, e=e, t=t, parent=b.node)
        push!(b.node, newnode)
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

backtrack!(b::BackTracker) = b.state.γ == 0 ? (return) : b.state.t == 1 ?
    _backtrack!(b, b.model[b.state.e]) : _backtrack!(b)

backtrack!(b::BackTracker, next::SliceState) = backtrack!(b(next))

function backtrack!(b::BackTracker, next::Vector)
    for nextstate in next
         backtrack!(b, nextstate)
    end
end

function _backtrack!(b, n)  # internode backtracking
    isleaf(n) ? (return) : nothing  # terminates recursion
    @unpack node, state, ccd = b
    @unpack e, γ, t = state
    r = rand()*ccd.ℓtmp[e][γ,t]
    return _backtrack!(r, b, n)  # dispatch on node type
end

# NOTE: implementation note
# there is a _backtrack!(r, b, n) function for each node type in the tree
# this should be fairly easy to extend in case we devise models with other
# events (e.g. WGT)
function _backtrack!(r, b, n::WhaleNode{T,Speciation{T}}) where T
    for f in [sploss, speciation]
        @unpack r, next = f(r, b, n)
        if r < 0.
            return backtrack!(b, next)
        end
    end
    error(b, r)
end

function _backtrack!(r, b, n::WhaleNode{T,Root{T}}) where T
    if b.state.γ == length(b.ccd)
        r /= n.event.η
    end
    η_ = 1.0/(1. - (1. - n.event.η) * getϵ(n))^2
    for f in [rootbifurcation, rootloss]
        @unpack r, next = f(r, b, n, η_)
        if r < 0.0
            return backtrack!(b, next)
        end
    end
    error(b, r)
end

function _backtrack!(r, b, n::WhaleNode{T,WGD{T}}) where T
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
    r = rand()*ccd.ℓtmp[e][γ,t]
    r -= getϕ(model[e], t) * ccd.ℓtmp[e][γ,t-1]
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
    nextsp = nonwgdchild(model[e], model)
    @unpack λ, μ = nextsp.event
    Δt = model[e].slices[t,1]
    ℓ = ccd.ℓtmp
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
    f, g = m.children
    tf = lastslice(model, f)
    tg = lastslice(model, g)
    ℓ = ccd.ℓtmp
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
    ℓ = ccd.ℓtmp
    f, g = m.children
    r -= ℓ[f][γ,end]*getϵ(model[g])
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(model, f)), Loss(g)])
    end
    r -= ℓ[g][γ,end]*getϵ(model[f])
    if r < 0.
        return (r=r, next=[SliceState(g, γ, lastslice(model, g)), Loss(g)])
    end
    return (r=r, next=SliceState[])
end

function wgdloss(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    f = first(m.children)
    q = m.event.q
    r -= (1. - q + 2q*getϵ(model[f])) * ccd.ℓtmp[f][γ,end]
    return (r=r, next=[SliceState(f, γ, lastslice(model, f)), Loss(f)])
end

function wgdretention(r, b, m)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    f = first(m.children)
    q = m.event.q
    tf = lastslice(model, f)
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= q * p * ccd.ℓtmp[f][γ1,end] * ccd.ℓtmp[f][γ2,end]
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(f, γ2, tf)])
        end
    end
    return (r=r, next=SliceState[])
end

function rootbifurcation(r, b, m, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    ℓ = ccd.ℓtmp
    f, g = m.children
    tf = lastslice(model, f)
    tg = lastslice(model, g)
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        # either stay in root and duplicate
        r -= p * ℓ[e][γ1,t] * ℓ[e][γ2,t] * √(1.0/η_)*(1.0-m.event.η)
        if r < 0.
            return (r=r, next=[SliceState(e, γ1, t), SliceState(e, γ2, t)])
        end
        # or speciate
        r -= p * ℓ[f][γ1,end] * ℓ[g][γ2,end] * η_
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(g, γ2, tg)])
        end
        r -= p * ℓ[g][γ1,end] * ℓ[f][γ2,end] * η_
        if r < 0.
            return (r=r, next=[SliceState(g, γ1, tg), SliceState(f, γ2, tf)])
        end
    end
    return (r=r, next=SliceState[])
end

function rootloss(r, b, m, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    ℓ = ccd.ℓtmp
    f, g = m.children
    r -= ℓ[f][γ,end] * getϵ(model[g]) * η_
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(model, f)), Loss(g)])
    end
    r -= ℓ[g][γ,end] * getϵ(model[f]) * η_
    if r < 0.
        # XXX (*):  state.node.kind = :sploss
        return (r=r, next=[SliceState(g, γ, lastslice(model, g)), Loss(f)])
    end
    return (r=r, next=SliceState[])
end

# NOTE: we should set the parent node at the relevant occasion (see e.g. * ↑)
