# XXX: the tracking algorithm is very memory intensive and quiet slow...
# should do some research
"""
    TreeTracker(model, data, df, fun)

A helper struct for backtracking trees from a posterior distribution.
This should be fairly generally applicable, with the `fun` field
holding a function that parses out a named tuple with the parameters
for a `WhaleModel` from an entry of the DataFrame in the `df` field.
"""
struct TreeTracker{T,V,X}
    model::WhaleModel{T}
    data ::AbstractVector{V}
    df   ::X # should implement `eachrow()`
    fun  ::Function
end

# XXX This can be optimized, speed-wise it would be better to sample a tree for
# all families in each iteration, while now we do it the other way around to
# prevent having a huge array of trees... If we could summarize trees on the go
# however, we could interchange the order of the loops.
track(tt::TreeTracker; kwargs...) = track_threaded(tt; kwargs...)

function flsh()  # flush streams (on cluster I have troubles with this)
    flush(stderr)
    flush(stdout)
end

# XXX: allocates too much memory!
function track_threaded(tt::TreeTracker; progress=true, outdir="", summary=true)
    @unpack model, data, df, fun = tt
    outdir != "" && mkpath(outdir)
    result = Vector{RecSummary}(undef, length(data))
    @threads for i=1:length(result)
        # for i=1:length(result)
        progress && (@info "Tracking $(data[i].fname)" ; flsh())
        result[i] = track_and_sum(model, df, fun, data[i], outdir, summary)
    end
    return result
end

# Both DArray and pmap based implementation. DArray allocates considerably less
# memory!
function track_distributed(tt::TreeTracker;
        progress=true, outdir::String="", summary=true)
    @unpack model, data, df, fun = tt
    outdir != "" && mkpath(outdir)
    f = progress ?
        x->begin @info "Tracking $(x.fname)"; flsh();
           track_and_sum(model, df, fun, x, outdir, summary) end :
        x->track_and_sum(model, df, fun, x, outdir, summary)
    # result = typeof(data)<:DArray ? map(x->f(x), data) : pmap(x->f(x), data)
    result = map(x->f(x), data)
    return result
end

# ccd is an individual family, not the full vector of ccds!
function track_and_sum(model, df, fun, ccd, outdir="", summary=true)
    trees = Array{RecNode,1}(undef, size(df)[1])
    for (i,x) in enumerate(eachrow(df))
        wmm = fun(model, x)
        ℓ = logpdf!(wmm, ccd)
        trees[i] = backtrack(wmm, ccd)
    end
    if summary
        rs = sumtrees(trees, ccd, model)
        if outdir != ""
            # CSV.write( joinpath(outdir, "$(ccd.fname).csv"), rs.events)
            writetrees(joinpath(outdir, "$(ccd.fname).trees"), rs.trees)
        end
    end
    return summary ? rs : trees
end

"""
    RecNode{I}

Reconciled tree node, holds a reference to its corresponding clade (`γ`)
and species tree node. The reference to the species tree node together
with the number of leaves should give sufficient information. Loss nodes
are identified by having `γ == 0`.
"""
@with_kw_noshow mutable struct RecData{I}
    γ::I                # clade in CCD
    e::I = UInt16(1)    # edge (in species tree)
    t::Int = 1          # slice index along edge `e`
    name::String  = ""
    cred::Float64 = NaN
    label::String = ""
end

const RecNode{I} = Node{I,RecData{I}} where I<:Integer

Base.show(io::IO, n::RecData) = write(io, "$(n.e), $(n.name)")
getγ(n::RecNode) = n.data.γ
gete(n::RecNode) = n.data.e
gett(n::RecNode) = n.data.t
rectuple(r::RecNode) = (r.data.γ, r.data.e)
sister(n::RecNode) = first(setdiff(children(parent(n)), Set([n])))
NewickTree.distance(r::RecNode) = NaN
NewickTree.support(r::RecData) = r.cred
NewickTree.name(r::RecData) = r.name

# NOTE `t` is not in hash, because duplications with different `t` should ==
# NOTE: sets of nodes use hashes (e.g. in `sister`), better not extend Base.hash!
# `nodehash` below is extremely slow for large trees! The one relying on
# `cladehash` is much better (but also unique?)
# nodehash(r::RecNode) = hash((r.data.γ, r.data.e, hash(Set(children(r)))))
nodehash(r::RecNode) = hash(Set([cladehash(n) for n in postwalk(r)]))

cladehash(r::RecNode) = r.data.γ == 0 ?
    hash((r.data.γ, r.data.e, sister(r).data.γ)) :
    hash((r.data.γ, r.data.e, Set(rectuple.(children(r)))))

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

# fixes a bug in GH actions tests?
SliceState(e, γ, t) = SliceState(promote(e, γ)..., Int64(t))

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

Base.error(b::BackTracker, r) = error("Backtracking failed, r=$r at Y=$b")

function Base.:-(b::BackTracker)
    @unpack e, γ, t = b.state
    BackTracker(SliceState(e, γ, t-1), b.node, b.model, b.ccd)
end

BackTracker(model::WhaleModel, ccd::CCD) = BackTracker(
    SliceState(id(root(model)), ccd[end].id, 1), Node(ccd[end].id,
        RecData(γ=ccd[end].id, e=id(root(model)))),
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
    b.state.γ == 0 ?
        (return addleafname!(b, "loss_$(hash(parent(b.node)))")) :
        b.state.t == 1 ?
            _backtrack!(b, b.model[b.state.e]) :
            _backtrack!(b)

# `n` is a ModelNode!
function _backtrack!(b, n)  # internode backtracking
    isleaf(n) ? (return addleafname!(b)) : nothing  # terminates recursion
    @unpack node, state, ccd = b
    @unpack e, γ, t = state
    r = rand()*getl(ccd, ccd.ℓ, e, γ, t)
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
    r = rand()*getl(ccd, ccd.ℓ, e, γ, t)
    r -= getϕ(model[e], t) * getl(ccd, ccd.ℓ, e, γ, t-1)
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
        r -= p * getl(ccd, ℓ, e, γ1, t-1) * 
                 getl(ccd, ℓ, e, γ2, t-1) * 
                 getψ(model[e], t)
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
        r -= p * getl(ccd, ℓ, f, γ1) * getl(ccd, ℓ, g, γ2)
        if r < 0.
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(g, γ2, tg)])
        end
        r -= p * getl(ccd, ℓ, g, γ1) * getl(ccd, ℓ, f, γ2)
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
    r -= getl(ccd, ℓ, f, γ)*getϵ(m[2])
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(g)])
    end
    r -= getl(ccd, ℓ, g, γ)*getϵ(m[1])
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
    r -= (1. - q + 2q*getϵ(m[1])) * getl(ccd, ccd.ℓ, f, γ)
    return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(f)])
end

function wgdretention(r, b, m)
    @unpack state, ccd, model = b
    @unpack ℓ = ccd
    @unpack e, γ, t, = state
    @unpack q = getθ(model.rates, m)
    f = id(m[1])
    tf = lastslice(m[1])
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= q * p * getl(ccd, ℓ, f, γ1) * getl(ccd, ℓ, f, γ2)
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
        r -= p * getl(ccd, ℓ, e, γ1, t) * 
                 getl(ccd, ℓ, e, γ2, t) * 
                 √(one(η)/η_)*(one(η)-η)
        if r < zero(r)
            return (r=r, next=[SliceState(e, γ1, t), SliceState(e, γ2, t)])
        end
        # or speciate
        r -= p * getl(ccd, ℓ, f, γ1) * getl(ccd, ℓ, g, γ2) * η_
        if r < zero(r)
            return (r=r, next=[SliceState(f, γ1, tf), SliceState(g, γ2, tg)])
        end
        r -= p * getl(ccd, ℓ, g, γ1) * getl(ccd, ℓ, f, γ2) * η_
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
    r -= getl(ccd, ℓ, f, γ) * getϵ(m[2]) * η_
    if r < 0.
        return (r=r, next=[SliceState(f, γ, lastslice(m[1])), Loss(g)])
    end
    r -= getl(ccd, ℓ, g, γ) * getϵ(m[1]) * η_
    if r < 0.
        # XXX (*):  state.node.kind = :sploss
        return (r=r, next=[SliceState(g, γ, lastslice(m[2])), Loss(f)])
    end
    return (r=r, next=SliceState[])
end

# NOTE: we should set the parent node at the relevant occasion (see e.g. * ↑)
