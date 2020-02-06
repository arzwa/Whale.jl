
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

Base.show(io::IO, s::SliceState) = write(io, "$(Int64.((s.e, s.γ, s.t)))")

struct BackTracker{I}
    state::SliceState{I}
    node ::RecNode{I}
    model::WhaleModel
    ccd  ::CCD
end

function Base.:-(b::BackTracker)
    @unpack e, γ, t = b.state
    BackTracker(SliceState(e, γ, t-1), b.node, b.model, b.ccd)
end

BackTracker(model::WhaleModel, ccd::CCD) = BackTracker(
    SliceState(model[1].id, ccd[end].id, 1),
    RecNode(γ=ccd[end].id, rec=model[1].id), model, ccd)

# We use 'the abstract types can be an implementation detail' technique I
# once read in a blog post by Tamas Papp, basically non-wgds (root speciation)
# are handled by dipatching on the abstract WhaleNode, wgds by dispatch on the
# concrete subtype.
# non-bifurcations
function update(b::BackTracker, newstate::SliceState, n::WhaleNode{T,WGD{T}}) where T
    @unpack γ, e = newstate
    newnode = RecNode(γ=γ, rec=e, kind=:wgdloss, parent=b.node)
    BackTracker(newstate, newnode, b.model, b.ccd)
end

function update(b::BackTracker, newstate::SliceState, n::WhaleNode)
    @unpack γ, e = newstate
    newnode = RecNode(γ=γ, rec=e, kind=:sploss, parent=b.node)
    BackTracker(newstate, newnode, b.model, b.ccd)
end

# bifurcations
function update(b::BackTracker, newstates::Vector, n::WhaleNode{T,WGD{T}}) where T
    next = similar(newstates, BackTracker)
    for (i, newstate) in enumerate(newstates)
        @unpack e, γ = newstate
        newnode = RecNode(γ=γ, rec=e, kind=:wgd, parent=b.node)
        next[i] = BackTracker(newstate, newnode, b.model, b.ccd)
    end
    next
end

function update(b::BackTracker, newstates::Vector, n::WhaleNode{T,Speciation{T}}) where T
    next = similar(newstates, BackTracker)
    for (i, newstate) in enumerate(newstates)
        @unpack e, γ = newstate
        newnode = RecNode(γ=γ, rec=e, kind=:sp, parent=b.node)
        next[i] = BackTracker(newstate, newnode, b.model, b.ccd)
    end
    next
end

function update(b::BackTracker, newstates::Vector, n::WhaleNode{T,Root{T}}) where T
    next = similar(newstates, BackTracker)
    for (i, newstate) in enumerate(newstates)
        @unpack e, γ = newstate
        newnode = RecNode(γ=γ, rec=e, kind=:duplication, parent=b.node)
        next[i] = BackTracker(newstate, newnode, b.model, b.ccd)
    end
    next
end

function update(b::BackTracker, newstates::Vector)
    next = similar(newstates, BackTracker)
    for (i, newstate) in enumerate(newstates)
        @unpack e, γ = newstate
        newnode = RecNode(γ=γ, rec=e, kind=:duplication, parent=b.node)
        next[i] = BackTracker(newstate, newnode, b.model, b.ccd)
    end
    next
end

# do the backtracking (exported function?)
backtrack(wm::WhaleModel, ccd) = backtrack!(BackTracker(wm, ccd))

# dispatch on node type
function backtrack!(b::BackTracker)
    @show b.state
    if b.state.t == 1
        backtrack!(b, b.model[b.state.e])
    else
        backtrack_intra!(b)
    end
end

# root backtracking
function backtrack!(b::BackTracker, n::WhaleNode{T,Root{T}}) where T
    @unpack node, state, ccd = b
    @unpack e, γ, t = state
    p = ccd.ℓtmp[e][γ,t]
    r = node.kind == :root ? rand()*p/n.event.η : rand()*p
    η_ = 1.0/(1. - (1. - n.event.η) * getϵ(n))^2
    if !isleaf(ccd[γ])  # bifurcating events
        @unpack r, next = rootbifurcation(r, b, n, n.event.η, η_)
        if r < 0.0
            newbs = update(b, next, n)
            return backtrack!.(newbs)
        end
    end
    # loss
    @unpack r, next = rootloss(r, b, n, η_)
    if r < 0.0
        newb = update(b, next, n)
        return backtrack!(newb)
    else
        error("Backtrace failed ($n), could not obtain latent state, $s")
    end
end

# speciation
function backtrack!(b::BackTracker, n::WhaleNode{T,Speciation{T}}) where T
    if isleaf(n)
        return b
    end
    @unpack node, state, ccd = b
    @unpack e, γ, t = state
    r = rand()*ccd.ℓtmp[e][γ,t]
    @unpack r, next = sploss(r, b, n)
    if r < 0.
        newb = update(b, next, n)
        backtrack!(newb)
    elseif isleaf(ccd[γ])
        error("Backtrace failed $(n), isleaf(γ) == true but no sp+loss")
    end
    @unpack r, next = speciation(r, b, n)
    if r < 0.
        newbs = update(b, next, n)
        return backtrack!.(newbs)
    else
        error("Backtrace failed ($n), could not obtain latent state, $s")
    end
end

function backtrack!(b::BackTracker, m::WhaleNode{T,WGD{T}}) where T
end

# intra branch backtracking
function backtrack_intra!(b::BackTracker)
    @unpack state, ccd, model = b
    @unpack e, γ, t = state
    if isleaf(ccd[γ])
        newstate = SliceState(e, γ, 1)
        return backtrack!(BackTracker(newstate, b.node, model, ccd))
    end
    r = rand()*ccd.ℓtmp[e][γ,t]
    r -= getϕ(model[e], t) * ccd.ℓtmp[e][γ,t-1]
    if r < 0.
        backtrack!(-b)
    end
    @unpack r, next = duplication(r, b)
    if r < 0.
        newbs = update(b, next)
        return backtrack!.(newbs)
    else
        error("Backtrace failed ($r), could not obtain latent state, $state")
    end
end

function duplication(r, b::BackTracker)
    @unpack state, ccd, model = b
    @unpack e, γ, t = state
    nextsp = nonwgdchild(model[e], model)
    @unpack λ, μ = nextsp.event
    Δt = model[e].slices[t,1]
    ℓ = ccd.ℓtmp
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        r -= p * ℓ[e][γ1,t-1] * ℓ[e][γ2,t-1] * pdup(λ, μ, Δt)
        if r < 0.
            return (r=r, next=[SliceState(e, γ1, t-1), SliceState(e, γ2, t-1)])
        end
    end
    return (r=r, next=SliceState[])
end

function speciation(r, b::BackTracker, m::WhaleNode)
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

function sploss(r, b::BackTracker, m::WhaleNode)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    ℓ = ccd.ℓtmp
    f, g = m.children
    r -= ℓ[f][γ,end]*getϵ(model[g])
    if r < 0.
        return (r=r, next=SliceState(f, γ, lastslice(model, f)))
    end
    r -= ℓ[g][γ,end]*getϵ(model[f])
    if r < 0.
        return (r=r, next=SliceState(g, γ, lastslice(model, g)))
    end
    return (r=r, next=SliceState[])
end

function rootbifurcation(r, b::BackTracker, m::WhaleNode, η, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    ℓ = ccd.ℓtmp
    f, g = m.children
    tf = lastslice(model, f)
    tg = lastslice(model, g)
    for triple in ccd[γ].splits
        @unpack p, γ1, γ2 = triple
        # either stay in root and duplicate
        r -= p * ℓ[e][γ1,t] * ℓ[e][γ2,t] * √(1.0/η_)*(1.0-η)
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

function rootloss(r, b::BackTracker, m::WhaleNode, η_)
    @unpack state, ccd, model = b
    @unpack e, γ, t, = state
    ℓ = ccd.ℓtmp
    f, g = m.children
    r -= ℓ[f][γ,end] * getϵ(model[g]) * η_
    if r < 0.
        return (r=r, next=SliceState(f, γ, lastslice(model, f)))
    end
    r -= ℓtmp[g][γ,end] * getϵ(model[f]) * η_
    if r < 0.
        return (r=r, next=SliceState(g, γ, lastslice(model, g)))
    end
    return (r=r, next=SliceState[])
end
