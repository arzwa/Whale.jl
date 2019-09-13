# Recursive stochastic backtracking of reconciled trees
"""
    backtrack!(ccd::CCD, w::WhaleModel, n=1)
    backtrack!(D::CCDArray, w::WhaleModel, n=1)
    backtrack!(D::CCDArray, w::WhaleChain, n=1000)
    backtrack!(D::CCDArray, df::DataFrame, st::SlicedTree, n=1000)

Sample reconciled trees (`RecTree`) from the Whale model conditional on the
conditional clade distribution (CCD) by stochastic backtracking along the
dynamic programming matrix as in Szollosi 2013. If the `recmat` field of the
`ccd` argument is non-empty, this matrix will be used, otherwise the
reconciliation matrix is computed from scratch. Reconciled trees are stored in
the CCD object for convenience.

!!! note
    If the second argument is a `WhaleChain` object (or data frame), than *post
    hoc* backtracking is performed, that is, reconciled trees are simulated
    from the posterior predictive distribution after running an MCMC chain.
"""
function backtrack!(ccd::CCD, w::WhaleModel, n=1)
    if length(ccd.recmat) == 0
        logpdf(w, ccd, matrix=true)
        ccd.recmat = ccd.tmpmat
    end
    for i=1:n
        R = RecTree()
        backtrack_root!(R, -1, ccd.Γ, ccd, w)
        push!(ccd.rectrs, R)
    end
    return ccd
end

backtrack!(D::CCDArray, w::WhaleModel, n=1) =
    map!((x)->backtrack!(x, w, n), D, D)

backtrack!(D::CCDArray, w::WhaleChain, n=1000) = backtrack!(D, w.df, w.S, n)

function backtrack!(D::CCDArray, df::DataFrame, st::SlicedTree, n=1000)
    @showprogress 1 "Sampling from from posterior predictive" for i=1:n
        w = getstate(st, df[rand(1:size(df, 2)),:])
        l = logpdf(w, D, matrix=true)
        set_recmat!(D)
        backtrack!(D, w)
    end
end

# Main backtracker
# The whole thing could be better integrated with the core Whale algorithm
function backtrack!(R::RecTree, node::Int64, γ::Int64, e::Int64, i::Int64,
        ccd::CCD, w::WhaleModel)
    if e == 1
        return backtrack_root!(R, node, γ, ccd, w)
    elseif i == 1
        return backtrack_inter!(R, node, γ, e, ccd, w)
    else
        return backtrack_intra!(R, node, γ, e, i, ccd, w)
    end
end

# Intra-branch backtracking
# =========================
function backtrack_intra!(R::RecTree, node::Int64, γ::Int64, e::Int64,
        i::Int64, ccd::CCD, w::WhaleModel)
    if haskey(ccd.m3, γ)  # leaf clade, cannot duplicate
        return backtrack!(R, node, γ, e, i-1, ccd, w)
    end
    p = ccd.recmat[e, γ, i]
    r = rand() * p
    λ = w.λ[w.S[e, :λ]]
    Δt = w.S[e, i]

    # propogation
    r -= w.ϕ[e, i] * ccd.recmat[e, γ, i-1]
    if r < 0.
        return backtrack!(R, node, γ, e, i-1, ccd, w)
    end

    # duplication
    r, γ1, γ2 = duplication(r, ccd, γ, e, i, λ, Δt)
    if r < 0.
        dnode = add_duplication_node!(R, e, node, γ, ccd.blens[γ])
        backtrack!(R, dnode, γ1, e, i-1, ccd, w)
        backtrack!(R, dnode, γ2, e, i-1, ccd, w)
        return
    else
        error("Backtracking failed (intra): r = $r; γ = $γ; e = $e")
    end
end

# duplication events
function duplication(r, ccd, γ, e, i, λ, Δt)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        # here the factor 2 or BDP issue comes up again
        r -= ccp * ccd.recmat[e, γ1, i-1] * ccd.recmat[e, γ2, i-1] * λ * Δt
        r < 0. ? (return r, γ1, γ2) : nothing
    end
    return r, -1, -1
end

# Inter-branch backtracking (speciation/WGD nodes)
# ================================================
function backtrack_inter!(R::RecTree, node::Int64, γ::Int64, e::Int64,
        ccd::CCD, w::WhaleModel)
    if isleaf(w.S, e) && isleaf(ccd, γ)
        # leaf node in both species and gene tree
        add_leaf_node!(R, e, node, γ, ccd.blens[γ], ccd.leaves[γ])
        return node  # ends recursion
    elseif length(childnodes(w.S, e)) == 2
        # speciation node
        return bt_sp!(R, node, γ, e, ccd, w)
    elseif length(childnodes(w.S, e)) == 1
        # wgd node
        return bt_wgd!(R, node, γ, e, ccd, w)
    else
        # happens when we're in a leaf branch, at time 1, but not in leaf clade!
        error("Backtracking failed (inter, γ ≠ leaf, e = leaf): γ = $γ, e = $e")
        return  # stops recursion, REALLY shouldn't occur
    end
end

# backtrack at bifurcating node in S (speciation)
function bt_sp!(R::RecTree, node::Int64, γ::Int64, e::Int64, ccd::CCD,
        w::WhaleModel)
    p = ccd.recmat[e, γ, 1]
    r = rand() * p
    f, g = childnodes(w.S, e)

    # speciation + loss event
    r, bs, bl = non_bifurcation(r, ccd, γ, w.ε, f, g)
    if r < 0.
        spnode = add_speciation_node!(R, e, node, γ, ccd.blens[γ])
        add_loss_node!(R, bl, spnode)
        return backtrack!(R, spnode, γ, bs, nslices(w.S, bs), ccd, w)
    end

    # speciation event
    if !(haskey(ccd.m2, γ))
        # if γ is a leaf, it should have gone through an SL event
        error("Backtracking failed (inter, γ = leaf): r = $r; γ = $γ; e = $e")
    end
    r, γ1, γ2, b1, b2 = bifurcation(r, ccd, γ, f, g)
    if r < 0.
        spnode = add_speciation_node!(R, e, node, γ, ccd.blens[γ])
        backtrack!(R, spnode, γ1, b1, nslices(w.S, b1), ccd, w)
        backtrack!(R, spnode, γ2, b2, nslices(w.S, b2), ccd, w)
        return
    else
        error("Backtracking failed (inter): r = $r; γ = $γ; e = $e")
    end
end

# backtrack at non-bifurcating node in S (wgd)
function bt_wgd!(R::RecTree, node::Int64, γ::Int64, e::Int64, ccd::CCD,
        w::WhaleModel)
    p = ccd.recmat[e, γ, 1]
    r = rand() * p
    f = childnodes(w.S, e)[1]
    q = w.q[w.S[e, :q]]

    # add wgd node
    # wgd_node = add_wgd_node!(R, e, node, γ, ccd.blens[γ])

    # non-retention/loss, currently no loss node added
    r, γ = wgd_non_bifurcation(r, ccd, γ, w.ε[f][end], f, q)
    #r -= wgd_nonretention(γ, q, f, ccd) + wgd_loss(γ, q, w.ε[f][end], f, ccd)
    if r < 0.
        #add_loss_node!(R, f, wgd_node)
        # return backtrack!(R, wgd_node, γ, f, nslices(w.S, f), ccd, w)
        return backtrack!(R, node, γ, f, nslices(w.S, f), ccd, w)
    end

    # retention
    r, γ1, γ2 = wgd_retention(r, ccd, γ, f, q)
    if r < 0.
        wgd_node = add_wgd_node!(R, e, node, γ, ccd.blens[γ])
        backtrack!(R, wgd_node, γ1, f, nslices(w.S, f), ccd, w)
        backtrack!(R, wgd_node, γ2, f, nslices(w.S, f), ccd, w)
        return
    else
        error("Backtracking failed (WGD): r = $r; γ = $γ; e = $e")
    end
end

# wgd retention
function wgd_retention(r, ccd, γ, f, q)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        r -= q * ccp * ccd.recmat[f][γ1, end] * ccd.recmat[f][γ2, end]
        if r < 0.
            return r, γ1, γ2
        end
    end
end

# wgd non-retention/loss
function wgd_non_bifurcation(r, ccd, γ, ε, f, q)
    r -= (1. - q + 2q*ε) * ccd.recmat[f][γ, end]
    return r, γ
end

# speciation + loss events
function non_bifurcation(r, ccd, γ, ε, f, g)
    r -= ccd.recmat[f][γ, end] * ε[g][end]
    if r < 0.
        return r, f, g
    end
    r -= ccd.recmat[g][γ, end] * ε[f][end]
    if r < 0.
        return r, g, f
    end
    return r, f, g
end

# speciation events
function bifurcation(r, ccd, γ, f, g)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]
        r -= ccp * ccd.recmat[f][γ1, end] * ccd.recmat[g][γ2, end]
        if r < 0.
            return r, γ1, γ2, f, g
        end
        r -= ccp * ccd.recmat[g][γ1, end] * ccd.recmat[f][γ2, end]
        if r < 0.
            return r, γ1, γ2, g, f
        end
    end
    return r, -1, -1, f, g
end

# Root backtracker
# ================
# The root probability for clade γ is ~ Π_root + η_*Π_spec + η_*Π_spec + loss
function backtrack_root!(R::RecTree, node::Int64, γ::Int64, ccd::CCD,
        w::WhaleModel)
    p = node == -1 ? ccd.recmat[1][γ,1] / w.η : ccd.recmat[1][γ, 1]
    r = rand() * p
    e0 = w.ε[1, 1]
    η_ = 1.0/(1. - (1. - w.η) * e0)^2
    root = findroot(w.S)
    f, g = childnodes(w.S, root)

    # bifurcating events
    if !haskey(ccd.m3, γ)
        r, γ1, γ2, b1, b2 = root_bifurcation(r, ccd, γ, w.η, η_, f, g)
        if r < 0. && b1 == b2 == root
            n = add_duplication_node!(R, root, node, γ, ccd.blens[γ])
            backtrack_root!(R, n, γ1, ccd, w)
            backtrack_root!(R, n, γ2, ccd, w)
            return
        elseif r < 0. && b1 != b2
            spnode = add_speciation_node!(R, root, node, γ, ccd.blens[γ])
            backtrack!(R, spnode, γ1, b1, nslices(w.S, b1), ccd, w)
            backtrack!(R, spnode, γ2, b2, nslices(w.S, b2), ccd, w)
            return
        end
    end

    # non-bifurcating events (speciation + loss)
    r, bs, bl = root_nonbifurcation(r, ccd, γ, η_, w.ε, f, g)
    # bs = branch where species is represented, bl = branch with loss
    if r < 0.
        spnode = add_speciation_node!(R, root, node, γ, ccd.blens[γ])
        add_loss_node!(R, bl, spnode)
        return backtrack!(R, spnode, γ, bs, nslices(w.S, bs), ccd, w)
    else
        error("Backtracking failed (root): r = $r; γ = $γ")
    end
end

# a bifurcation at (speciation) or before (duplication) the root
function root_bifurcation(r, ccd, γ, η, η_, f, g)
    for (γ1, γ2, count) in ccd.m2[γ]
        ccp = ccd.ccp[(γ, γ1, γ2)]

        # either stay in root and duplicate
        r -= ccp * ccd.recmat[1,γ1,1] * ccd.recmat[1,γ2,1] * √(1.0/η_)*(1.0-η)
        r < 0. ? (return r, γ1, γ2, 1, 1) : nothing

        # or speciate
        r -= ccp * ccd.recmat[f][γ1, end] * ccd.recmat[g][γ2, end] * η_
        r < 0. ? (return r, γ1, γ2, f, g) : nothing

        r -= ccp * ccd.recmat[g][γ1, end] * ccd.recmat[f][γ2, end] * η_
        r < 0. ? (return r, γ1, γ2, g, f) : nothing
    end
    return r, -1, -1, g, f
end

# speciation + loss at the root (this should only be possible when there
# are multiple lineages at the root)
function root_nonbifurcation(r, ccd, γ, η_, ε, f, g)
    r -= ccd.recmat[f][γ, end] * ε[g][end] * η_
    r < 0. ? (return r, f, g) : nothing
    r -= ccd.recmat[g][γ, end] * ε[f][end] * η_
    r < 0. ? (return r, g, f) : nothing
    return r, f, g
end

# Rectree functions
function add_loss_node!(R, e, node)
    new_node = addnode!(R.tree)
    addbranch!(R.tree, node, new_node, 0.01)
    R.σ[new_node] = e
    R.labels[new_node] = "loss"
    return new_node
end

function add_duplication_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    node != -1 ? addbranch!(R.tree, node, new_node, b) : nothing
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "duplication"
    return new_node
end

function add_speciation_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    node != -1 ? addbranch!(R.tree, node, new_node, b) : nothing
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "speciation"
    return new_node
end

function add_wgd_node!(R, e, node, γ, b)
    new_node = addnode!(R.tree)
    addbranch!(R.tree, node, new_node, b)
    R.σ[new_node] = e
    R.γ[new_node] = γ
    R.labels[new_node] = "wgd"
    return new_node
end

function add_leaf_node!(R, e, node, γ, b, leaf_name)
    new_node = addnode!(R.tree)
    addbranch!(R.tree, node, new_node, b)
    R.γ[new_node] = γ
    R.σ[new_node] = e
    R.labels[new_node] = "leaf"
    R.leaves[new_node] = leaf_name
    return new_node
end
