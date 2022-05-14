# Rewrite of the simulator
Random.randexp(λ) = -log(rand())/λ

# Use the same struct as in `track.jl`
"""
    simulate([rng,] M::WhaleModel; prune=:rec)

Simulate reconstructed trees from a WhaleModel, taking into account the
conditioning. Returns a gene tree, profile and the number of rejected
simulations. Using `prune=:rec` will return a reconciled tree, `prune=:full`
will return a gene tree, and any other setting will return the tree as
simulated (i.e. including extinct lineages).
"""
simulate(M::WhaleModel; kwargs...) = simulate(Random.default_rng(), M; kwargs...)
function simulate(rng::AbstractRNG, M::WhaleModel; prune=:rec, kwargs...)
    G, p, n = simulate_conditional(rng, M; kwargs...)
    G = prune == :rec ? recprune!(G) : (prune == :full ? fullprune!(G) : G)
    return G, p, n
end

simulate(M::WhaleModel, n; kwargs...) = simulate(Random.default_rng(), M, n; kwargs...)
function simulate(rng::AbstractRNG, M::WhaleModel, n; kwargs...)
    xs = map(x->simulate(rng, M; kwargs...), 1:n)
    first.(xs), vcat(DataFrame.(getindex.(xs, Ref(2)))...), last.(xs)
end

function simulate_conditional(rng::AbstractRNG, M::WhaleModel; minn=3)
    L = name.(getleaves(getroot(M)))
    i = 0
    while true
        G = randtree(rng, M)
        p = profile(G, L)
        n = sum(values(p))
        (satisfies_condition(M.condition, M, p) && n ≥ minn) && return G, p, i
        i += 1
    end
end

function satisfies_condition(c::RootCondition, M, profile)
    o = getroot(M)
    a = sum([profile[k] for k in name.(getleaves(o[1]))])
    b = sum([profile[k] for k in name.(getleaves(o[2]))])
    a > 0 && b > 0
end

function satisfies_condition(c::NowhereExtinctCondition, M, profile)
    all(collect(values(profile)) .> 0)
end

profile(G, M) = profile(G, name.(getleaves(getroot(M))))
function profile(G, sleaves::Vector)
    m = countmap(name.(getleaves(G)))
    Dict(k=>haskey(m, k) ? m[k] : 0 for k in sleaves)
end

"""
    randtree([rng,] M::WhaleModel)

Simulate from the WhaleModel without conditioning on any form of
non-extinction.
"""
randtree(M::WhaleModel) = randtree(Random.default_rng(), M)
function randtree(rng::AbstractRNG, M::WhaleModel)
    a = rand(rng, Geometric(M.rates.η))+1
    o = getroot(M)
    γ = zero(id(o))
    A = map(1:a) do ancestral
        γ += one(γ)
        n = newnode(γ, id(o), "speciation")
        γ = dlsim!(n, 0., o, M.rates)
        n
    end
    # resolve the root
    while length(A) > 1
        γ += one(γ)
        a = pop!(A)
        b = pop!(A)
        n = newnode(γ, id(o), "duplication")
        push!(n, a, b)
        a.parent = n
        b.parent = n
        push!(A, n)
    end
    return A[1]
end

newnode(γ, e) = Node(γ, RecData(γ=γ, e=e, t=0.))
newnode(γ, e, n::RecNode) = Node(γ, RecData(γ=γ, e=e, t=0.), n)
newnode(γ, e, l::String) = Node(γ, RecData(γ=γ, e=e, t=0., label=l))
newnode(γ, e, l::String, n::RecNode) = Node(γ, RecData(γ=γ, e=e, t=0., label=l), n)

function dlsim!(n::RecNode{I,T}, t, e, rates) where {I,T}
    @unpack λ, μ = getθ(rates, e)
    w = randexp(λ+μ)
    t -= w
    γ = T(n.data.γ + 1)
    if t > zero(t)
        n.data.t += w
        if rand() < λ/(λ+μ)  # dup
            n.data.label = "duplication"
            γ = dlsim!(newnode(γ, id(e), n), t, e, rates)
            γ = dlsim!(newnode(γ, id(e), n), t, e, rates)
        else  # loss
            n.data.label = "loss"
            return γ
        end
    else
        n.data.t += t + w
        n.data.label = "speciation"
        isleaf(e) && (n.data.name = name(e))
        # if next is wgd -> wgd model
        if iswgd(e)
            @unpack q = getθ(rates, e)
            f = e[1]
            d = distance(f)
            if rand() < q  # retention
                n.data.label = "wgd"
                γ = dlsim!(newnode(γ, id(f), n), d, f, rates)
                γ = dlsim!(newnode(γ, id(f), n), d, f, rates)
            else  # non-retention
                n.data.label = "wgdloss"
                γ = dlsim!(newnode(γ, id(f), n), d, f, rates)
            end
        else
            for c in children(e)
                γ = dlsim!(newnode(γ, id(c), n), distance(c), c, rates)
            end
        end
    end
    return γ
end

"""
    fullprune

Prune loss nodes from the tree until we get the would-be observed gene tree.
"""
fullprune(tree) = fullprune!(deepcopy(tree))
function fullprune!(n)
    for node in postwalk(n)
        isroot(node) && continue
        (isleaf(node) && name(node) != "") && continue
        l = getlabel(node)
        if isleaf(node) && l ∈ ["loss", "duplication", "speciation", "wgd", "wgdloss"]
            delete!(parent(node), node)
            node.parent = node
        elseif degree(node) == 1
            p = parent(node)
            c = node[1]
            c.data.t += distance(node)
            delete!(p, node)
            push!(p, c)
            c.parent = p
        end
    end
    return prunefromroot!(n)
end

function prunefromroot!(n)
    while degree(n) < 2
        n = n[1]
        n.parent = n  # HACK
    end
    return n
end

"""
    recprune!(tree)

Prune and relabel a `RecNode` tree so that we get a reconciled tree as we would
have for a Whale gene tree reconciliation for gene tree data.
Note that the labeling must correspond exactly to the labeling in `track` if we 
want to use this for e.g. posterior predictive simulations.
"""
recprune(tree) = recprune!(deepcopy(tree))

# this is tricky, we want to have it exactly as in `track`...
function recprune!(n)
    for node in postwalk(n)
        isroot(node) && continue
        (isleaf(node) && name(node) != "") && continue
        l = getlabel(node)
        if isleaf(node) && l ∈ ["loss", "duplication", "speciation", "wgd", "wgdloss"]
            # this happens when everything below is extinct
            node.data.label = "loss"
            # we don't delete the node just yet when the parent is a speciation
            getlabel(parent(node)) == "speciation" && continue
            # if parent is wgdloss, we prune, if speciation/wgd, we need to
            # check later
            delete!(parent(node), node)
            node.parent = node
        elseif degree(node) == 2
            # check what's below
            l1 = getlabel(node[1])
            l2 = getlabel(node[2])
            if l1 == l2 == "loss"  # remove node
                delete!(parent(node), node)
                node.parent = node
            elseif l1 == "loss" || l2 == "loss"
                node.data.label = "sploss"
            end
        elseif degree(node) == 1 && l == "wgd"
            node.data.label = "wgdloss"
        elseif degree(node) == 1 && l != "wgdloss"
            p = parent(node)
            c = node[1]
            c.data.t += distance(node)
            delete!(p, node)
            push!(p, c)
            c.parent = p
        end
    end
    return prunefromroot!(n)
end

# run ALEobserve
# assume single tree per family
function aleobserve(trees::Vector{<:Node}; outdir="/tmp/ccd$(rand())")
    mkpath(outdir)
    for (i,t) in enumerate(trees)
        nw = joinpath(outdir, "$i.nw")
        writenw(joinpath(outdir, "$i.nw"), t)
        run(pipeline(`ALEobserve $nw`, devnull))
        rm(nw)
    end
    return outdir  # handy for `read_ale`
end
