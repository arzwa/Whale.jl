# simulation of profiles and trees from the DL+WGD model (with prior on root)

profile_fromtree(t::RecTree, s::SlicedTree) =
    countmap([s.leaves[t.σ[n]] for n in keys(t.leaves)])

profile_fromtree(t::RecTree) = countmap([t.σ[n] for n in keys(t.leaves)])

function Base.rand(m::WhaleModel)
    @unpack S, λ, μ, q, η = m
    Ψ, extant = initialize_sim(η)
    i = 1
    function simulate_tree!(e, extant)
        for n in extant
            Ψ.σ[n] = e
            Ψ.labels[n] = iswgd(S, e) ? "wgd" : "speciation"
        end
        if isleaf(S, e)
            for n in extant
                Ψ.leaves[n] = "$(S.leaves[e])_$i"
                i += 1
            end
            return
        else
            for f in childnodes(S, e)
                t = distance(S, e, f)
                new_extant = Int64[]
                for u in extant
                    λf = λ[S[f, :λ]]
                    μf = μ[S[f, :μ]]
                    Ψ, extant_ = dlsim_branch!(Ψ, u, λf, μf, t, f)
                    new_extant = [new_extant ; extant_]
                    if iswgd(S, e) && rand() < q[S[e, :q]]
                        T, extant_ = dlsim_branch!(Ψ, u, λf, μf, t, f)
                        new_extant = [new_extant ; extant_]
                    end

                end
                simulate_tree!(f, new_extant)
            end
        end
    end
    simulate_tree!(findroot(S), extant)
    return Ψ
end

function initialize_sim(η)
    Ψ = RecTree()
    addnode!(Ψ.tree)             # initialize root
    extant = [1]                 # records all lineges that are currently extant
    nroot = rand(Geometric(η)) + 1        # lineages at the root
    n = 1                        # node counter
    for i in 2:nroot
        source = pop!(extant)
        for j in 1:2
            n += 1
            addnode!(Ψ.tree)
            addbranch!(Ψ.tree, source, n, 1.)  # branch lengths are meaningless
            push!(extant, n)
            Ψ.σ[n] = 1
            Ψ.labels[n] = "duplication"
        end
    end
    return Ψ, extant
end

function dlsim_branch!(T::RecTree, u::Int64, λ::Float64, μ::Float64,
        t::Float64, label::Int64)
    W = Exponential(1 / (λ + μ))
    waiting_time = rand(W)
    t -= waiting_time
    if t > 0
        # birth
        if rand() < λ / (λ + μ)
            addnode!(T.tree)
            v = maximum(keys(T.tree.nodes))
            addbranch!(T.tree, u, v, waiting_time)
            T.σ[v] = label
            T.labels[v] = "duplication"
            T, l = dlsim_branch!(T, v, λ, μ, t, label)
            T, r = dlsim_branch!(T, v, λ, μ, t, label)
            return T, [l ; r]
        # death
        else
            addnode!(T.tree)
            v = maximum(keys(T.tree.nodes))
            addbranch!(T.tree, u, v, waiting_time)
            T.σ[v] = label
            T.labels[v] = "loss"
            return T, []
        end
    else
        addnode!(T.tree)
        v = maximum(keys(T.tree.nodes))
        addbranch!(T.tree, u, v, t + waiting_time)
        return T, [v]
    end
end


# Simulate from posterior
"""
    simulate(w::WhaleChain, n::Int64, burnin::Int64=1000)

Simulate gene trees from the posterior for a WhaleChain. Note: this simulates
gene families from the posterior model of gene family evolution, not gene trees
for the input families (see `backtrack` and related functionalities to sample
reconciled trees from the posterior).
"""
function simulate(w::WhaleChain, n::Int64, burnin=1000)
    @unpack S, df = w
    simulate(df, S, n, burnin)
end

function simulate(df::DataFrame, s::SlicedTree, n::Int64, burnin=1000;
        non_extinct=true)
    r = burnin:size(df, 1)
    i = 1
    ts = RecTree[]
    while i < n
        t = rand(getstate(s, df[rand(r),:]))
        if length(t.leaves) > 0 || !non_extinct
            push!(ts, t)
            i += 1
        end
    end
    ts
end


"""
    summarize(s::SlicedTree, tree::Array{RecTree,1}, event="duplication")

Summarize events for eac hbranch of the species tree in a bunch of reconciled
trees.
"""
function summarize(s, trees, event="duplication")
    m = length(s.tree.nodes)
    n = length(trees)
    M = zeros(Int64, n, m)
    for (i, t) in enumerate(trees)
        for (k,v) in t.labels
            if v == event
                M[i, t.σ[k]] += 1
            end
        end
    end
    names = [Symbol(join(leafset(s, k), ";")) for (k,v) in s.tree.nodes]
    df = DataFrame(M, names, makeunique=true)
end
