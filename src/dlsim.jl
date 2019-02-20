# Arthur Zwaenepoel - 2018
# Gene tree topology simulation under the DL + WGD model.

# constant-rates rate index
constant_ri(S::SpeciesTree) = Dict(x => 1 for x in 1:length(S.tree.nodes))

"""
    dlsim(S, λ, μ, q, η)
Simulate one tree following the Dl + WGD model.
"""
function dlsim(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, ri::Dict{Int64,Int64})
    # initialize
    T, extant, labels = initialize(η)
    # recursion
    function simulate_tree!(e, extant)
        if isleaf(S.tree, e)
            for n in extant
                labels[n] = e
            end
            return
        end
        for f in childnodes(S.tree, e)
            t = distance(S.tree, e, f)
            new_extant = Int64[]
            for u in extant
                labels[u] = e
                λ_ = λ[ri[f]]
                μ_ = μ[ri[f]]
                T, extant_ = dlsim_branch!(T, u, λ_, μ_, t)
                new_extant = [new_extant ; extant_]
                if haskey(S.wgd_index, e) && rand() < q[S.wgd_index[e]]
                    @debug "WGD node retained below node $u"
                    T, extant_ = dlsim_branch!(T, u, λ_, μ_, t)
                    new_extant = [new_extant ; extant_]
                end
            end
            simulate_tree!(f, new_extant)
        end
    end
    simulate_tree!(1, extant)
    return T, labels
end

# dlsim with trees written to files
function dlsim(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
        η::Float64, ri::Dict{Int64,Int64}, N::Int64, outdir::String; oib::Bool=false,
        min::Int64=3, max=1000)
    @info "Simulating $N gene families ..."
    trees = dlsim(S, λ, μ, q, η, ri, N; oib=oib, min=min, max=max)
    for (T, ll) in trees
        open(joinpath(outdir, string(uuid1()) * ".nw"), "w") do f
            write(f, write_nw(T, ll) * "\n")
        end
    end
end

"""
    dlsim(S::SpeciesTree, λ::Float64, μ::Float64, q::Array{Float64},
          η::Float64, N::Int64; oib::Bool=false)
Simulate a bunch of duplication-loss model trees.
"""
function dlsim(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64}, q::Array{Float64},
        η::Float64, ri::Dict{Int64,Int64}, N::Int64; oib::Bool=false, min::Int64=3, max=1000)
    c = childnodes(S.tree, 1)
    left = sp_leaves(S.tree, c[1])
    right = sp_leaves(S.tree, c[2])
    n_invalid = 0
    i = 0
    trees = []
    while length(trees) < N
        T, labels = dlsim(S, λ, μ, q, η, ri)
        remove_loss_nodes!(T, S.tree, labels)
        remove_continuation_nodes!(T)
        if length(childnodes(T, 1)) == 1
            reset_root!(T, 1)
        end
        if oib
            if !(one_in_both_filter(values(labels), left, right))
                @debug "Not one in both clades!"
                continue
            end
        end
        ll = get_leaf_labels(labels, S.species)
        if  min <= length(findleaves(T)) <= max
            push!(trees, (T, ll))
            i += 1
        else
            @debug " .. tree has only $(length(T.nodes))"
            n_invalid += 1
        end
    end
    @debug " .. number of invalid trees: $n_invalid"
    return trees
end


"""
    dlsim_branch!(T::Tree, u::Int64, λ::Float64, μ::Float64, t::Float64)
Simulate the Kendall process over a species tree branch.
"""
function dlsim_branch!(T::Tree, u::Int64, λ::Float64, μ::Float64, t::Float64)
    W = Exponential(1 / (λ + μ))
    waiting_time = rand(W)
    t -= waiting_time
    if t > 0
        # birth
        if rand() < λ / (λ + μ)
            addnode!(T)
            v = maximum(keys(T.nodes))
            addbranch!(T, u, v, waiting_time)
            T, l = dlsim_branch!(T, v, λ, μ, t)
            T, r = dlsim_branch!(T, v, λ, μ, t)
            return T, [l ; r]
        # death
        else
            addnode!(T)
            v = maximum(keys(T.nodes))
            addbranch!(T, u, v, waiting_time)
            return T, []
        end
    else
        addnode!(T)
        v = maximum(keys(T.nodes))
        addbranch!(T, u, v, t + waiting_time)
        return T, [v]
    end
end

function dlsim(S::SpeciesTree, λ::Float64, μ::Float64, q::Array{Float64},
        η::Float64, N::Int64, outdir::String; oib::Bool=false)
    return dlsim(S, [λ], [μ], q, η, constant_ri(S), N, outdir; oib=oib)
end

function dlsim(S::SpeciesTree, λ::Float64, μ::Float64, q::Array{Float64},
        η::Float64)
    return dlsim(S, [λ], [μ], q, η, contant_ri(S))
end

function dlsim(S::SpeciesTree, λ::Array{Float64}, μ::Array{Float64},
        q::Array{Float64}, η::Float64, N::Int64, outdir::String; oib::Bool=false)
    return dlsim(S, λ, μ, q, η, get_rateindex(S), N, outdir, oib=oib)
end

function dlsim(S::SpeciesTree, prior::PriorSettings, N::Int64, outdir::String; oib=true)
    params = draw_from_prior(S, prior)
    dlsim(S, params, N, outdir, oib=oib)
    return params
end

function dlsim(S::SpeciesTree, params::Dict, N::Int64, outdir::String; oib=true, max=1000)
    @show params
    return dlsim(S, params["λ"], params["μ"], params["q"], params["η"][1], get_rateindex(S),
        N, outdir, oib=oib, max=max)
end

function convert_nw_ale(dir::String)
    for f in readdir(dir)
        nw = joinpath(dir, f)
        run(`ALEobserve $nw`)
        rm(nw)
    end
end

# Initialize the tree using a geometric prior on the number of lineages at the
# root
function initialize(η::Float64)
    T = Tree()
    addnode!(T)  # initialize root
    labels = Dict{Int64,Int64}()  # gene tree node to species node map
    extant = [1]  # records all lineges that are currently extant
    nroot = rand(Geometric(η)) + 1  # lineages at the root
    @debug "# lineages at root = $nroot"
    n = 1  # node counter
    for i in 1:nroot
        if i > 1
            source = pop!(extant)
            for j in 1:2
                addnode!(T) ; n += 1
                addbranch!(T, source, n, 0.1)
                push!(extant, n)
                labels[n] = 1
            end
        end
    end
    return T, extant, labels
end

# recursive function to get the leaves under a particular node
function sp_leaves(T, n)
    leaves = []
    function walk(n)
        if isleaf(T, n)
            push!(leaves, n)
            return
        end
        for c in childnodes(T, n)
            walk(c)
        end
    end
    walk(n)
    return Set(leaves)
end

# Filter on the condition that here should be at least on non-extinct lineages
# in both clades stemming from the root of the species tree.
function one_in_both_filter(labels, left, right)
    length(intersect(labels, left)) > 0 &&
        length(intersect(labels, right)) > 0 ? true : false
end

# Remove non-bifurcations across a tree
function remove_continuation_nodes!(T)
    for n in keys(T.nodes)
        if outdegree(T, n) == indegree(T, n) == 1
            p = parentnode(T, n)
            c = childnodes(T, n)[1]
            l = distance(T, n, p) + distance(T, n, c)
            deletebranch!(T, T.nodes[n].out[1])
            deletebranch!(T, T.nodes[n].in[1])
            addbranch!(T, p, c, l)
            deletenode!(T, n)
        end
    end
end

# Remove loss nodes across the tree.
function remove_loss_nodes!(T::Tree, S::Tree, labels::Dict{Int64,Int64})
    sp_leaves = findleaves(S)
    function walk(n)
        if n > 1 && loss_node(T, sp_leaves, labels, n)
            deletebranch!(T, T.nodes[n].in[1])
            deletenode!(T, n)
        else
            for c in childnodes(T, n)
                walk(c)
            end
            if loss_node(T, sp_leaves, labels, n) && n > 1
                deletebranch!(T, T.nodes[n].in[1])
                deletenode!(T, n)
            end
        end
    end
    walk(1)
end

# Check if a node is a loss node.
function loss_node(T, sp_leaves, labels, n)
    if outdegree(T, n) == 0
        if !(haskey(labels, n)) || !(labels[n] in sp_leaves)
            return true
        end
    end
    return false
end

# Get leaf labels for a simulated gene tree based on the species mapping.
function get_leaf_labels(labels, sp_labels)
    leaf_labels = Dict{Int64,String}()
    for (k, v) in labels
        if haskey(sp_labels, v)
             leaf_labels[k] = sp_labels[v] * "_" * string(k)
         end
     end
     return leaf_labels
end

# Given a current root ρ, this function will set the new root to the daughter
# node of ρ.
function reset_root!(T, ρ)
    real_root = childnodes(T, ρ)
    @assert length(real_root) == 1
    @assert length(T.nodes[ρ].out) == 1
    deletebranch!(T, T.nodes[ρ].out[1])
    for c in childnodes(T, real_root[1])
        dist = distance(T, c, real_root[1])
        deletebranch!(T, T.nodes[c].in[1])
        addbranch!(T, ρ, c, dist)
    end
    deletenode!(T, real_root[1])
end


function get_profile(nwdir::String, S::SpeciesTree)
    isdir(nwdir) ? nothing : @error "Not a directory $nwdir"
    profile = Dict{AbstractString,Int64}[]
    for f in readdir(nwdir)
        d = Dict(x => 0 for x in values(S.species))
        nw = open(joinpath(nwdir, f), "r") do file
            readlines(file)[1]
        end
        l = [split(x, "_")[1] for x in values(read_nw(nw)[2])]
        for s in l ; d[s] += 1 ; end
        push!(profile, d)
    end
    return profile
end


function write_profile(profile, fname::String)
    sp = collect(keys(profile[1]))
    open(fname, "w") do f
        write(f, ",", join(sp, ","), "\n")
        for (i, gf) in enumerate(profile)
            write(f, "$i,", join([gf[s] for s in sp], ","), "\n")
        end
    end
end
