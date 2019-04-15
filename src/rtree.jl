# Various tree manipulation routines. In particular related to handling reconciled trees
# © Arthur Zwaenepoel - 2019
"""
    read_nw(nw_str::String)
Read a tree from a newick string. This does not store bootstrap support values
anywhere.
"""
function read_nw(nw_str::String)
    if nw_str[end] != ';'
        error("Newick string does not end with ';'")
    end
    t = Tree(); leaves = Dict{Int64,String}(); stack = []
    i = 1; n = 1; bl = 0.0
    while i < length(nw_str)
        if nw_str[i] == '('
            # add an internal node to stack & tree
            push!(stack, n); n += 1
            addnode!(t)
        elseif nw_str[i] == ')'
            # add branch + collect data on node
            target = pop!(stack); source = stack[end]
            # @printf "%s ➞ %s\n" target source
            addbranch!(t, source, target, bl)
            sv, bl, i = get_node_info(nw_str, i+1)
        elseif nw_str[i] == ','
            # add branch
            target = pop!(stack); source = stack[end]
            # @printf "%s ➞ %s\n" target source
            addbranch!(t, source, target, bl)
        else
            # store leaf name and get branch length
            leaf, i = get_leaf_name(nw_str, i)
            sv, bl, i = get_node_info(nw_str, i)
            leaves[n] = leaf
            # add leaf node to stack & tree
            push!(stack, n); n += 1
            addnode!(t)
        end
        i += 1
    end
    return t, leaves
end

"""
    read_trees(file)
Read trees from a file.
"""
function read_trees(file_name::String)
    trees = LabeledTree[]
    open(file_name, "r") do f
        for ln in eachline(f)
            if ln[end] == ';'
                t, l = read_nw(ln)
                push!(trees, LabeledTree(t, l))
            end
        end
    end
    return trees
end

"""
    arbitrary_rooting(tree)
Resolve root trifurcation arbitrarily.
"""
function arbitrary_rooting!(t::Tree)
    c = childnodes(t, 1)
    d = distance(t, 1, c[3]) / 2
    addnode!(t)
    sub_root = maximum(keys(t.nodes))
    addbranch!(t, 1, sub_root, d)
    d1 = t.branches[t.nodes[c[1]].in[1]].length
    d2 = t.branches[t.nodes[c[2]].in[1]].length
    deletebranch!(t, t.nodes[c[3]].in[1])
    deletebranch!(t, t.nodes[c[2]].in[1])
    deletebranch!(t, t.nodes[c[1]].in[1])
    addbranch!(t, 1, c[3], d)
    addbranch!(t, sub_root, c[1], d1)
    addbranch!(t, sub_root, c[2], d2)
end

# get the last node added to the tree
last_node(t::Tree) = maximum(keys(t.nodes))

"""
    get_leaf_name(nw_str::String, i::Int64)
Get the leaf name that starts at index i in nw_str.
"""
function get_leaf_name(nw_str::String, i::Int64)
    j = i
    while (nw_str[j] != ':') & (nw_str[j] != ',') & (nw_str[j] != ')')
        j += 1
    end
    leaf = nw_str[i:j-1]
    return leaf, j
end

"""
    get_node_info(nw_str::String, i::Int64)
Get all info associated with the node before index i in nw_str.
"""
function get_node_info(nw_str, i)
    # get everything up to the next comma or semicolon
    substr = nw_str[i:end]
    substr = split(substr, ',')[1]
    substr = split(substr, ')')[1]
    if substr == ";"
        return -1.0, -1.0, i
    end
    substr = split(substr, ';')[1]
    if occursin(":", substr)
        sv, bl = split(substr, ':')
        if sv == ""  # only branch length
            bl = parse(Float64, bl); sv = 0.0
        else  # both (B)SV and branch length are there
            sv = parse(Float64, sv); bl = parse(Float64, bl)
        end
    else
        # nothing there
        bl = sv = 0.0
    end
    return sv, bl, i + length(substr) -1
end

"""
    lca_rec!(G::Tree, S::Tree, rec::Dict{Int64,Int64})
Least common ancestor (LCA) gene tree - species tree reconciliation, using the
algorithhm of Zmasek & Eddy (2001) which has a worst-case runtime of O(n^2).
Note that this returns sets of duplication and speciation nodes, not loss
events. Note this extends the rec dict to a full reconciliation.
"""
function lca_rec!(G::Tree, S::Tree, rec::Dict{Int64,Int64})
    if length(childnodes(G, 1)) > 2
        error("Can only perform LCA reconciliation for binary trees.")
    end
    node_index = preorder_ids(S)
    # do a post-order traversal and record a mapping from nodes in G to S
    # nodes of S should be numbered in preorder traversal I think they might
    # be as a result of read_nw but I'm not sure.
    node_labeling = Dict{Int64,String}()
    function walk(node)
        if isleaf(G, node)
            return node
        else
            children = childnodes(G, node)
            walk(children[1])
            walk(children[2])
            a = rec[children[1]]
            b = rec[children[2]]
            while node_index[a] != node_index[b]
                if node_index[a] > node_index[b]
                    a = parentnode(S, a)
                else
                    b = parentnode(S, b)
                end
            end
            rec[node] = a
            if rec[node] == rec[children[1]]
                node_labeling[node] = "D"
            elseif rec[node] == rec[children[2]]
                node_labeling[node] = "D"
            else
                node_labeling[node] = "S"
            end
        end
    end
    walk(1)
    return node_labeling
end

"""
    leaf_labeling(leaves_g::Dict, leaves_s::Dict, g2s::Dict)
Get a mapping from leaves in gene tree to leaves in species tree.
"""
function leaf_labeling(leaves_g, leaves_s, g2s)
    node2sp = Dict{Int64,Int64}()
    for kv in leaves_g
        sp = g2s[kv[1]]
        node2sp[kv[2]] = leaves_s[sp]
    end
    return node2sp
end

"""
    gene_to_species(genes)
Get a gene to species map automatically. Assumes species identifier is the
substring before the first '|' or '_' character.
"""
function gene_to_species(genes::Array{String})
    return Dict(x => String(split(x, ['|', '_'])[1]) for x in genes)
end

"""
    postorder_t(T::Tree)
Get a postorder of nodes in a tree, using a recursive algorithm.
"""
function postorder_t(T::Tree)
    l = []
    function walk(node)
        if PhyloTrees.isleaf(T, node)
            push!(l, node)
            return node
        else
            children = childnodes(T, node)
            for c in children
                walk(c)
            end
            push!(l, node)
        end
    end
    walk(1)
    return l
end

"""
    preorder_t(T::Tree)
Get a preorder of nodes in a tree, using a recursive algorithm.
"""
function preorder_t(tree)
    l = []
    function walk(node, depth=1)
        push!(l, node)
        if PhyloTrees.isleaf(tree, node)
            return
        end
        for c in childnodes(tree, node)
            walk(c, depth + 1)
        end
    end
    walk(findroots(tree)[1])
    return l
end

"""
    preorder_ids(T::Tree)
Get preorder IDs (indices).
"""
function preorder_ids(tree)
    ids = Dict{Int64,Int64}(); i = 1
    function walk(node, depth=1)
        ids[node] = i
        i += 1
        if isleaf(tree, node)
            return
        end
        for c in childnodes(tree, node)
            walk(c, depth + 1)
        end
    end
    walk(findroots(tree)[1])
    return ids
end

"""
    subtree_leaves(t, node)
Get the leaves of a subtree.
"""
function subtree_leaves(t::Tree, node::Int64)
    return intersect(Set(descendantnodes(t, node)), Set(findleaves(t)))
end

function subtree_leaves(rtree::RecTree, node::Int64)
    if isleaf(rtree.tree, node) && haskey(rtree.leaves, node)
        return [rtree.leaves[node]]
    end
    leafnodes = intersect(Set(descendantnodes(rtree.tree, node)), Set(findleaves(rtree.tree)))
    return [rtree.leaves[n] for n in leafnodes if haskey(rtree.leaves, n)]
end

function subtree_leaves(S::SpeciesTree, node::Int64)
    if isleaf(S.tree, node) && haskey(S.species, node)
        return [S.species[node]]
    end
    leafnodes = intersect(Set(descendantnodes(S.tree, node)), Set(findleaves(S.tree)))
    return [S.species[n] for n in leafnodes]
end

"""
    read_gene_tree(newick_file, species_tree)
Read a GeneTree directly from a file.
"""
function read_gene_tree(nw_file, S::SpeciesTree)
    f = open(nw_file); g, lg = read_nw(readline(f)); close(f)
    gene_to_species(collect(values(lg)))
    gene2sp = gene_to_species(collect(values(lg)))
    G = GeneTree(g, S, lg, gene2sp)
    return G
end

"""
    read_sp_tree(newick_file)
Read a SpeciesTree directly from a file.
"""
function read_sp_tree(nw_file)
    f = open(nw_file); s, ls = read_nw(readline(f)); close(f)
    S = SpeciesTree(s, ls)
    return S
end

"""
    reverse_labels(mapping)
Reverse a node labeling.
"""
function reverse_labels(mapping::Dict{Int64,String})
    return Dict(kv[2] => kv[1] for kv in mapping)
end

"""
    lca_node(leaves, tree)
Get the least common ancestor for a set of leaves.
"""
function lca_node(leaves::Set, t::Tree)
    for n in postorder_t(t)
        if length(leaves) == 1
            if length(intersect(leaves, n)) == 1
                return n
            end
        elseif length(intersect(
            descendantnodes(t, n), leaves)) == length(leaves)
            return n
        end
    end
end

"""
    lca_node(species, species_tree)
Get the LCA node of a set of species in a SpeciesTree.
"""
function lca_node(leaves, S::SpeciesTree)
    s2n = reverse_labels(S.species)
    leaves = Set([s2n[x] for x in leaves])
    return lca_node(leaves, S.tree)
end

"""
    add_wgd_node!(S::SpeciesTree, node; [τ])
Add WGD node to a SpeciesTree. τ is the time before the parent node.
"""
function add_wgd_node!(S::SpeciesTree, u::Int64; τ::Float64=-1.0)
    wgd_node, τ = insert_node!(u, S.tree, distance_top=τ)
    S.clades[wgd_node] = S.clades[childnodes(S.tree, wgd_node)[1]]
    if length(S.wgd_index) == 0
        S.wgd_index[wgd_node] = 1
    else
        S.wgd_index[wgd_node] = maximum(collect(values(S.wgd_index))) + 1
    end
    return wgd_node, τ
end

"""
    insert_node!(node, t::Tree; [distance_top])
Insert a node in a Tree object. Above a specified node.
"""
function insert_node!(node, t::Tree ; distance_top::Float64=-1.0)
    # delete the branch where the node will be inserted
    p_node = parentnode(t, node)
    branch_length = distance(t, p_node, node)
    deletebranch!(t, t.nodes[node].in[1])
    if distance_top < 0.0
        distance_top = branch_length / 2
    end

    # add a new node
    addnode!(t)
    new_node = length(t.nodes)

    # add the branches
    addbranch!(t, p_node, new_node, distance_top)  # TODO: fix branch length
    addbranch!(t, new_node, node, branch_length - distance_top)  # TODO: fix branch length

    @info "Added node: $new_node"
    return new_node, distance_top
end

"""
    set_rates_clade!(node, rate_class, rate_index, S)
Set rates for all species below a given node that not yet have an entry in the rate_index.
"""
function set_rates_clade!(node, class, rate_index, S)
    function walk(n)
        if !(haskey(rate_index, n))
            rate_index[n] = class
        end
        if isleaf(S.tree, n)
            return
        end
        for x in childnodes(S.tree, n)
            walk(x)
        end
    end
    walk(node)
end

"""
    write(io::IO, tree::Tree)
Write a tree in newick format.
"""
function Base.write(io::IO, tree::Tree)
    root = findroots(tree)[1]
    function walk(n)
        if isleaf(tree, n)
            return "$n:$(distance(tree, n, parentnode(tree, n)))"
        else
            nw_str = ""
            for c in childnodes(tree, n); nw_str *= walk(c) * ","; end
            return n != root ? "($(nw_str[1:end-1])):$(distance(tree, n, parentnode(tree, n)))" :
                "($(nw_str[1:end-1]));"
        end
    end
    write(io, walk(root))
end

"""
    write(io::IO, tree::Tree, labels::Dict)
Write a tree in newick format.
"""
function Base.write(io::IO, tree::Tree, labels::Dict{Int64,String})
    root = findroots(tree)[1]
    function walk(n)
        if isleaf(tree, n)
            return "$(labels[n]):$(distance(tree, n, parentnode(tree, n)))"
        else
            nw_str = ""
            for c in childnodes(tree, n); nw_str *= walk(c) * ","; end
            return n != root ? "($(nw_str[1:end-1])):$(distance(tree, n, parentnode(tree, n)))" :
                "($(nw_str[1:end-1]));"
        end
    end
    write(io, walk(root))
end


"""
    write(io::IO, tree::RecTree)
Write a tree in newick format.
"""
function Base.write(io::IO, rectree::RecTree)
    tree = rectree.tree
    root = findroots(tree)[1]
    function walk(n)
        if isleaf(tree, n)
            return rectree.labels[n] != "loss" ?
                "$(rectree.leaves[n]):$(distance(tree, n, parentnode(tree, n)))" : "loss$n:0.0"
        else
            nw_str = ""
            for c in childnodes(tree, n); nw_str *= walk(c) * ","; end
            return n != root ? "($(nw_str[1:end-1])):$(distance(tree, n, parentnode(tree, n)))" :
                "($(nw_str[1:end-1]));"
        end
    end
    write(io, walk(root) * "\n")
end

"""
    write(io::IO, tree::RecTree, sptree::SpeciesTree)
Write a rectree to an IO stream in RecPhyloXML format.
"""
function Base.write(io::IO, tree::RecTree, sptree::SpeciesTree; family::String="NA")
    write(io, "<recGeneTree\n\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" ")
    write(io, "\n\txmlns=\"http://www.recgenetreexml.org\"\n\txsi:schemaLocation=")
    write(io, "\"http://www.recgenetreexml.org ../../xsd/recGeneTreeXML.xsd\"> ")
    write(io, "\n\t<phylogeny rooted=\"true\">")
    write(io, "\n\t\t<id>$family</id>")
    l = "\t"
    function walk(n)
        if isleaf(tree.tree, n)
            l *= "\t"
            s  = "\n$l<clade>\n$l\t<name>"
            s *= tree.labels[n] == "loss" ? "LOSS" : tree.leaves[n]
            s *= "</name>\n$l\t<eventsRec><$(tree.labels[n]) speciesLocation=\""
            s *= haskey(sptree.species, tree.σ[n]) ? sptree.species[tree.σ[n]] : string(tree.σ[n])
            s *= tree.labels[n] == "loss" ? "\" >" : "\" geneName=\"$(tree.leaves[n])\">"
            s *= "</$(tree.labels[n])>"
            s *= "</eventsRec>\n$l</clade>"
            l = l[1:end-1]
            return s
        else
            l *= "\t"
            s  = "\n$l<clade>\n$l\t<name>$n</name>\n$l\t<eventsRec><$(tree.labels[n]) "
            s *= "speciesLocation=\"$(tree.σ[n])\"></$(tree.labels[n])></eventsRec>"
            for c in childnodes(tree.tree, n); s *= walk(c); end
            s *= "\n$l</clade>"
            l = l[1:end-1]
            return s
        end
    end
    write(io, walk(findroots(tree.tree)[1]))
    write(io, "\n\t</phylogeny>\n</recGeneTree>")
end

function write_rectrees(rtrees::Dict{Any,Array{RecTree}}, S::SpeciesTree, fname::String)
    open(fname, "w") do f
        for (k, v) in rtrees
            for t in v
                write(f, t, S, family=k)
            end
        end
    end
end

function prune_loss_nodes(rt::RecTree)
    rt = deepcopy(rt)
    function walk(n)
        if isleaf(rt.tree, n)
            if rt.labels[n] == "loss"
                deletebranch!(rt.tree, rt.tree.nodes[n].in[1])
                deletenode!(rt.tree, n)
                delete!(rt.labels, n)
                delete!(rt.σ, n)
            end
        else
            for c in childnodes(rt.tree, n)
                walk(c)
            end
        end
    end
    walk(findroots(rt.tree)[1])
    return rt
end
