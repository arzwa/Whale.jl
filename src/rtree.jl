# Various tree manipulation routines. In particular related to handling
# reconciled trees
# © Arthur Zwaenepoel - 2019
# XXX some dead code here

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
        if isleaf(T, node)
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
        if isleaf(tree, node)
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
    leafnodes = intersect(
        Set(descendantnodes(rtree.tree, node)), Set(findleaves(rtree.tree)))
    return [rtree.leaves[n] for n in leafnodes if haskey(rtree.leaves, n)]
end

function subtree_leaves(S::SpeciesTree, node::Int64)
    if isleaf(S.tree, node) && haskey(S.leaves, node)
        return [S.leaves[node]]
    end
    leafnodes = intersect(Set(descendantnodes(S.tree, node)), Set(findleaves(S.tree)))
    return [S.leaves[n] for n in leafnodes]
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
    s2n = reverse_labels(S.leaves)
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
    write(io::IO, tree::RecTree)
Write a tree in newick format.
"""
function Base.write(io::IO, rectree::RecTree)
    tree = rectree.tree
    root = findroots(tree)[1]
    function walk(n)
        if isleaf(tree, n)
            return rectree.labels[n] != "loss" ?
                "$(rectree.leaves[n])_$(rectree.σ[n]):$(distance(tree, n, parentnode(tree, n)))" : "loss$n:0.0"
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
    write(io::IO, tree::RecTree)
Write a tree in newick format.
"""
function Base.write(io::IO, crt::ConRecTree)
    root = findroots(crt.tree)[1]
    function walk(n)
        if isleaf(crt, n)
            return crt.labels[n] != "loss" ?
                "$(crt.leaves[n])_$(crt.σ[n]):$(parentdist(crt, n))" :
                    "loss$n:0.0"
        else
            nw_str = ""
            for c in childnodes(crt, n); nw_str *= walk(c) * ","; end
            if n != root
                supstr = haskey(crt.tsupport, n) ?
                    "$(crt.tsupport[n])-$(crt.rsupport[n])" : ""
                return "($(nw_str[1:end-1]))$supstr:$(parentdist(crt, n))"
            else
                return "($(nw_str[1:end-1]));"
            end
        end
    end
    write(io, walk(root) * "\n")
end

"""
    write(io::IO, tree::RecTree, sptree::SpeciesTree)
Write a rectree to an IO stream in RecPhyloXML format.
"""
function Base.write(io::IO, tree::AbstractRecTree, sptree::SpeciesTree;
    family::String="NA")
    write(io, "<recGeneTree\n\txmlns:xsi=\"http://www.w3.org/2001/")
    write(io, "XMLSchema-instance\" ")
    write(io, "\n\txmlns=\"http://www.recgenetreexml.org\"\n\"")
    write(io, "txsi:schemaLocation=\"http://www.recgenetreexml.org ")
    write(io, "../../xsd/recGeneTreeXML.xsd\"> ")
    write(io, "\n\t<phylogeny rooted=\"true\">")
    write(io, "\n\t\t<id>$family</id>")
    l = "\t"
    function walk(n)
        if isleaf(tree.tree, n)
            l *= "\t"
            s  = "\n$l<clade>\n$l\t<name>"
            s *= tree.labels[n] == "loss" ? "LOSS" : tree.leaves[n]
            s *= "</name>\n$l\t<eventsRec><$(tree.labels[n]) speciesLocation=\""
            s *= haskey(sptree.leaves, tree.σ[n]) ?
                sptree.leaves[tree.σ[n]] : string(tree.σ[n])
            s *= tree.labels[n] == "loss" ?
                "\" >" : "\" geneName=\"$(tree.leaves[n])\">"
            s *= "</$(tree.labels[n])>"
            s *= "</eventsRec>\n$l</clade>"
            l = l[1:end-1]
            return s
        else
            l *= "\t"
            s  = "\n$l<clade>\n$l\t<name>$n</name>\n$l\t"
            s *= "<eventsRec><$(tree.labels[n]) "
            s *= "speciesLocation=\"$(tree.σ[n])\"></$(tree.labels[n])>"
            s *= "</eventsRec>"
            for c in childnodes(tree.tree, n); s *= walk(c); end
            s *= "\n$l</clade>"
            l = l[1:end-1]
            return s
        end
    end
    write(io, walk(findroots(tree.tree)[1]))
    write(io, "\n\t</phylogeny>\n</recGeneTree>")
end

"""
    write_rectrees()
"""
function write_rectrees(ccd, S::SpeciesTree, fname::String)
    open(fname, "w") do f
        @showprogress 1 "Writing reconciled trees " for (k, v) in rtrees
            for t in v
                write(f, t, S, family=k)
            end
        end
    end
end

"""
    prune_loss_nodes(rt::RecTree)
"""
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
            if length(childnodes(rt.tree, n)) == 1
                node = rt.tree.nodes[n]
                pnode = parentnode(rt.tree, n)[1]
                cnode = childnodes(rt.tree, n)[1]
                d = distance(rt.tree, pnode, cnode)
                deletebranch!(rt.tree, node.out[1])
                deletebranch!(rt.tree, node.in[1])
                deletenode!(rt.tree, n)
                addbranch!(rt.tree, pnode, cnode, d)
            end
        end
    end
    walk(findroots(rt.tree)[1])
    return rt
end
