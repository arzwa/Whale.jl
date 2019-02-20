#===============================================================================
Write a small tree visualizer using Luxor
===============================================================================#
import Luxor
using ColorSchemes

"""
A minimal tree visuzalition function.
"""
function minimal_tree(
    T::Tree; width::Int64=400, height::Int64=300, fname::String=""
)
    paths, labels = get_tree_paths(T, width=width, height=height)
    draw_array_of_paths(paths, width, height, fname=fname)
end


function node_labeled_tree(
    T::Tree; width::Int64=400, height::Int64=300, fname::String=""
)
    paths, labels = get_tree_paths(T, width=width, height=height)
    draw_array_of_paths(paths, labels, width, height, fname=fname)
end


function get_tree_paths(T::Tree; width::Int64=400, height::Int64=300)
    paths = []
    labels = []
    leaves = findleaves(T)
    Δy = height / (length(leaves) + 2)
    leaf_y = -Δy*(length(leaves) / 2)
    Δx = width*0.8/2
    max_x = maximum([PhyloTrees.distance(T, 1, x) for x  in leaves])
    a = width*0.8 / max_x
    function walk(node)
        if PhyloTrees.isleaf(T, node)
            leaf_y += Δy
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            return Luxor.Point(x, leaf_y)
        else
            child_coord = Luxor.Point[]
            label = Int64[]
            children = childnodes(T, node)
            for c in children
                child_point = walk(c)
                push!(child_coord, child_point)
                push!(label, c)
            end
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            y = sum([p.y for p in child_coord])/length(child_coord)
            if length(child_coord) == 1
                path = Luxor.Point[
                    child_coord[1], Luxor.Point(x, child_coord[1].y)]
                label = [label; ""]
            elseif length(child_coord) == 2
                path = Luxor.Point[
                    child_coord[1], Luxor.Point(x, child_coord[1].y),
                    Luxor.Point(x, child_coord[2].y), child_coord[2]
                ]
                label = [label[1]; "" ; "" ; label[2]]
            else
                error("No multifurcations allowed!")
            end
            push!(paths, path)
            push!(labels, label)
            return Luxor.Point(x, y)
        end
    end
    walk(1)
    return paths, labels
end

"""
Draw a set of paths
"""
function draw_array_of_paths(paths, width::Int64, height::Int64;
        fname::String="")
    if fname != ""
        d = Luxor.Drawing(width, height, :svg, fname)
    else
        d = Luxor.Drawing(width, height, :svg)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1)
    for p in paths
        Luxor.poly(p, :stroke)
    end
    Luxor.finish()
    Luxor.
    Luxor.preview()
end


"""
Draw a set of paths
"""
function draw_array_of_paths(
        paths, labels, width::Int64, height::Int64; fname::String=""
    )
    if fname != ""
        d = Luxor.Drawing(width, height, :svg, fname)
    else
        d = Luxor.Drawing(width, height, :svg)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1)
    Luxor.fontsize(10)
    for p in 1:length(paths)
        labs = labels[p]
        length(paths[p]) > 2 ? size = 0 : size = 2
        Luxor.prettypoly(
            paths[p], :stroke, () ->
            begin
                Luxor.circle(Luxor.O, size, :fill)
            end,
            vertexlabels = (n, l) ->
                (Luxor.label(string(labs[n]), :E, offset=height/40))
        )
    end
    Luxor.finish()
    #Luxor.save()
    Luxor.preview()
end


# Rectree ----------------------------------------------------------------------

"""
I'd like to have a fancy rectree visualizer, more minimalistic than the typical
rectree visualisations.

    - speciation node  → nothing
    - duplication node → square? circle?
    - wgd node         → star? square? circle?
"""
function rectree_paths(R::RecTree; width::Int64=400, height::Int64=300)
    T = R.tree
    paths = []
    labels = []
    dup_nodes = Luxor.Point[]
    wgd_nodes = Luxor.Point[]
    leaves = findleaves(T)
    Δy = height / (length(leaves) + 2)
    leaf_y = -Δy*(length(leaves) / 2)
    Δx = width*0.8/2
    max_x = maximum([PhyloTrees.distance(T, 1, x) for x  in leaves])
    a = width*0.8 / max_x

    function walk(node)
        if PhyloTrees.isleaf(T, node)
            leaf_y += Δy
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            return Luxor.Point(x, leaf_y)
        else
            child = Luxor.Point[] ; label = String[]

            # recurse
            children = childnodes(T, node)
            if length(children) != 2
                error("Only bifurcations allowed!")
            end
            for c in children
                child_point = walk(c)
                push!(child, child_point)
                if haskey(R.leaves, c)
                    push!(label, R.leaves[c])
                else
                    push!(label, "")
                end
            end

            # construct path
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            y = sum([p.y for p in child])/length(child)
            label = [label[1]; "" ; "" ; label[2]]
            path = Luxor.Point[
                child[1], Luxor.Point(x, child[1].y),
                Luxor.Point(x, child[2].y), child[2]
            ]

            # record dups/wgds
            if R.labels[node] == "duplication"
                push!(dup_nodes, Luxor.midpoint(path[2], path[3]))
            elseif R.labels[node] == "wgd"
                push!(wgd_nodes, Luxor.midpoint(path[2], path[3]))
            end

            push!(paths, path)
            push!(labels, label)
            return Luxor.Point(x, y)
        end
    end
    walk(1)
    return paths, labels, dup_nodes, wgd_nodes
end


"""
Draw a set of paths
"""
function draw_array_of_paths(
        paths, labels, dup_nodes, wgd_nodes, width::Int64, height::Int64;
        fname::String=""
    )
    if fname == ""
        d = Luxor.Drawing(width, height, :svg)
    else
        d = Luxor.Drawing(width, height, :svg, fname)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1)
    Luxor.fontsize(6)
    for p in 1:length(paths)
        labs = labels[p]
        Luxor.prettypoly(
            paths[p],
            :stroke,
            () ->
                begin
                    Luxor.circle(Luxor.O, 0, :fill)
                end,
            vertexlabels = (n, l) ->
                (Luxor.label(string(labs[n]), :E, offset=height/40))
        )
    end
    for dup_node in dup_nodes
        Luxor.box(dup_node, 5, 5, 0, :fill)
    end

    for wgd_node in wgd_nodes
        Luxor.star(wgd_node, 4, 8, 0.5, 0, :fill)
    end
    Luxor.finish()
    #Luxor.save()
    Luxor.preview()
end


"""
A minimal tree visualization function.
"""
function draw_rectree(R::RecTree; width::Int64=400, height::Int64=300, fname="")
    paths, labels, d, w = rectree_paths(R, width=width, height=height)
    draw_array_of_paths(paths, labels, d, w, width, height, fname=fname)
end


# Species tree -----------------------------------------------------------------

"""
Species Tree
"""
function sptree_paths(S::SpeciesTree; width::Int64=400, height::Int64=300)
    T = S.tree
    paths = []
    labels = []
    wgd_nodes = Luxor.Point[]
    leaves = findleaves(T)
    Δy = height / (length(leaves) + 2)
    leaf_y = -Δy*(length(leaves) / 2)
    Δx = width*0.8/2
    max_x = maximum([PhyloTrees.distance(T, 1, x) for x  in leaves])
    a = width*0.7 / max_x

    function walk(node)
        if PhyloTrees.isleaf(T, node)
            leaf_y += Δy
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            return Luxor.Point(x, leaf_y)
        else
            child = Luxor.Point[] ; label = String[]

            # recurse
            children = childnodes(T, node)
            for c in children
                child_point = walk(c)
                push!(child, child_point)
                if haskey(S.species, c)
                    push!(label, S.species[c])
                else
                    push!(label, "")
                end
            end

            # construct path
            x = PhyloTrees.distance(T, 1, node)*a - Δx
            y = sum([p.y for p in child])/length(child)
            if length(children) == 1
                path = Luxor.Point[child[1], Luxor.Point(x, child[1].y)]
                label = [label; ""]
            else
                path = Luxor.Point[
                    child[1], Luxor.Point(x, child[1].y),
                    Luxor.Point(x, child[2].y), child[2]
                ]
                label = [label[1]; "" ; "" ; label[2]]
            end

            # record dups/wgds
            if haskey(S.wgd_index, node)
                push!(wgd_nodes, path[2])
            end

            push!(paths, path)
            push!(labels, label)
            return Luxor.Point(x, y)
        end
    end
    walk(1)
    return paths, labels, wgd_nodes
end


"""
Draw a set of paths
"""
function draw_array_of_paths(
        paths, labels, wgd_nodes, width::Int64, height::Int64; fname::String=""
    )
    if fname == ""
        d = Luxor.Drawing(width, height, :svg)
    else
        d = Luxor.Drawing(width, height, :svg, fname)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1.2)
    Luxor.setfont("Lato Italic", 8)
    for p in 1:length(paths)
        labs = labels[p]
        Luxor.prettypoly(
            paths[p],
            :stroke,
            () ->
                begin
                    Luxor.circle(Luxor.O, 0, :fill)
                end,
        )
        for (i, l) in enumerate(labs)
            Luxor.settext(
                "  " * labs[i], paths[p][i], valign="center", halign="left"
            )
        end
    end
    for wgd_node in wgd_nodes
        Luxor.star(wgd_node, 4, 8, 0.5, 0, :fill)
    end
    Luxor.finish()
    Luxor.preview()
end


"""
A minimal tree visualization function.
"""
function draw_sptree(
    S::SpeciesTree; width::Int64=400, height::Int64=300, fname::String=""
)
    paths, labels, w = sptree_paths(S, width=width, height=height)
    draw_array_of_paths(paths, labels, w, width, height, fname=fname)
end


"""
A minimal tree visualization function.
"""
function draw_sptree(
    S::SpeciesTree, leaf_names::Dict{String,String}; width::Int64=400,
    height::Int64=300, fname::String=""
)
    paths, labels, w = sptree_paths(S, width=width, height=height)
    modify_labels_!(labels, leaf_names)
    draw_array_of_paths(paths, labels, w, width, height, fname=fname)
end


function modify_labels_!(labels, mapping)
    for (i, p) in enumerate(labels)
        for (j, label) in enumerate(p)
            if haskey(mapping, label)
                labels[i][j] = mapping[label]
            end
        end
    end
end

#=function draw_branch_gradients(dict, width::Int64, height::Int64;
        fname::String="", rect::Bool=true)
    if fname == ""
        d = Luxor.Drawing(width, height, :svg)
    else
        d = Luxor.Drawing(width, height, :svg, fname)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(2)
    Luxor.setfont("Lato Italic", 8)
    for p in dict["pths"]
        ca = get(ColorSchemes.viridis, p[3])
        cb = get(ColorSchemes.viridis, p[4])
        if rect
            corner = Luxor.Point(p[1].x, p[2].y)
            bl = Luxor.blend(corner, p[2], ca, cb)
            Luxor.line(corner, p[2])
            Luxor.line(p[1], corner)
        else
            bl = Luxor.blend(p[1], p[2], ca, cb)
            Luxor.line(p[1], p[2])
        end
        Luxor.setblend(bl)
        Luxor.strokepath()
    end
    Luxor.setcolor("black")
    for l in dict["labs"]
        Luxor.settext("  " * l[2], l[1], valign="center", halign="left")
    end
    for wgd_node in dict["wgds"]
        Luxor.star(wgd_node, 4, 8, 0.5, 0, :fill)
    end
    draw_colorbar(dict["cbar"], 0.0, dict["nval"][2])
    Luxor.finish()
    Luxor.preview()
end=#

# Color trees ======================================================================================
"""
Species Tree
"""
function sptree_paths(S::SpeciesTree, node_values;
        width::Int64=400, height::Int64=300)
    # prepare
    tree  = S.tree
    pths = []  # contains tuples with (p1, p2, l2, v1, v2)
    labs = []
    wgds = Luxor.Point[]
    leaves = findleaves(tree)

    # (initial) settings
    nval = (minimum(node_values), maximum(node_values))
    Δy = height / (length(leaves) + 2)
    Δx = width * 0.8 / 2
    leaf_y = -Δy * (length(leaves) / 2)  # initial leaf y coordinate
    max_x = maximum([PhyloTrees.distance(tree, 1, x) for x  in leaves])
    a = width * 0.7 / max_x
    @show nvalues = (node_values .- nval[1]) ./ (nval[2] - nval[1])

    function walk(node)
        if PhyloTrees.isleaf(tree, node)
            leaf_y += Δy
            x = PhyloTrees.distance(tree, 1, node) * a - Δx
            push!(labs, [Luxor.Point(x, leaf_y), S.species[node]])
            return Luxor.Point(x, leaf_y)
        else
            child = Luxor.Point[]
            cv = Float64[]

            # recurse
            children = childnodes(tree, node)
            for c in children
                child_point = walk(c)
                push!(child, child_point)
                push!(cv, nvalues[c])
            end

            # construct path
            x = PhyloTrees.distance(tree, 1, node) * a - Δx
            y = sum([p.y for p in child])/length(child) # midpoint
            n1 = nvalues[node]

            if length(children) == 1
                push!(pths, (Luxor.Point(x, y), child[1], n1, cv[1]))
            else
                push!(pths, (Luxor.Point(x, y), child[1], n1, cv[1]))
                push!(pths, (Luxor.Point(x, y), child[2], n1, cv[2]))
            end

            # record dups/wgds
            if haskey(S.wgd_index, node)
                push!(wgds, Luxor.Point(x, y))
            end

            return Luxor.Point(x, y)
        end
    end
    walk(1)
    cbar = get_colorbar(width, height, 25)
    return Dict("pths" => pths, "wgds" => wgds, "labs" => labs, "cbar" => cbar, "nval" => nval)
end

# get a colorbar
function get_colorbar(width, height, pad; breaks=7)
    path = []
    c = 0.
    x = -width/2 + pad
    y = height/2 - pad
    y2 = height/2 - 3*pad
    Δy = (y2 - y) / breaks
    Δc = 1/breaks
    for i = 1:breaks
        push!(path, (Luxor.Point(x, y), Luxor.Point(x, y+Δy), c, c+Δc))
        y += Δy
        c += Δc
    end
    return path
end

# draw a colorbar
function draw_colorbar(cbar, minv, maxv; pad=2)
    for p in cbar
        Luxor.setline(5)
        ca = get(ColorSchemes.viridis, p[3])
        cb = get(ColorSchemes.viridis, p[4])
        bl = Luxor.blend(p[1], p[2], ca, cb)
        Luxor.line(p[1], p[2])
        Luxor.setblend(bl)
        Luxor.strokepath()
    end
    Luxor.setcolor("black")
    Luxor.setfont("Lato", 6)
    s1 = @sprintf " %3.2f" minv
    s2 = @sprintf " %3.2f" maxv
    p1 = Luxor.Point(cbar[1][1].x + pad, cbar[1][1].y)
    p2 = Luxor.Point(cbar[end][2].x + pad, cbar[end][2].y)
    Luxor.settext(s1, p1, valign="center", halign="left")
    Luxor.settext(s2, p2, valign="center", halign="left")
end

# draw colored branches
function draw_branch_colors(dict, width::Int64, height::Int64;
        fname::String="", rect::Bool=true)
    if fname == ""
        d = Luxor.Drawing(width, height, :svg)
    else
        d = Luxor.Drawing(width, height, :svg, fname)
    end
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(2)
    Luxor.setfont("Lato Italic", 8)
    for p in dict["pths"]
        ca = get(ColorSchemes.viridis, p[3])
        cb = get(ColorSchemes.viridis, p[4])
        if rect
            corner = Luxor.Point(p[1].x, p[2].y)
            bl = Luxor.blend(p[1], corner, ca, cb)
            Luxor.line(corner, p[2])
            Luxor.line(p[1], corner)
        else
            bl = Luxor.blend(p[1], p[2], cb, cb)
            Luxor.line(p[1], p[2])
        end
        Luxor.setblend(bl)
        Luxor.strokepath()
    end
    Luxor.setcolor("black")
    for l in dict["labs"]
        Luxor.settext("  " * l[2], l[1], valign="center", halign="left")
    end
    for wgd_node in dict["wgds"]
        Luxor.star(wgd_node, 4, 8, 0.5, 0, :fill)
    end
    draw_colorbar(dict["cbar"], dict["nval"][1], dict["nval"][2])
    Luxor.finish()
    Luxor.preview()
end

# modify labels based on leaf name mapping
function modify_labels!(dict, mapping)
    for x in values(dict["labs"])
        if haskey(mapping, x[2])
            x[2] = mapping[x[2]]
        end
    end
end

"""
Draw a colortree.
"""
function draw_colortree(S::SpeciesTree, node_values::Array{Float64},
        leaf_names::Dict{String,String}; width::Int64=400, height::Int64=300,
        fname::String="", nval=nothing)
    d = sptree_paths(S, node_values, width=width, height=height)
    modify_labels!(d, leaf_names)
    if nval != nothing
        d["nval"] = nval
    end
    draw_branch_colors(d, width, height, fname=fname)
end

function draw_colortree(S::SpeciesTree, node_values::Array{Float64};
        width::Int64=400, height::Int64=300, fname::String="")
    d = sptree_paths(S, node_values, width=width, height=height)
    draw_branch_colors(d, width, height, fname=fname)
end
