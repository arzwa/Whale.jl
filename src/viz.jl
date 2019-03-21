# A new take on visuzalizations for some tree objects in Whale. In particular, (1) branch colored
# species trees (with WGD nodes) and (2) reconciled gene trees.

# draw a tree topology
"""
    drawtree(tree; [kwargs])
Draw a phylogenetic tree. Keyword arguments include image width, height,
linewidth, nodelabels (bool), linewidth and fname (svg file name). tree can
be of a PhyloTrees tree, RecTree or SpeciesTree.
"""
function drawtree(tree::Tree; nodelabels::Bool=false,
        width::Int64=400, height::Int64=300, rect::Bool=true, linewidth=1, fname::String="")
    d = fname == "" ? Luxor.Drawing(width, height, :svg) : Luxor.Drawing(width, height, :svg, fname)
    coords, paths = treecoords(tree, width=width, height=height)
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(linewidth)
    Luxor.setfont("Lato Italic", 10)
    drawtree(coords, paths, width=width, height=height, rect=rect)
    nodelabels ? labelnodes(coords) : nothing
    Luxor.finish()
    Luxor.preview()
end

# draw a tree topology with leaflabels
function drawtree(tree::Tree, labels::Dict; width::Int64=400, height::Int64=300, rect::Bool=true)
    coords, paths = treecoords(tree, width=width, height=height)
    Luxor.Drawing(width, height, :svg)
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1)
    Luxor.setfont("Lato Italic", 8)
    drawtree(coords, paths, width=width, height=height, rect=rect)
    leaflabels(labels, coords, fontfamily="monospace", fontsize=6)
    Luxor.finish()
    Luxor.preview()
end

# draw a reconciled tree
function drawtree(rtree::RecTree; width::Int64=400, height::Int64=300, fname::String="",
        nonretained::Bool=true, fontsize::Int64=7)
    d = fname == "" ? Luxor.Drawing(width, height, :svg) : Luxor.Drawing(width, height, :svg, fname)
    coords, paths = treecoords(rtree.tree, width=width, height=height)
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(1)
    drawtree(coords, paths, width=width, height=height)
    rectreenodes(rtree, coords, nonretained=nonretained)
    leaflabels(rtree.leaves, coords, fontfamily="monospace", fontsize=fontsize)
    Luxor.finish()
    Luxor.preview()
end

# draw a species tree
function drawtree(stree::SpeciesTree; width::Int64=400, height::Int64=300, linewidth=1,
        fname::String="")
    d = fname == "" ? Luxor.Drawing(width, height, :svg) : Luxor.Drawing(width, height, :svg, fname)
    coords, paths = treecoords(stree.tree, width=width, height=height)
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(linewidth)
    drawtree(coords, paths, width=width, height=height)
    wgdnodes(stree, coords)
    leaflabels(stree.species, coords, fontfamily="Lato italic", fontsize=9)
    Luxor.finish()
    Luxor.preview()
end

"""
    coltree(stree::SpeciesTree, values::Array{Float64};
        q::Array{Float64}=[], [kwargs])
Draw a species tree with inferred rates and retention rates. `values`
should be an array with a value at index `i` for branch `i`.
"""
function coltree(stree::SpeciesTree, values::Array{Float64}; q::Array{Float64}=[],
        width::Int64=400, height::Int64=300, fname::String="")
    d = fname == "" ? Luxor.Drawing(width, height, :svg) : Luxor.Drawing(width, height, :svg, fname)
    coords, paths = treecoords(stree.tree, width=width, height=height)
    Luxor.sethue("black")
    Luxor.origin()
    Luxor.setline(3)
    data, nval = process_values(values)
    coltree(coords, paths, data, stree, width=width, height=height)
    length(q) > 0 ? wgdnodes(stree, coords, q) : wgdnodes(stree, coords)
    leaflabels(stree.species, coords, fontfamily="Lato italic", fontsize=9)
    cbar = get_colorbar(width, height, 25)
    draw_colorbar(cbar, nval[1], nval[2])
    Luxor.finish()
    Luxor.preview()
end

# get the tree paths, general (for a PhyloTrees Tree), should just assign a
# coordinate to every node actually
function treecoords(tree::Tree; width::Int64=400, height::Int64=300)
    leaves = findleaves(tree)
    coords = Dict{Int64,Luxor.Point}()
    Δx = width * 0.8 / 2  # offset x-coordinate
    Δy = height / (length(leaves) + 1)  # vertical space between leaves
    yleaf = -Δy * (length(leaves) / 2)  # initial leaf y coordinate
    xmax = maximum([distance(tree, 1, x) for x in leaves])  # maximum distance from root
    a = width * 0.7 / xmax  # modifier
    paths = []

    # postorder traversal
    function walk(node)
        if isleaf(tree, node)
            yleaf += Δy
            x = distance(tree, 1, node) * a - Δx
            coords[node] = Luxor.Point(x, yleaf)
            return yleaf
        else
            ychildren = []
            for child in childnodes(tree,node)
                push!(ychildren, walk(child))
                push!(paths, (node, child))
            end
            y = sum(ychildren) / length(ychildren)
            x = distance(tree, 1, node) * a - Δx
            coords[node] = Luxor.Point(x, y)
            return y
        end
    end
    walk(1)
    return coords, paths
end

# minimal tree drawing routine
function drawtree(coords, paths; width::Int64=400, height::Int64=300, rect::Bool=true)
    for p in paths
        rect ? drawhook(coords[p[1]], coords[p[2]]) : Luxor.line(coords[p[1]], coords[p[2]])
        Luxor.strokepath()
    end
end

# colored tree drawing routine
function coltree(coords, paths, values, stree; width::Int64=400, height::Int64=300)
    for p in paths
        c2 = haskey(values, p[2]) ? get(ColorSchemes.viridis, values[p[2]]) :
            get(ColorSchemes.viridis, values[Whale.non_wgd_child(stree, p[2])])
        c1 = haskey(values, p[1]) ? get(ColorSchemes.viridis, values[p[1]]) : c2
        p1, p2 = coords[p[1]], coords[p[2]]
        p3 = Luxor.Point(p1.x, p2.y)
        Luxor.setblend(Luxor.blend(p1, p3, c1, c2))
        drawhook(p1, p2)
        Luxor.strokepath()
    end
end

# connect two points with a square corner
function drawhook(p1, p2)
    p3 = Luxor.Point(p1.x, p2.y)
    Luxor.poly([p1, p3, p2])
end

# add node markers for rectree nodes
function rectreenodes(rtree::RecTree, coords::Dict; nonretained::Bool=true)
    for (node, coord) in coords
        if rtree.labels[node] == "duplication"
            Luxor.box(coord, 5, 5, 0, :fill)
        elseif rtree.labels[node] == "wgd"
            if !(nonretained)
                if length(childnodes(rtree.tree, node)) == 2
                    Luxor.star(coord, 4, 8, 0.5, 0, :fill)
                end
            else
                Luxor.star(coord, 4, 8, 0.5, 0, :fill)
            end
        elseif rtree.labels[node] == "loss"
            Luxor.star(coord, 1, 4, 5, 0, :fill)
        end
    end
end

# add node markers for wgd nodes
function wgdnodes(stree::SpeciesTree, coords::Dict)
    for (node, i) in stree.wgd_index
        Luxor.star(coords[node], 4, 8, 0.5, 0, :fill)
    end
end

# add node markers for wgd nodes
function wgdnodes(stree::SpeciesTree, coords::Dict, q::Array{Float64};
        fontfamily="Lato", fontsize=8, thresh=0.1)
    Luxor.setfont(fontfamily, fontsize)
    Luxor.sethue("black")
    Luxor.setline(1)
    for (node, i) in stree.wgd_index
        q[i] > thresh ? Luxor.box(coords[node], 3, 9, 0, :fill) :
            Luxor.box(coords[node], 2, 8, 0, :stroke)
        i % 2 == 1 ? Δy = -10 : Δy = 10
        tcoord = Luxor.Point(coords[node].x, coords[node].y + Δy)
        Luxor.settext(string(round(q[i], digits=2)), tcoord, valign="center", halign="center")
    end
end

# add node labels
function labelnodes(coords::Dict; fontfamily="monospace", fontsize=9)
    Luxor.setfont(fontfamily, fontsize)
    for (node, coord) in coords
        Luxor.box(coords[node], 5, 5, 0, :fill)
        tcoord = Luxor.Point(coords[node].x, coords[node].y + 10)
        Luxor.settext(" " * string(node), tcoord, valign="center", halign="left")
    end
end

# add leaf labels
function leaflabels(leaves::Dict, coords::Dict; fontfamily="monospace", fontsize=6)
    Luxor.setfont(fontfamily, fontsize)
    for (node, lab) in leaves
        Luxor.settext("  " * lab, coords[node], valign="center", halign="left")
    end
end

# preprocessing for colortree
function process_values(values::Array{Float64})
    nval = (minimum(values), maximum(values))
    nvalues = (values .- nval[1]) ./ (nval[2] - nval[1])
    data = Dict(i => nvalues[i] for i in 1:length(nvalues))
    return data, nval
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
        Luxor.setline(8)
        ca = get(ColorSchemes.viridis, p[3])
        cb = get(ColorSchemes.viridis, p[4])
        bl = Luxor.blend(p[1], p[2], ca, cb)
        Luxor.line(p[1], p[2])
        Luxor.setblend(bl)
        Luxor.strokepath()
    end
    Luxor.setcolor("black")
    Luxor.setfont("Lato", 8)
    s1 = @sprintf " %3.2f" minv
    s2 = @sprintf " %3.2f" maxv
    p1 = Luxor.Point(cbar[1][1].x + pad, cbar[1][1].y)
    p2 = Luxor.Point(cbar[end][2].x + pad, cbar[end][2].y)
    Luxor.settext(s1, p1, valign="center", halign="left")
    Luxor.settext(s2, p2, valign="center", halign="left")
end
