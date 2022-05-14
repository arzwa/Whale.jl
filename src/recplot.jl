# plot a reconciled tree

# compute the species tree layout
function species_layout(root::Node{T}; scale::V=10., width=10., xpad=5., θ=π/9) where {T,V}
    xleaf = 0 
    coord = Dict{T,Vector{Tuple{V,V}}}()
    function walk(n)
        if isleaf(n)
            t = distance(n) * scale
            coord[id(n)] = [(xleaf,      0.), (xleaf,       t ), (xleaf, t),
                            (xleaf+width,t) , (xleaf+width, t), (xleaf+width, 0.)]
            xleaf += width+xpad
        else
            cs = map(walk, children(n))
            xmid = (cs[1][1][1] + cs[end][end][1]) ÷ 2
            t = distance(n) * scale
            x = xmid - (width÷2)
            y = maximum(last.([c[2] for c in cs])) 
            coord[id(n)] = [(x,y), (x,y+t), (x,y+t), (x+width,y+t), (x+width,y+t), (x+width,y)]
            left = (x, y)
            rght = (x+width, y)
            # modify children diagonal bits
            # left child
            if length(cs) > 1  # not a WGD
                c = cs[1]
                b = x - c[1][1]
                tc = c[2][2] - c[1][2]
                a = min(b * tan(θ), tc)
                qs = [(0,0), (0,a), (-b,0), (-b,0), (0,a), (0,0)]
                p1 = [p .- q for (p,q) in zip(coord[id(n[1])], qs)] 
                coord[id(n[1])] = p1
                # right child
                c = cs[2]
                tc = c[2][2] - c[1][2]
                a = min(b * tan(θ), tc)
                qs = [(0,0), (0,a), (b,0), (b,0), (0,a), (0,0)]
                p2 = [p .- q for (p,q) in zip(coord[id(n[2])], qs)] 
                p2[3] = p1[3]; p2[4] = p1[4]
                coord[id(n[2])] = p2
            end
        end
        return coord[id(n)]
    end 
    walk(root)
    return coord
end

function genetree_layout(wm, groot, slay; scale=10., loss=0.05)
    layout = Dict()
    function walk(sn)  # sn is a species tree node
        !isleaf(sn) && walk.(children(sn))
        e = id(sn)
        # all nodes with rec in this branch (dups, speciation at end, loss at
        # begin)
        gns = filter(x->x.data.e == e, prewalk(groot))
        # losses and leaves
        leaves = filter(x->x.data.label != "duplication", gns)
        # get the relevant coordinates for the branch
        x1 = slay[e][1][1]
        x2 = slay[e][6][1]
        x3 = slay[e][3][1]
        y1 = slay[e][1][2]
        y2 = slay[e][3][2]
        y3 = slay[e][2][2]
        b  = x3 - x1
        a  = y2 - y3
        θ  = tanh(a/b)
        Δx = (x2 - x1) / (length(leaves) + 1)  # x distance between leaves in this species tree branch
        x0 = x1 + Δx
        #tu = wm[e].data.slices[end,1]  # time unit
        # postorder over genes within branch
        # we store two coordinates, for the beginning of the gene tree branch
        # within e (near leaves) and end (near root).
        for n in reverse(gns)
            # we have seen the children (if any) already
            if n ∈ leaves
                layout[n] = [(x0, n.data.label == "loss" ? y2 - loss*scale : y1)]
                x0 += Δx
            else  
                x = 0.
                y = 0.
                for c in children(n)
                    xc = layout[c][1][1]
                    yc = layout[c][1][2] + distance(c)*scale
                    x += xc
                    y += yc
                    push!(layout[c], (xc, yc))
                end
                layout[n] = [(x/2, y/2)]
            end
        end
        # second pass, correct diagonals etc.
        for n in reverse(gns)
            isroot(n) && continue
            if length(layout[n]) == 1
                x, y = layout[n][1]
                push!(layout[n], (x, y2))
            end
            if (parent(n).data.label == "wgd" || parent(n).data.label == "wgdloss")
                p1, p2 = layout[n]
                layout[n] = [p1, (p1[1], p2[2]), p2]
            end
            # correct diagonal bits
            (xa,ya), (xb,yb) = layout[n]
            #ya < y3 && yb < y3 && continue  # we're fine
            dy = yb - ya
            if ya < y3 && yb > y3
                # insert nick
                xshift = (yb-y3)/tan(θ)
                layout[n] = [(xa, ya), (xa, y3), (xb+xshift, yb)]
            elseif ya > y3 && yb > y3 && isleaf(n)
                xshift = dy/tan(θ)
                layout[n] = [(xa+b-xshift, ya), (xb+b, yb)]
            elseif ya >= y3 && yb >= y3 && !isleaf(n)
                xc = mean([layout[c][end][1] for c in children(n)])
                dy = yb - ya
                xshift = dy/tan(θ)
                layout[n] = [(xc, ya), (xc+xshift, yb)]
            end
        end
    end
    walk(getroot(wm))
    for n in postwalk(groot)
        isroot(n) && continue
        parent(n).data.label == "duplication" && continue
        layout[n][end] = layout[parent(n)][1]
    end
    return layout
end

# This is a recipe for plotting a gene tree within a species tree
@recipe function f(S::WhaleModel, G::RecNode; mul=true, c1=:lightgray,
                   c2=:black, fs=10, θ=π/9, sscale=10., loss=0.05)
    slay = species_layout(getroot(S), θ=θ, scale=sscale)
    glay = genetree_layout(S, G, slay, scale=sscale, loss=loss)
    grid --> false
    framestyle --> :none
    legend --> false
    # species tree
    for (k,v) in slay
        @series begin
            seriestype := :shape
            color --> c1
            linecolor --> c1
            v
        end
    end
    for (k,v) in glay
        @series begin
            seriestype := :path
            color --> c2
            v
        end
        if !isroot(k)
            @series begin
                seriestype := :path
                color --> c2
                [v[end], glay[parent(k)][1]]
            end
        end
    end
    anns = map(getleaves(getroot(S))) do n
        cs = slay[id(n)]
        x = (cs[end][1] + cs[1][1])/2
        y = cs[1][2]
        (x, y, (name(n), fs, :top))
    end
    sp   = [v[1] for (k,v) in glay if k.data.label ∈ ["speciation", "sploss"] && !isleaf(k)]
    dup  = [v[1] for (k,v) in glay if k.data.label == "duplication" && !isroot(k)]
    wgd  = [v[1] for (k,v) in glay if k.data.label == "wgd"]
    wgdl = [v[1] for (k,v) in glay if k.data.label == "wgdloss"]
    @series begin
        seriestype := :scatter
        markersize --> 3
        seriescolor --> 1
        markerstrokecolor --> 1
        sp
    end
    @series begin
        seriestype := :scatter
        markersize --> 3
        seriescolor --> 2
        markerstrokecolor --> 2
        dup
    end
    @series begin
        seriestype := :scatter
        markersize --> 3
        seriescolor --> 3
        markerstrokecolor --> 3
        wgd
    end
    @series begin
        seriestype := :scatter
        markersize --> 3
        seriescolor --> 4
        markerstrokecolor --> 4
        wgdl
    end
    @series begin
        annotations := anns
        [], []
    end
end 

const nodecolors = Dict("speciation"=>1, "sploss"=>1, "duplication"=>2, "wgd"=>3, "wgdloss"=>4, "loss"=>5)

# This is a recipe for plotting a reconciled gene tree
@recipe function f(G::RecNode; namefun=identity, fs=9, transform=false, cred=false, fs2=7)
    d = NewickTree.treepositions(G, transform)
    framestyle --> :none
    grid --> false
    legend --> false
    for n in prewalk(G)
        isroot(n) && continue
        color = n.data.label == "loss" ? :lightgray : :black
        @series begin
            seriestype := :path
            seriescolor --> color
            (x1, y1) = d[parent(n)]
            (x2, y2) = d[n]
            [(x1, y1), (x1, y2), (x2, y2)]
        end
    end
    for n in prewalk(G)
        n.data.label == "loss" && continue
        isleaf(n) && continue
        @series begin
            seriestype := :scatter
            seriescolor --> nodecolors[n.data.label]
            markerstrokecolor --> nodecolors[n.data.label]
            markersize --> 4
            [d[n]]
        end
    end
    anns = [(d[n]..., (" " * namefun(name(n)), fs, :left)) 
            for n in getleaves(G) if !(n.data.label == "loss")]
    if cred
        for n in prewalk(G)
            (isleaf(n) || degree(n) == 1) && continue
            push!(anns, (d[n]..., (@sprintf(" %.2f", n.data.cred), fs2, :left)))
        end
    end
    @series begin
        annotations := anns
        [], []
    end
end

@recipe function f(M::WhaleModel, data::Dict; 
                   fs=9, transform=false, leaflabels=true,
                   textcol=:black, textalign=:top)
    d = NewickTree.treepositions(getroot(M), transform)
    framestyle --> :none
    grid --> false
    legend --> false
    anns = []
    for n in prewalk(getroot(M))
        isroot(n) && continue
        (x1, y1) = d[parent(n)]
        (x2, y2) = d[n]
        @series begin
            seriestype := :path
            seriescolor --> :black
            [(x1, y1), (x1, y2), (x2, y2)]
        end
        if haskey(data, id(n))
            xm = (x2 + x1)/2
            push!(anns, (xm, y2, (@sprintf("%d", data[id(n)]), textcol, textalign, fs)))
        end
    end
    if leaflabels
        for n in getleaves(getroot(M))
            push!(anns, (d[n]..., (" " * name(n), fs, :left)))
        end
    end
    @series begin
        annotations := anns
        [],[]
    end
end

@userplot QuantilePlot
@recipe function f(up::QuantilePlot)
    xs = up.args[1]
    ys = up.args[2]
    ym = vec(mean(ys, dims=1))
    yq1 = map(x->quantile(x, 0.025), eachcol(ys))
    yq2 = map(x->quantile(x, 0.975), eachcol(ys))
    ye1 = ym .- yq1
    ye2 = yq2 .- ym
    ymn = minimum(map(x->quantile(x, 0.01), eachcol(ys)))
    ymx = maximum(map(x->quantile(x, 0.99), eachcol(ys)))
    xtrema = (ymn, ymx)
    xlims --> xtrema
    ylims --> xtrema
    legend --> false
    gridstyle --> :dot
    framestyle --> :box
    @series begin
        seriescolor := :lightgray
        x->x
    end
    @series begin
        seriestype := :scatter
        seriescolor --> :black
        markersize --> 3
        yerr := (ye1, ye2)
        xs, ym
    end
    primary := false
    ()
end


