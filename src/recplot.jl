# plot a reconciled tree

# compute the species tree layout
function species_layout(root::Node{T}; scale::V=10., width=10., xpad=5., ypad=2.) where {T,V}
    xleaf = 0 
    coord = Dict{T,Vector{Tuple{V,V}}}()
    diags = Dict{T,Vector{Tuple{V,V}}}()
    function walk(n)
        if isleaf(n)
            t = distance(n) * scale
            coord[id(n)] = [(xleaf,      0.), (xleaf,       t ), 
                            (xleaf+width, t), (xleaf+width, 0.)]
            xleaf += width+xpad
        else
            cs = map(walk, children(n))
            xmid = (cs[1][1][1] + cs[end][end][1]) ÷ 2
            t = distance(n) * scale
            x = xmid - (width÷2)
            y = maximum(last.([c[2] for c in cs])) + (length(cs) > 1 ? ypad : 0)
            coord[id(n)] = [(x, y), (x, y+t), (x+width, y+t), (x+width,y)]
            left = (x, y)
            rght = (x+width, y)
            for (c, x) in zip(cs, children(n))
                diags[id(x)] = [c[2], left, rght, c[3]]
            end
        end
        return coord[id(n)]
    end 
    walk(root)
    return coord, diags
end

# compute the species tree layout
function species_layout2(root::Node{T}; scale::V=10., width=10., xpad=5., θ=π/9) where {T,V}
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
        return coord[id(n)]
    end 
    walk(root)
    return coord
end

function genetree_layout2(wm, groot, slay; scale=10., loss=0.05)
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
            if length(layout[n]) == 1
                x, y = layout[n][1]
                push!(layout[n], (x, y2))
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

# embed the gene tree within the species tree -- but should first make a proper
# timetree from the RecNode tree...
function genetree_layout(wm, groot, slay; scale=10.)
    layout = Dict()
    function walk(sn)  # sn is a species tree node
        if !isleaf(sn)
            for c in children(sn)
                walk(c)
            end
        end
        e = id(sn)
        # all nodes with rec in this branch (dups, speciation at end, loss at
        # begin)
        gns = filter(x->x.data.e == e, prewalk(groot))
        # losses and leaves
        leaves = filter(x->x.data.label != "duplication", gns)
        xmin = slay[e][1][1]
        xmax = slay[e][3][1]
        ymin = slay[e][1][2]
        ymax = slay[e][3][2]
        Δx = (xmax - xmin) / (length(leaves) + 1)
        x0 = xmin + Δx
        tu = wm[e].data.slices[end,1]  # time unit
        # postorder over genes within branch
        # we store two coordinates, for the beginning of the gene tree branch
        # within e (near leaves) and end (near root).
        for n in reverse(gns)
            # we have seen the children (if any) already
            if n ∈ leaves
                layout[n] = [(x0, n.data.label == "loss" ? ymax - 0.1scale : ymin)]
                x0 += Δx
            else
                #y = ymin + distance(n) * scale
                x = 0.
                for c in children(n)
                    xc = layout[c][1][1]
                    yc = layout[c][1][2]
                    x += xc
                    push!(layout[c], (xc, y+distance(c)))
                end
                layout[n] = [(x/length(children(n)), y)]
            end
        end
        for n in reverse(gns)
            if length(layout[n]) == 1
                x, y = layout[n][1]
                push!(layout[n], (x, ymax))
            end
        end
    end
    walk(getroot(wm))
    return layout
end

@recipe function f(S::WhaleModel, G::RecNode; mul=true, c1=:lightgray,
                   c2=:black, fs=10, θ=π/9, sscale=10., loss=0.05)
    slay = species_layout2(getroot(S), θ=θ, scale=sscale)
    glay = genetree_layout2(S, G, slay, scale=sscale, loss=loss)
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
    sp  = [v[1] for (k,v) in glay if k.data.label ∈ ["speciation", "sploss"] && !isleaf(k)]
    dup = [v[1] for (k,v) in glay if k.data.label == "duplication" && !isroot(k)]
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
        annotations := anns
        [], []
    end
end 
