"""
    consensus(x::CCD, S::SlicedTree)
    consensus(D::DArray, S::SlicedTree)

Get consensus trees from the backtracked `RecTree`s in the `rectrs` field of the
conditional clade distribution object(s) (CCD).

!!! warning
    Branch lengths are not yet correct (they are meaningfully related to the
    true branch length estimates nevertheless).

!!! warning
    A consensus reconciled tree does not include loss events. This is because
    different reconciled trees for the same family can have different numbers of
    loss events.
"""
function consensus(ccd::CCD, S::SlicedTree, thresh=0.0)
    rt = try
        PhyloTrees.prune_loss_nodes(ccd.rectrs)
    catch
        @warn "Prune loss nodes failed at CCD $(ccd.fname)"
    end
    ct = majority_consensus(rt, thresh=thresh)
    return consensus(ct, rt)
end

consensus(D::CCDArray, S::SlicedTree, thresh=0.0) =
    map((x)->consensus(x, S, thresh), D)

function consensus(contree::Arboreal, rectrees::Array{RecTree})
    recdist = reconciliation_distribution(rectrees)
    T = ConRecTree(contree.tree, contree.leaves)
    function walk(n)
        hsh = hashnode(n, T)
        if haskey(recdist, hsh)  # HACK, should test by outdegree?
            cnt, rec = findmax(recdist[hsh])
            T.labels[n] = rec[1]
            T.σ[n] = rec[2]
            T.rsupport[n] = cnt / length(rectrees)
            T.tsupport[n] = contree.support[n]
            T.recdist[n] = collect(recdist[hsh])
        end
        isleaf(T.tree, n) ? (return) : [walk(c) for c in childnodes(T.tree, n)]
    end
    walk(findroots(T.tree)[1])
    return T
end

function reconciliation_distribution(rectrees::Array{RecTree})
    dist = Dict{UInt64,Dict{Tuple,Int64}}()
    for rt in rectrees
        addrectodist!(dist, rt)
    end
    return dist
end

function addrectodist!(dist, tree::RecTree)
    function walk(n)
        hsh = hashnode(n, tree)
        haskey(dist, hsh) ? nothing : dist[hsh] = Dict{Tuple,Int64}()
        tup = (tree.labels[n], tree.σ[n])
        haskey(dist[hsh], tup) ? dist[hsh][tup] += 1 : dist[hsh][tup] = 1
        if isleaf(tree.tree, n)
            return
        else
            for c in childnodes(tree.tree, n)
                walk(c)
            end
        end
    end
    walk(findroots(tree.tree)[1])
end

function wgd_summary(rectrees::Array{RecTree}, st::SlicedTree)
    d = Dict()
    for rt in rectrees
        for (k,v) in rt.labels
            if v == "wgd"
                wgd = rt.σ[k]
                wgd_id = [k for (k,v) in st.windex if v == wgd][1]
                genes = Set(leafset(k, rt))
                h = hash((wgd, genes))
                !haskey(d, h) ? d[h] = Dict("wgd"=>wgd, "wgd_id"=>wgd_id,
                    "genes"=>genes, "x"=>1) : d[h]["x"] += 1
            end
        end
    end
    df = DataFrame(
        :wgd=>Int64[], :wgd_id=>String[], :x=>Float64[], :genes=>String[])
    for (k,v) in d
        push!(df, [v["wgd"], v["wgd_id"],
            v["x"]/length(rectrees), join(v["genes"], ";")])
    end
    df
end

leafset(node::Int64, tree::Arboreal) = [tree.leaves[n] for n in
    [node ; descendantnodes(tree.tree, node)] if haskey(tree.leaves, n)]

hashnode(node::Int64, tree::Arboreal) = hash(Set(leafset(node, tree)))

"""
    contreetable([io::IO,] contree::ConRecTree, S::SlicedTree)

Write a table representation of a consensus reconciled tree.
"""
function contreetable(io::IO, contree::ConRecTree, S::SlicedTree)
    write(io,"gnode,hash,leaves,mvrec,mvlabel,mvperc,mvspeciesset\n")
    for (n, node) in contree.tree.nodes
        haskey(contree.labels, n) ? nothing : continue
        h = hashnode(n, contree)
        s = contree.σ[n]
        l = contree.labels[n]
        p = contree.rsupport[n]
        ls = join(sort(leafset(n, contree)), ";")
        ss = getspecies(S, s, sep=";")
        write(io, "$n,$h,$ls,$s,$l,$p,$ss\n")
    end
end

contreetable(c::ConRecTree, S::SlicedTree) = contreetable(stdout, c, S)

function write_speciestable(io::IO, S::SlicedTree)
    write(io, "node,label,clades\n")
    for (n, node) in S.tree.nodes
        clades = join(sort(leafset(n, S)), ";")
        l = haskey(S.wgd_index, n) ? "wgd" : "speciation"
        write(io, "$n,$l,$clades\n")
    end
end

getspecies(S::SlicedTree, n::Int64; sep=",") = join([S.leaves[x] for x in
    S.clades[n]], sep)

#= serial version function  write_consensusrectrees(outdir::String, D::CCDArray, st::SlicedTree)
    isdir(outdir) ? nothing : mkdir(outdir)
    @showprogress 1 "Writing tree files " for (i, ccd) in enumerate(D)
        crt = consensus(ccd, st)
        prefix = basename(D[i].fname)
        open("$outdir/$prefix.nw", "w") do f
            write(f, crt)
        end
        open("$outdir/$prefix.csv", "w") do f
            contreetable(f, crt, st)
        end
    end
end=#

function write_consensusrectrees(outdir::String, D::CCDArray, st::SlicedTree)
    isdir(outdir) ? nothing : mkdir(outdir)
    @info "Writing consensus tree files"
    function writefiles(x)
        crt = consensus(x, st)
        prefix = basename(x.fname)
        open("$outdir/$prefix.nw", "w") do f
            write(f, crt)
        end
        open("$outdir/$prefix.csv", "w") do f
            contreetable(f, crt, st)
        end
        open("$outdir/$prefix.wgd.csv", "w") do f
            CSV.write(f, wgd_summary(x.rectrs, st))
        end
    end
    map(x->writefiles(x), D)  # parallelism
end
