wm = WhaleModel(Whale.extree, Δt=0.1)
ccd = read_ale("example/example-ale/", wm)
NewickTree.distance(n::Whale.RecNode) = 1.
wm = wm(BranchRates(r=exp.(randn(2,17)), η=0.9))
logpdf!(wm, ccd)


r = backtrack(wmm, ccd)

begin
    i = 2
    ts = rtrees[:,i]
    smry = sumtrees(ts)
    r = smry[1].tree
    cred = cladecredibility(r, ts)
    loss = [cladehash(n) for n in postwalk(r) if n.γ == 0]
    names = Dict(cladehash(n)=> 0 < n.γ <= length(D[i].leaves) ?
        D[i].leaves[n.γ] : "" for n in postwalk(r))
    tl = TreeLayout(r)
    PalmTree.cladogram!(tl)

    @svg begin
        origin(Point(10,10))
        setfont("Noto sans", 11)
        drawtree(tl, color=(n)->n ∉ loss ? RGB() : RGB(0.99,0.99,0.99))
        nodemap(tl, (k, p)->settext(" $(names[k])", p, valign="center"), tl.leaves)
        nodemap(tl, (k, p)->settext(" $(cred[k])", p, valign="center"), keys(cred))
    end 800 350 "/tmp/tree.svg"
end
