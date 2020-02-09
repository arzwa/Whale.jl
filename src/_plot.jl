wm = WhaleModel(Whale.extree, Δt=0.1)
ccd = read_ale("example/example-ale/", wm)
NewickTree.distance(n::Whale.RecNode) = 1.


wm = wm(BranchRates(r=exp.(randn(2,17)), η=0.9))
logpdf!(wm, ccd)
r = backtrack(wm, ccd)
tl = TreeLayout(r)
PalmTree.cladogram!(tl)
function leafname(n)
    startswith(n, "0") ? (return "") : nothing
    γ, _ = parse.(Int16, split(n, "."))
    return 0 < γ <= length(ccd.leaves)  ? " "*ccd.leaves[γ] : ""
end

@svg begin
    origin(Point(10,10))
    drawtree(tl, color=(n)->startswith(n, "0") ? RGB(0.8,0.8,0.8) : RGB())
    nodemap(tl, (k, p)->settext(leafname(k), p, valign="center"), tl.leaves)
end 800 320 "/tmp/tree.svg"
