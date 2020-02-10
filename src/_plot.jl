using DynamicHMC, Whale, DistributedArrays, Distributions, Random
using TransformVariables

wm = WhaleModel(Whale.extree, Δt=0.1)
ccd = distribute(read_ale("example/example-ale/", wm))
prior = Whale.CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = Whale.WhaleProblem(wm, ccd, prior)
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100)
posterior = TransformVariables.transform.(problem.trans, results.chain)
trees = backtrack(wm, ccd, posterior, ConstantRates)
rectrees = sumtrees(trees, ccd, wm)

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
