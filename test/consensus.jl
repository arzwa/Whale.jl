using Whale
using ConsensusTrees

S = read_sp_tree("./example/morris-9taxa.nw")
conf = read_whaleconf("./example/whalemle.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
ccd = get_ccd("./example/example-ale/OG0004540.fasta.nex.treesample.ale", S)[1]

# whale
results, D = nmwhale(S, [ccd], slices, 0.8)
λ = results.minimizer[1:1]; μ = results.minimizer[2:2]; q = Float64[]; η = 0.8
bt = BackTracker(S, slices, rate_index, λ, μ, q, η)
rt = [backtrack(D[1], bt) for i=1:1000]

# consensus tree
rt = [Whale.prune_loss_nodes(t) for t in rt]
ct = majority_consensus(rt, thresh=0.0)
crt = Whale.consensus_tree_reconciliation(ct, rt)
drawtree(crt)
