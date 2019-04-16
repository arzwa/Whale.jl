using Whale

S = read_sp_tree("./example/morris-9taxa.nw")
conf = read_whaleconf("./example/whalemle.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
ccd = get_ccd("./example/example-ale/OG0004512.fasta.nex.treesample.ale", S)[1]

# whale
results, D = nmwhale(S, [ccd], slices, 0.8)
λ = results.minimizer[1:1]; μ = results.minimizer[2:2]; q = Float64[]; η = 0.8
bt = BackTracker(S, slices, rate_index, λ, μ, q, η)
rt = [backtrack(D[1], bt) for i=1:1000]

Whale.drawtree(rt[7], height=300, width=500, fontsize=8)

# consensus reconciliation
contree = Whale.ConRecTree(Whale.prune_loss_nodes(rt[1]))
Whale.consensus_tree_reconciliation!(contree, rt)

rt1 = Whale.prune_loss_nodes(deepcopy(rt[1]))

rt = [Whale.prune_loss_nodes(t) for t in rt]
recnodes = Whale.count_recnodes(rt)
