using Whale
using PalmTree

S = read_sp_tree("./example/coffee.nw")
conf = read_whaleconf("./example/whaleamb.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
add_ambiguous!(S, conf)
drawtree(S.tree, S.leaves)
drawtree(S.tree, nodelabels=true)

# non-ambiguous
#ccd = get_ccd("$base/OG0002088.fasta.nex.treesample.ale", S)[1]

# ambiguous
base = "/home/arzwa/coffee/ale-out.1"
ccd = get_ccd("$base/OG0010729.fasta.nex.treesample.ale", S)[1]
ccd = get_ccd("$base/OG0001012.fasta.nex.treesample.ale", S)
#ccd = get_ccd("$base/OG0009650.fasta.nex.treesample.ale", S)[1]

# multiple
ccd = get_ccd("/home/arzwa/coffee/whale/100ale.lst", S)

# whale
results, D = nmwhale(S, ccd, slices, 0.8)
λ = results.minimizer[1:1]; μ = results.minimizer[2:2]; q = Float64[]; η = 0.8
bt = BackTracker(S, slices, rate_index, λ, μ, q, η)
rt = [backtrack(D[1], bt) for i=1:1000]
drawtree(rt[2], height=200, width=300, fontsize=6)

rt1 = Whale.prune_loss_nodes(rt[1])
Whale.drawtree(rt1, height=300, width=500, fontsize=8)
write(stdout, rt1)


open("./example/rectree.xml", "w") do io
    write(io, rt[1], S, family=ccd.fname)
end


mr = mrpencode(rt, loss=false)

parsfitch2(parstree, mr.matrix, initleafmap(parstree))

parstree, leaves = parsfitch(mr, ctol=5)
drawtree(parstree, leaves)
write("/home/arzwa/tmp/test.nw", parstree, leaves)
