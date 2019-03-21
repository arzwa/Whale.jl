using Whale

S = read_sp_tree("./example/coffee_mock.nw")
Whale.drawtree(S)
conf = read_whaleconf("./example/whaleamb.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
add_ambiguous!(S, conf)

# non-ambiguous
#ccd = get_ccd("/home/arzwa/coffee/ale-out.1/OG0002088.fasta.nex.treesample.ale", S)[1]

# ambiguous
ccd = get_ccd("/home/arzwa/coffee/ale-out.1/OG0001012.fasta.nex.treesample.ale", S)[1]
ccd = get_ccd("/home/arzwa/coffee/ale-out.1/OG0010729.fasta.nex.treesample.ale", S)[1]
#ccd = get_ccd("/home/arzwa/coffee/ale-out.1/OG0009650.fasta.nex.treesample.ale", S)[1]

# multiple
#ccd = get_ccd("/home/arzwa/coffee/whale/100ale.lst", S)

# whale
results, D = nmwhale(S, [ccd], slices, 0.8)
λ = results.minimizer[1:1]; μ = results.minimizer[2:2]; q = Float64[]; η = 0.8
bt = BackTracker(S, slices, rate_index, λ, μ, q, η)
rt = backtrack(D[1], bt)
Whale.drawtree(rt, height=200, width=300, fontsize=8)
