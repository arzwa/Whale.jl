using Whale

S = read_sp_tree("./example/coffee_mock.nw")
conf = read_whaleconf("./example/whaleamb.conf")
slices = get_slices_conf(S, conf["slices"])
rate_index = Whale.constant_ri(S)
add_ambiguous!(S, conf)

# get some test CCD and see whether the CCD reading works correctly
ccd = get_ccd("/home/arzwa/coffee/ale-out.1/OG0002088.fasta.nex.treesample.ale", S)[1]

# compute the likelihood, and see whether the probabilistic stuff is correct
λ = [0.002] ; μ = [0.003] ; q = Float64[] ; η = 0.85
d, l = whale_likelihood_bw(S, ccd, slices, λ, μ, q, η, rate_index)
nmwhale(S, [ccd], slices, 0.8)
ccd.recmat = d

bt = BackTracker(S, slices, rate_index, λ, μ, q, η)
rt = backtrack(ccd, bt)
Whale.drawtree(rt, height=200, width=300, fontsize=8)
