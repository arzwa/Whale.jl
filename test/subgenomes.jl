
S = read_sp_tree("example/testamb.nw")
conf = read_whaleconf("./example/whaleamb.conf")
Whale.add_ambiguous!(S, conf)

# get some test CCD and see whether the CCD reading works correctly

# compute the likelihood, and see whether the probabilistic stuff is correct
