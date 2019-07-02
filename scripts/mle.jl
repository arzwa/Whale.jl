# default MLE script
# config
ccds = ""
treefile = ""
wgdconf = Dict()
constant = false

# modules
using Distributed
using DistributedArrays
@everywhere using Whale

# io
st = SlicedTree(treefile)
ccd = read_ale(ccds, st)

# inference
constant ? set_constantrates!(st) : nothing
w = WhaleModel(st)
out = mle(w, ccd)
