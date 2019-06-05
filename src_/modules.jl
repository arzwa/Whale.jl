
using PhyloTrees
using Distributions
import Distributions: logpdf, @check_args
import DocStringExtensions: TYPEDSIGNATURES, SIGNATURES, TYPEDEF

include("slicedtree.jl")
include("config.jl")


# test
using Whale
S = read_sp_tree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
conf = Whale.read_whaleconf("./example/whalebay.conf")
q_, ids = mark_wgds!(S, conf["wgd"])
slices = get_slices_conf(S, conf["slices"])
ccd = get_ccd("/home/arzwa/Whale.jl/example/example-ale/", S)
x = ccd[1]

conf = read_whaleconf("./example/whalebay.conf")
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree, conf)

λ = rand(length(collect(values(st.rindex))))
μ = rand(length(collect(values(st.rindex))))
q = rand(length(st.qindex))
η = 0.9
m = WhaleParams(λ, μ, q, η)
w = WhaleModel(st, m)

logpdf(x, w) |> show
whale_likelihood_bw(S, x, slices, λ, μ, q, 0.9, st.rindex)[2] |> show
# some slight differences, probably FP inaccuracies?
