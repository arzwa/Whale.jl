
# this will become the module
using Distributed
#addprocs(3)
@everywhere whalemodule = quote
    using PhyloTrees
    using Distributions
    using Distributed
    using DistributedArrays
    using Optim
    using Printf
    import PhyloTrees: isleaf
    import ProgressMeter: @showprogress
    import Distributions: logpdf, @check_args, rand
    import DocStringExtensions: TYPEDSIGNATURES, SIGNATURES, TYPEDEF

    # I guess order matters for type dependencies
    base = "/home/arzwa/Whale.jl/src_/"
    include("$base/slicedtree.jl")
    include("$base/conf.jl")
    include("$base/ccd.jl")
    include("$base/core.jl")
    include("$base/distributed.jl")
end

# temporary way of testing out the refactored package ==========================
@everywhere eval(whalemodule)

wgdconf = Dict(
    "PPAT" => ("PPAT", 0.655, -1.0),
    "CPAP" => ("CPAP", 0.275, -1.0),
    "BETA" => ("ATHA", 0.55, -1.0),
    "ANGI" => ("ATRI,ATHA", 3.08, -1.0),
    "SEED" => ("GBIL,ATHA", 3.9, -1.0),
    "MONO" => ("OSAT", 0.91, -1.0),
    "ALPH" => ("ATHA", 0.501, -1.0))

# validation by comparing with previous implementation ========================
using Whale
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree, wgdconf)
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)
x = ccd[1]

λ = rand(length(Set(collect(values(st.rindex)))))
μ = rand(length(Set(collect(values(st.rindex)))))
q = rand(length(st.qindex))
η = 0.9

m = WhaleParams(λ, μ, q, η)
w = WhaleModel(st, m)
logpdf(x, w) |> show

S = SpeciesTree(st.tree, st.leaves, st.clades, st.qindex, Dict{Int64,String}())
conf = Whale.read_whaleconf("./example/whalebay.conf")
slices = get_slices_conf(S, conf["slices"])
ccd_ = get_ccd("/home/arzwa/Whale.jl/example/example-ale/", S)
x_ = ccd_[1]
y, lp_ = whale_likelihood_bw(S, x_, slices, λ, μ, q, 0.9, st.rindex)
lp = logpdf(x, w)
println("new: $lp, old: $lp_")

# MLE comparison with previous implementation ==================================
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree)
set_constantrates!(st)
m = WhaleParams(0.3, 0.2, Float64[], 0.9)
w = WhaleModel(st, m)

S = SpeciesTree(st.tree, st.leaves, st.clades, st.qindex, Dict{Int64,String}())
conf = Whale.read_whaleconf("./example/whalebay.conf")
slices = get_slices_conf(S, conf["slices"])

x = read_ale("example/example-ale/OG0004544.fasta.nex.treesample.ale", st)
x_ = get_ccd("example/example-ale/OG0004544.fasta.nex.treesample.ale", S)

@time out = nmwhale(distribute(x), w)
@time out_, d_ = Whale.nmwhale(S, x_, slices, 0.9, Float64[])
