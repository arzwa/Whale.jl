
# this will become the module
using Distributed
#addprocs(3)
@everywhere whalemodule = quote
    using PhyloTrees
    using Distributions
    using Distributed
    using DistributedArrays
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

@everywhere eval(whalemodule)

conf = read_whaleconf("./example/whalebay.conf")
tree = readtree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
st = SlicedTree(tree, conf)
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st)
x = ccd[1]

λ = rand(length(collect(values(st.rindex))))
μ = rand(length(collect(values(st.rindex))))
q = rand(length(st.qindex))
η = 0.9

m = WhaleParams(λ, μ, q, η)
w = WhaleModel(st, m)

@time logpdf(x, w) |> show
@time logpdf(x, w, 4) |> show

d = distribute(vcat([ccd for i=1:100]...))
@time logpdf(d, w) |> show
@time [logpdf(d, w) for i=1:10]
@time [logpdf(d, w, 5) for i=1:10]


# compare with previous implementation
using Whale
S = read_sp_tree("/home/arzwa/Whale.jl/example/morris-9taxa.nw")
conf = Whale.read_whaleconf("./example/whalebay.conf")
q_, ids = mark_wgds!(S, conf["wgd"])
slices = get_slices_conf(S, conf["slices"])
ccd = get_ccd("/home/arzwa/Whale.jl/example/example-ale/", S)
xx = ccd[1]
@time whale_likelihood_bw(S, xx, slices, λ, μ, q, 0.9, st.rindex)[2] |> show
