using NewickTree, Parameters, TransformVariables, AbstractTrees
using DistributedArrays, BenchmarkTools, Distributed, Test
using Distributions, LogDensityProblems, ForwardDiff, LinearAlgebra
import TransformVariables: TransformTuple
import Distributions: logpdf
include("../../BirdDad/src/rates.jl")
include("_model.jl")
include("_ccd.jl")
include("_core.jl")
include("_prior.jl")
include("_dhmc.jl")
include("_track.jl")
include("rmodels.jl")
import RatesModels: DLGWGD, ConstantDLGWGD

t = readnw("((MPOL:4.752,(PPAT:2.752)wgd:2):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
# t = readnw("((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
# t = readnw("(((A:1,B:1):0.5),C:1.5);")
n = length(postwalk(t))
r = RatesModel(DLGWGD(λ=ones(n), μ=ones(n), q=[0.2], η=0.9))
w = WhaleModel(r, t)
ccd = read_ale("example/example-1/ale", w)
@test logpdf!(w, ccd) == -399.4124195572149
@test ts = backtrack(w, ccd)

@btime logpdf!(w, ccd)
@btime logpdf!(w, ccd[1])

x = randn(Float32, dimension(r.trans))
@time w(x)
@btime logpdf!(w, ccd)
@btime logpdf(w, ccd)

ForwardDiff.gradient(x->logpdf(w(x), ccd[1]), randn(5))



using DynamicHMC, Random
# r = RatesModel(DLGWGD(λ=ones(n), μ=ones(n), q=rand(1), η=0.9), fixed=(:κ,))
# p = IWIRPrior()
r = RatesModel(ConstantDLGWGD(λ=0.1, μ=0.1, q=[0.2], η=0.9), fixed=(:κ,))
p = CRPrior()
w = WhaleModel(r, t)
d = read_ale("example/example-1/ale", w)

problem = WhaleProblem(d, w, p)
LogDensityProblems.logdensity_and_gradient(problem, randn(dimension(r.trans)))

wup = DynamicHMC.default_warmup_stages(doubling_stages=3)
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, warmup_stages=wup)
posterior = transform.(Ref(problem), results.chain)

trees = backtrack(problem, posterior)

using StatsBase, DataFrames
rectrees = sumtrees(trees, ccd, w)
