using Whale, NewickTree

t = Whale.extree2
# t = readnw("((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
# t = readnw("(((A:1,B:1):0.5),C:1.5);")
n = length(postwalk(t))
r = Whale.DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9)
w = WhaleModel(r, t)
ccd = read_ale("example/example-1/ale", w)
@test logpdf!(w, ccd) == -538.8237337383428
@test ts = backtrack(w, ccd)

@btime logpdf!(w, ccd)
@btime logpdf!(w, ccd[1])

x = randn(Float32, dimension(r.trans))
@time w(x)
@btime logpdf!(w, ccd)

@btime logpdf(w, ccd)

ForwardDiff.gradient(x->logpdf(w(x), ccd[1]), randn(5))



using DynamicHMC, Random, LogDensityProblems, TransformVariables, Distributions

r = Whale.DLWGD(λ=ones(n), μ=ones(n), q=rand(1), η=0.9)
p = Whale.IWIRPrior(Ψ=[0.1 0. ; 0. 0.1], πr=MvNormal(ones(2)))
# p = Whale.IRPrior(πr=MvNormal(ones(2)))
# r = RatesModel(ConstantDLGWGD(λ=0.1, μ=0.1, q=[0.2], η=0.9), fixed=(:κ,))
# p = CRPrior()
w = WhaleModel(r, t)
d = read_ale("example/example-1/ale", w)

problem = WhaleProblem(d, w, p)
LogDensityProblems.logdensity_and_gradient(problem, randn(dimension(r.trans)))

logpdf(p, r(randn(dimension(r.trans))))

wup = DynamicHMC.default_warmup_stages(doubling_stages=3)
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, warmup_stages=wup)
posterior = transform.(Ref(problem), results.chain)

trees = backtrack(problem, posterior)

using StatsBase, DataFrames
rectrees = sumtrees(trees, ccd, w)
