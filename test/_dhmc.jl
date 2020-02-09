using Test
using Distributed
# addprocs(3)
@everywhere using Pkg
@everywhere Pkg.activate("./test/")
@everywhere begin
    using Whale, DistributedArrays, Distributions, DynamicHMC, Random
end

wm = Whale.WhaleModel(Whale.extree)
addwgd!(wm, wm[5], 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm)[1:10])

# ConstantRates
prior = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(wm, D, prior)
@time Whale.logdensity_and_gradient(problem, zeros(4))

# BranchRates
prior = IRPrior(Ψ=[.2 0.; 0. .2])
problem = WhaleProblem(wm, D, prior)
@time Whale.logdensity_and_gradient(problem, zeros(36))

# LogDensityProblems style - for comparisons (they turn out to be comparable
# in single core setting)
(p::WhaleProblem)(θ) = logpdf(p.prior, θ) + logpdf(p.model(BranchRates(θ)), p.data)
P = TransformedLogDensity(problem.trans, problem)
∇P = ADgradient(:ForwardDiff, P)

plog = LogProgressReport(step_interval=100, time_interval_s=10,
    logger=SimpleLogger(open("log.txt", "w+")))

plog = LogProgressReport(step_interval=100, time_interval_s=10)

@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, reporter=plog)

@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100, reporter=plog,
    initialization = (ϵ=0.1,),
    warmup_stages = fixed_stepsize_warmup_stages(
        middle_steps=20, doubling_stages=2))

posterior = TransformVariables.transform.(problem.trans, results.chain)
l = [x.r[2,7] for x in posterior]
m = [x.r[1,4] for x in posterior]
q = [x.q[1] for x in posterior]
e = [x.η for x in posterior]

plot(e)

wmm = wm(ConstantRates(posterior[5]))
logpdf!(wmm, D)
R = backtrack(wmm, D)
