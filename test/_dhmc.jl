using Test
using Distributed
# addprocs(3)
@everywhere using Pkg
@everywhere Pkg.activate("./test/")
@everywhere begin
    using Whale, DistributedArrays, Distributions, DynamicHMC, Random
end
using TransformVariables

@testset "DynamicHMC" begin
    wm = WhaleModel(Whale.extree)
    Whale.addwgd!(wm, 5, 0.25, rand())
    D = distribute(read_ale("./example/example-ale", wm)[1:2])

    @testset "CR prior" begin
        prior = Whale.CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
        problem = Whale.WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))
        @test p ≈ -78.89706608972695
        @test all(∇p .≈ [15.017958801444323, -35.95111615485038,
            0.8885559632495077, 0.6181645286506255])
    end

    @testset "IR prior" begin
        prior = Whale.IRPrior(Ψ=[1. 0.; 0. 1.])
        problem = Whale.WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(36))
    end
end



# ConstantRates
wm = WhaleModel(Whale.extree)
addwgd!(wm, 5, 0.25, rand())
D = distribute(read_ale("./example/example-ale", wm))

prior = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
problem = WhaleProblem(wm, D, prior)
@show Whale.logdensity_and_gradient(problem, zeros(4))

# BranchRates
prior = IRPrior(Ψ=[.2 0.; 0. .2])
problem = WhaleProblem(wm, D, prior)
@show Whale.logdensity_and_gradient(problem, zeros(36))

plog = LogProgressReport(step_interval=100, time_interval_s=10)
@time results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 500, reporter=plog)

posterior = TransformVariables.transform.(problem.trans, results.chain)
l = [x.r[2,7] for x in posterior]
m = [x.r[1,4] for x in posterior]
q = [x.q[1] for x in posterior]
e = [x.η for x in posterior]

plot(plot(l), plot(m), plot(q), plot(e), legend=false, grid=false)


function to_csv(io::IO, results, problem)
    post = TransformVariables.transform.(problem.trans, results.chain)
    for k in keys(post[1])
        typeof(post[1].k)<:AbstractArray ? write(io, join([
            "$k$i" for i=1:length(post[1].k)]), ",") : write(io, "$k")
    end
end
