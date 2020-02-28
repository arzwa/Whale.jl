# # Sampling from the prior

# Here I inspect particular prior paraeterizations for Whale by running the HMC
# sampler in the absence of data.

using DynamicHMC, Whale, DistributedArrays, Distributions, Random
using DynamicHMC.Diagnostics, Test
using Plots, StatsPlots

# ## Constant rates prior
# Specify the problem without data
wm = WhaleModel(Whale.extree, Δt=0.1)
prior = CRPrior(
    πr=MvNormal(ones(2)),
    πη=Beta(3,1))
problem = WhaleProblem(wm, nothing, prior)

# run DynamicHMC
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 1000)
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

x = [x.λ for x in posterior]
@testset "CRPrior" begin
    @test isapprox(mean(x), 0., atol=0.1)
    @test isapprox(var(x), 1., atol=0.1)
end

# ## Independent rates prior

# This is a non-hierarchical independent rates prior, so just an iid bivariate
# log-normal density prior on branch-wise rates, instead of a bivariate prior on
# the tree-wide duplication and loss rates.
wm = WhaleModel(Whale.extree, Δt=0.1)
prior = IRPrior(
    πr=MvNormal(ones(2), ones(2)./2),
    πη=Beta(3,1))
problem = WhaleProblem(wm, nothing, prior)

# run DynamicHMC
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 1000,
    initialization = (ϵ = 1.0, ),
    warmup_stages = default_warmup_stages(; stepsize_search = nothing))
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

λ1 = [x.r[1,4] for x in posterior]
μ8 = [x.r[2,8] for x in posterior]
@testset "IRPrior" begin
    @test isapprox(mean(λ1), 1., atol=0.1)
    @test isapprox(mean(μ8), 1., atol=0.1)
    @test isapprox(std(λ1), 0.5, atol=0.1)
    @test isapprox(std(μ8), 0.5, atol=0.1)
end

# ## Inverse-Wishart independent rates prior

# This is a hierarchical prior (or multi-level model), where we assume the
# branch-wise rates are assigned a bivariate Normal distribution with
# a mean vector $\theta_0$ that is itself distributed according to a bivariate
# Normal and a covariance matrix that is assigned an Inverse-wishart prior.
# Note that we make use of the conjugate relation of the multivariate Normal
# and the Inverse-Wishart distribution to integrate out the covariance matrix,
# i.e. we marginalize the posterior over the unknown covariance matrix $\Sigma$.
# This means we are not tracking the variance and covariance parameters explicitly
# during the MCMC.
wm = WhaleModel(Whale.extree, Δt=0.1)
addwgd!(wm, wm[4], wm[4].event.t*0.5)
prior = Whale.IWIRPrior(
    Ψ =[1. 0. ; 0. 1.],
    πr=MvNormal(ones(2)),
    πη=Beta(3,1))
problem = WhaleProblem(wm, nothing, prior)

# run DynamicHMC
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 5000)
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

df = Whale.unpack(posterior)
plot(stephist(df[!,:η],     fill=true, alpha=0.5, xlabel="\\eta"),
     stephist(df[!,:q1],    fill=true, alpha=0.5, xlabel="q"),
     stephist(df[!,:r1_9],  fill=true, alpha=0.5, xlabel="\\lambda_9"),
     stephist(df[!,:r2_1],  fill=true, alpha=0.5, xlabel="\\mu_1"),
     grid=false, legend=false, color=:black)

@testset "IRPrior" begin
    @test isapprox(mean(df[!,:r2_1]), 0., atol=0.1)
    @test isapprox(mean(df[!,:r1_9]), 0., atol=0.1)
    @test isapprox(mean(df[!,:q1]), 0.5, atol=0.1)
end


# ## LKJ independent rates prior

# This is a similar hierarchical prior on the branch-rates as the Inverse-Wishart
# based hierarchical prior. However, here we specify a prior for the covariance
# matrix $\Sigma$ by assigning a prior to the scale $\tau$ and the correlation
# matrix $R$. We use a LKJ prior for $R$ and any univariate distribution with
# positive support for $\tau$. The covariance matrix is then given by $\Sigma =
# \tauI \times R \tauI$. For more infor, refer to for instance [the stan manual
# ](https://mc-stan.org/docs/2_22/stan-users-guide/multivariate-hierarchical-priors-section.html)

# Note that the parameter of the LKJ prior (usually called $\eta$, but we'll
# call it $\omega$) can be interpreted similarly as the $\alpha$ parameter of a
# Dirichlet distribution, with $\omega > 1$ favoring less correlation (increasing
# mass to the identty matrix) and $\omega < 1$ favoring more correlation.
# In our case we use a bivariate distribution, so the LKJ prior reduces in fact
# to a prior on the correlation coefficient.

# Note that in contrast with Inverse-Wishart based prior, we here do sample the
# covariance matrix (decomposed as a scale and correlation factor) explicitly.
wm = WhaleModel(Whale.extree, Δt=0.1)
prior = Whale.LKJIRPrior(
    πR=Whale.LKJCorr(.1),
    πτ=Exponential(5.),
    πr=MvNormal([0.5, -1.2], ones(2)),
    πη=Beta(3,1))
problem = WhaleProblem(wm, nothing, prior)

# run DynamicHMC
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 5000)
posterior = Whale.transform.(problem.trans, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

df = Whale.unpack(posterior)
plot(stephist(df[!,:U1_2],  fill=true, alpha=0.5, xlabel="\\rho"),
     stephist(df[!,:τ],     fill=true, alpha=0.5, xlabel="\\tau"),
     stephist(df[!,:r1_1],  fill=true, alpha=0.5, xlabel="\\lambda_1"),
     stephist(df[!,:r2_8],  fill=true, alpha=0.5, xlabel="\\mu_8"),
     grid=false, legend=false, color=:black)

@testset "LKJPrior" begin
    @test isapprox(mean(df[!,:r1_1]),  0.5, atol=0.1)
    @test isapprox(mean(df[!,:r2_8]), -1.2, atol=0.1)
    @test isapprox(mean(df[!,:U1_2]),   0.0, atol=0.1)
    @test isapprox(mean(df[!,:τ]), mean(prior.πτ), atol=0.3)
end

# this seems to be a somewhat trickier prior.
