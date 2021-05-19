using Distributions, Test, DynamicHMC, Random

@testset "Log density and gradient" begin
    t = deepcopy(Whale.extree)
    n = length(postwalk(t))
    insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_1")

    @testset "CR prior" begin
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), w, true)
        prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(D, w, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))
        shouldbe = [-286.51921458, 182.41911264, -1.07830352, -1.34490497]
        @test p ≈ -447.84586660
        @test all(∇p .≈ shouldbe)
    end

    @testset "IWIR prior" begin
        r = RatesModel(DLWGD(λ=ones(n), μ=ones(n), q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), w, true)
        prior = Whale.IWIRPrior(Ψ=[1. 0.; 0. 1.])
        problem = WhaleProblem(D, w, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(36))
        @test p ≈ -452.73871886
        @test all(∇p[3:5] .≈ [-34.0031273, -37.39635692, -35.3375338])
        @test all(∇p[end-2:end] .≈ [0.0, -1.34490497, -0.07830352])
    end

    @testset "DHMC prior sampling" begin
        Random.seed!(9217)
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(nothing, w, prior)
        results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 500,
            reporter=NoProgressReport(),
            warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
        post = Whale.transform(problem, results.chain)
        @test isapprox((map(x->x.λ, post) |> mean), mean(prior.πr)[1], atol=0.5)
        @test isapprox((map(x->x.η, post) |> mean), mean(prior.πη), atol=0.2)
    end

    @testset "DHMC sampling" begin
        Random.seed!(2971)
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), w, true)
        prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(D, w, prior)
        results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
            reporter=NoProgressReport(),
            warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
        posterior = Whale.transform(problem, results.chain)
        @test isapprox((map(x->x.λ, posterior) |> mean), 0.13, atol=0.05)
        @test isapprox((map(x->x.η, posterior) |> mean), 0.7, atol=0.1)
    end
end
