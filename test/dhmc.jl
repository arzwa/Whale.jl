using Distributions, Test, DynamicHMC

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
        shouldbe = [-276.061487822, 160.141680772, -7.199728049, -1.26771978]
        @test p ≈ -462.771909465
        @test all(∇p .≈ shouldbe)
    end

    @testset "IR prior" begin
        r = RatesModel(DLWGD(λ=ones(n), μ=ones(n), q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), w, true)
        prior = Whale.IWIRPrior(Ψ=[1. 0.; 0. 1.])
        problem = WhaleProblem(D, w, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(36))
        @test p ≈ -467.66476172315
        @test all(∇p[3:5] .≈ [-32.807670074, -37.277598588, -35.218775468])
        @test all(∇p[end-2:end] .≈ [0.0, -1.26771978, -6.199728049])
    end

    @testset "DHMC prior sampling" begin
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(nothing, w, prior)
        results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 500,
            reporter=NoProgressReport(),
            warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
        post = Whale.transform.(Ref(problem), results.chain)
        @test isapprox((map(x->x.λ, post) |> mean), mean(prior.πr)[1], atol=0.2)
        @test isapprox((map(x->x.η, post) |> mean), mean(prior.πη), atol=0.1)
    end

    @testset "DHMC sampling" begin
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        w = WhaleModel(r, t, 0.1)
        D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), w, true)
        prior = CRPrior(MvLogNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(D, w, prior)
        results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
            reporter=NoProgressReport(),
            warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
        posterior = Whale.transform.(Ref(problem), results.chain)
        @test isapprox((map(x->x.λ, posterior) |> mean), 0.13, atol=0.05)
        @test isapprox((map(x->x.η, posterior) |> mean), 0.7, atol=0.1)
    end
end
