using Distributions, Test

@testset "Log density and gradient" begin
    wm = WhaleModel(Whale.extree1)
    D = read_ale(joinpath(@__DIR__, "../example/example-1/ale"), wm)
    D = DArray(D[1:2])

    @testset begin
        n = length(postwalk(t))
        r = RatesModel(DLGWGD(λ=ones(n), μ=ones(n), q=[0.2], η=0.9))
        w = WhaleModel(r, t)
        ccd = read_ale("example/example-1/ale", w)
        @test logpdf!(w, ccd) == -399.4124195572149
    end

    @testset "CR prior" begin
        prior = CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
        problem = WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))
        @test p ≈ -78.89706608972695
        @test all(∇p .≈ [15.01795880, -35.95111615, 0.888555963, 0.61816452])
    end

    @testset "IR prior" begin
        prior = Whale.IWIRPrior(Ψ=[1. 0.; 0. 1.])
        problem = WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(36))
        @test p ≈ -83.50223627571505
        @test all(∇p[3:6] .≈ [1.188753499, 0.716933557562, 1.0, 0.5572538127])
        @test all(∇p[end-2:end] .≈ [-5.665618052, -0.1114440367, 1.6181645286])
    end
end
