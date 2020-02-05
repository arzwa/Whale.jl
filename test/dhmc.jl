using Distributions

@testset "Priors" begin
    wm = WhaleModel(Whale.extree)
    Whale.addwgd!(wm, 5, 0.25, rand())
    D = distribute(read_ale("../example/example-ale", wm)[1:2])

    @testset "CR prior" begin
        prior = Whale.CRPrior(MvNormal(ones(2)), Beta(3,1), Beta())
        problem = Whale.WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(4))
        @test p ≈ -78.89706608972695
        @test all(∇p .≈ [15.01795880, -35.95111615, 0.888555963, 0.61816452])
    end

    @testset "IR prior" begin
        prior = Whale.IRPrior(Ψ=[1. 0.; 0. 1.])
        problem = Whale.WhaleProblem(wm, D, prior)
        p, ∇p = Whale.logdensity_and_gradient(problem, zeros(36))
        @test p ≈ -83.50223627571505
        @test all(∇p[3:6] .≈ [1.188753499, 0.716933557562, 1.0, 0.5572538127])
        @test all(∇p[end-2:end] .≈ [-5.665618052, -0.1114440367, 1.6181645286])
    end
end
