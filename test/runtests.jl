using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Parameters
using Test
using Random

@testset "Whale tests" begin
    @testset "likelihood" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
        insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
        r = RatesModel(
                DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9), fixed=(:p,))
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        # @test logpdf!(w, ccd) ≈ -588.0647568294178
        # @test logpdf(w, ccd) ≈ -588.0647568294178
        @test logpdf!(w, ccd) ≈ -570.9667405899105
        @test logpdf(w, ccd) ≈ -570.9667405899105
        ccd = read_ale(data, w, true)
        # @test logpdf!(w, ccd) ≈ -588.0647568294178
        # @test logpdf(w, ccd) ≈ -588.0647568294178
        @test logpdf!(w, ccd) ≈ -570.9667405899105
        @test logpdf(w, ccd) ≈ -570.9667405899105

        # root rates do not have an effect
        r.params.λ[id(getroot(w))] = NaN
        r.params.μ[id(getroot(w))] = NaN
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        @test logpdf!(w, ccd) ≈ -570.9667405899105
    end

    @testset "Conditioning - nowhere extinct" begin
        t = deepcopy(Whale.extree)
        insertnode!(t[1][1], name="wgd_1")
        insertnode!(t[1][2][1], name="wgd_2")
        p = -Inf
        # the higher the retention rates, the higehr the probability of non-ext
        for q = 0:0.1:1
            r = RatesModel(ConstantDLWGD(λ=0.3, μ=0.4, q=[q,q], η=0.66))
            w = WhaleModel(r, t, 0.05,
                condition=Whale.NowhereExtinctCondition(t))
            @test Whale.condition(w) > p
            p = Whale.condition(w)
        end

        # Monte Carlo based verification
        Random.seed!(123)
        n = length(postwalk(t)) -2
        for i=1:10
            θ = DLWGD(λ=randn(n) .- 3, μ=randn(n) .- 3, q=rand(2), η=0.67)
            r = Whale.RatesModel(θ, fixed=(:p,))
            w = WhaleModel(r, t, 0.05,
                    condition=NowhereExtinctCondition(t))
            p1 = exp(Whale.condition(w))
            ts, ps = FakeFamily.dlsimbunch(w, 1000, condition=:none);
            c = map(x->all(Array(x) .> 0), eachrow(ps))
            p2 = sum(c)/1000
            @test isapprox(p1, p2, rtol=0.1)
        end
    end

    # These tests rely on ALEobserve being available and in the $PATH
    include(joinpath(@__DIR__, "mle.jl"))

    @testset "set sampling probabilities" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        r = RatesModel(DLWGD(λ=ones(n), μ=ones(n), η=0.9), fixed=(:p,))
        w = WhaleModel(r, t, 0.05)
        d = Dict(name(n)=>rand() for n in getleaves(t))
        Whale.setsamplingp!(w, d)
        for n in getleaves(getroot(w))
            @unpack p = Whale.getθ(w.rates, n)
            @test p == d[name(n)]
        end
    end

    @testset "Backtracking, pairwise event PPs" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
        insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
        r = RatesModel(DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9))
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        logpdf!(w, ccd)
        out = [Whale.backtrack(w, ccd[1]) for i=1:100]
        rsum = sumtrees(out, ccd[1], w)
        df = Whale.getpairs([rsum], w)
        @test all(isapprox.(map(sum, eachrow(df[!,1:end-2])), Ref(1.)))
    end
end

writedlm("/tmp/test.csv", eachrow(df), ",")
