using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Parameters
using Test, Random, Distributed

const ALEOBSERVE = false  # do tests requiring `ALEobserve` in the path
const DHMC = true         # do DynamicHMC related tests
const DISTRIBUTED = true

@testset "Whale tests" begin
    @testset "Likelihood" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
        insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
        r = RatesModel(
                DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9), fixed=(:p,))
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        @test logpdf!(w, ccd) ≈ -570.9667405899105
        @test logpdf(w, ccd) ≈ -570.9667405899105
        ccd = read_ale(data, w, true)
        @test logpdf!(w, ccd) ≈ -570.9667405899105
        @test logpdf(w, ccd) ≈ -570.9667405899105

        # root rates should not have an effect
        r.params.λ[id(getroot(w))] = NaN
        r.params.μ[id(getroot(w))] = NaN
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        @test logpdf!(w, ccd) ≈ -570.9667405899105
    end

    @testset "All different likelihood routines" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
        r = RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2], η=0.9))
        x = randn(4)
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        ccd_da = read_ale(data, w, true)  # DArray
        l1 = logpdf!(w(x), ccd)
        l2 = logpdf(w(x), ccd)
        l3 = logpdf!(w(x), ccd_da)
        l4 = logpdf(w(x), ccd_da)
        l5 = Whale.fand∇f(w(x), ccd_da, x)
        @test l1 ≈ l2 ≈ l3 ≈ l4 ≈ l5[1]
        if DISTRIBUTED
            addprocs(2)
            @everywhere using Pkg
            @everywhere Pkg.activate(@__DIR__)
            @everywhere using Whale
            ccd_da = read_ale(data, w, true)  # DArray
            l6 = logpdf!(w(x), ccd_da)
            l7 = logpdf(w(x), ccd_da)
            l8 = Whale.fand∇f(w(x), ccd_da, x)
            @test l6 ≈ l7 ≈ l8[1] ≈ l1
            rmprocs(workers())
            @test length(workers()) == 1
        end
    end

    @testset "Conditioning - nowhere extinct" begin
        t = deepcopy(Whale.extree)
        insertnode!(t[1][1], name="wgd_1")
        insertnode!(t[1][2][1], name="wgd_2")
        p = -Inf
        # the higher the retention rates, the higher the probability of non-ext
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
                    condition=Whale.NowhereExtinctCondition(t))
            p1 = exp(Whale.condition(w))
            ts, ps = Whale.dlsimbunch(w, 1000, condition=:none);
            c = map(x->all(Array(x) .> 0), eachrow(ps))
            p2 = sum(c)/1000
            @test isapprox(p1, p2, rtol=0.1)
        end
    end

    @testset "Set sampling probabilities" begin
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

    ALEOBSERVE && include(joinpath(@__DIR__, "mle.jl"))
    DHMC && include(joinpath(@__DIR__, "dhmc.jl"))
end
