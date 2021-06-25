using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Parameters, ForwardDiff
using Test, Random, Distributed

@testset "Whale tests" begin
    @testset "Likelihood" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        t = deepcopy(Whale.extree)
        n = length(postwalk(t))
        insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd_1")
        insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")
        r = DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9)

        # super simple, 5 slices, 1 family
        w = WhaleModel(r, t, 0.05, maxn=5)
        ccd = read_ale(data, w)
        l = logpdf(w, ccd[1])
        # -35.739464951792215
        @test l ≈ -59.03141590814281

        w = WhaleModel(r, t, 0.05, maxn=10000)
        ccd = read_ale(data, w)
        # two threads
        #julia> @btime logpdf($w, $ccd)  non-log scale
        #  220.969 μs (252 allocations: 290.95 KiB)
        #julia> @btime logpdf($w, $ccd)
        #  1.263 ms (252 allocations: 297.09 KiB)
        # slowdown exclusively due to logaddexp I fear...
        # unconditioned -614.1255864680994
        l = -570.9667405899105
        @test logpdf!(w, ccd) ≈ l
        @test logpdf(w, ccd)  ≈ l
        
        g(x) = logpdf(w(DLWGD(λ=x[1:n], μ=x[n+1:2n], q=x[2n+1:2n+3], η=x[end])), ccd)
        x = [ones(n) ; ones(n); [0.1, 0.2] ; 0.8]
        G = ForwardDiff.gradient(g, x)
        @test all(isfinite.(G))
        # two threads
        #julia> @btime ForwardDiff.gradient($g, $x);
        #  2.723 ms (2240 allocations: 12.28 MiB)

        # root rates should not have an effect
        r.λ[id(getroot(w))] = NaN
        r.μ[id(getroot(w))] = NaN
        w = WhaleModel(r, t, 0.05, maxn=10000)
        ccd = read_ale(data, w)
        @test logpdf!(w, ccd) ≈ l
    end

    @testset "Conditioning - nowhere extinct" begin
        t = deepcopy(Whale.extree)
        insertnode!(t[1][1], name="wgd_1")
        insertnode!(t[1][2][1], name="wgd_2")
        p = -Inf
        # the higher the retention rates, the higher the probability of non-ext
        for q = 0:0.1:1
            r = ConstantDLWGD(λ=0.3, μ=0.4, q=[q,q], η=0.66)
            w = WhaleModel(r, t, 0.05, condition=Whale.NowhereExtinctCondition(t))
            @test Whale.condition(w) > p
            p = Whale.condition(w)
        end

        # Monte Carlo based verification
        Random.seed!(123)
        n = length(postwalk(t)) -2
        for i=1:10
            θ = DLWGD(λ=randn(n) .- 3, μ=randn(n) .- 3, q=rand(2), η=0.67)
            w = WhaleModel(θ, t, 0.05,
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
        r = DLWGD(λ=ones(n), μ=ones(n), η=0.9)
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
        r = DLWGD(λ=ones(n), μ=ones(n), q=[0.2, 0.1], η=0.9)
        w = WhaleModel(r, t, 0.05)
        ccd = read_ale(data, w)
        logpdf!(w, ccd)
        out = [Whale.backtrack(w, ccd[1]) for i=1:100]
        rsum = sumtrees(out, ccd[1], w)
        df = Whale.getpairs([rsum], w)
        @test all(isapprox.(map(sum, eachrow(df[!,1:end-2])), Ref(1.)))
    end

    @testset "MUL trees" begin
        data = joinpath(@__DIR__, "../example/example-1/ale")
        multree = readnw("((MPOL:4.752,PPAT:4.752):0.292,((SMOE:4.0"*
                         ",PPAT:4.0):0.457,(((OSAT:1.555,(ATHA:0.55"*
                         "48,CPAP:0.5548):1.0002):0.738,ATRI:2.293)"*
                         ":1.225,(GBIL:3.178,PABI:3.178):0.34):0.93"*
                         "9):0.587);")
        n = length(postwalk(multree))
        r = DLWGD(λ=zeros(n).-1, μ=zeros(n).-1, η=0.9)
        w = WhaleModel(r, multree, 0.05)
        ccd = read_ale(data, w)
        @test isfinite(logpdf!(w, ccd))
        ts = hcat([Whale.backtrack(w, ccd) for i=1:100]...)
        rs = Whale.sumtrees(permutedims(ts), ccd, w)
        tables = Whale.gettables(rs, leaves=true)
    end

    @testset "Discretization" begin
        t = readnw(readline(joinpath(@__DIR__, "../example/example-5/tree.nw")))
        d = joinpath(@__DIR__, "../example/example-5/OG0014587.ale")
        for n in prewalk(t); n.data.distance = 1.; end
        n = length(postwalk(t))
        for i=1:10
            r = randn(n)
            r = DLWGD(λ=r, μ=r, η=0.9)
            ℓ = map(-5:0.5:-1) do n
                w = WhaleModel(r, t, 10^n)
                data = read_ale(d, w)
                #@show n, logpdf!(w, data)
                logpdf!(w, data)
            end
            @test all(abs.(ℓ .- ℓ[1]) .< 0.1)
        end
    end
end

