# A test suite for maximum likelihood estimation, which is at the same time
# a simulation 'study' for the core algorithm.
using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree
using Optim, ForwardDiff
using Test, Random, Distributions
import Whale: dlsimbunch, aleobserve

function mlfuns(model, ccd, η, nwgd=0)
    function f(x::Vector{T}) where T
        rates = (λ=exp(x[1]), μ=exp(x[2]), q=x[3:2+nwgd], η=T(η))
        -logpdf(model(rates), ccd)
    end
    g!(G, x) = G .= ForwardDiff.gradient(f, x)
    return f, g!
end

@testset "ML constant rates" begin
    Random.seed!(105)
    t = deepcopy(Whale.extree)
    l = length(getleaves(t))
    θ = ConstantDLWGD(λ=.5, μ=.4, η=0.66)
    r = Whale.RatesModel(θ, fixed=(:η, :p))
    w = WhaleModel(r, t, 0.05)
    ts, ps = dlsimbunch(Whale.root(w), w.rates, 200, condition=:root)
    ale = aleobserve(ts)
    ccd = read_ale(ale, w)
    f, g! = mlfuns(w, ccd, 0.66)
    # And now optimize
    results = optimize(f, g!, randn(2), LBFGS())
    λ, μ = exp.(results.minimizer)
    @info λ, μ
    @test abs(λ - θ.λ) < 0.1*θ.λ
    @test abs(μ - θ.μ) < 0.1*θ.μ
end

@testset "ML a bunch" begin
    Random.seed!(179)
    t = deepcopy(Whale.extree)
    l = length(getleaves(t))
    n = 10
    η = 0.75
    for λ = -3:1:0
        θ = ConstantDLWGD(λ=exp(λ), μ=exp(λ), η=η)
        r = Whale.RatesModel(θ, fixed=(:η, :p))
        w = WhaleModel(r, t, 0.05)
        x = []
        for i=1:n
            ts, ps = dlsimbunch(Whale.root(w), w.rates, 20, condition=:root)
            ale = aleobserve(ts)
            ccd = read_ale(ale, w)
            f, g! = mlfuns(w, ccd, η)
            results = optimize(f, g!, randn(2), LBFGS())
            λ, μ = results.minimizer
            push!(x, (λ=λ, μ=μ))
            rm(ale, recursive=true)
        end
        ls = first.(x)
        ms = last.(x)
        for v in [ls, ms]
            m = round(mean(v), digits=3)
            se = round(std(v)/√n, digits=3)
            @info "$m ($λ) ± $se"
            @test abs(m - λ) < 2*se
        end
    end
end

@testset "ML with WGDs" begin
    Random.seed!(11)
    t = deepcopy(Whale.extree)
    insertnode!(t[1][2], name="wgd_physcomitrella")
    insertnode!(t[2][2][1], name="wgd_angiosperms")
    l = length(getleaves(t))
    n = 5
    N = 50
    η = 0.85
    nwgd = 2
    for λ = [-3., -2.5, -2., -1.3]
        q = rand(2)
        θ = ConstantDLWGD(λ=exp(λ)*1.1, μ=exp(λ), η=η, q=q)
        r = Whale.RatesModel(θ, fixed=(:η, :p))
        w = WhaleModel(r, t, 0.05)
        xs = []
        for i=1:n
            ts, ps = dlsimbunch(Whale.root(w), w.rates, N, condition=:root)
            ale = aleobserve(ts)
            ccd = read_ale(ale, w)
            f, g! = mlfuns(w, ccd, η, nwgd)
            results = optimize(f, g!, [-Inf,-Inf,0.,0.], [Inf,Inf,1.,1.],
                [randn(2) ; rand(2)], Fminbox(LBFGS()))
            push!(xs, results.minimizer)
            rm(ale, recursive=true)
        end
        ref = [[log(exp(λ)*1.1), λ] ; q]
        for i=1:4
            v = [x[i] for x in xs]
            m = round.(mean(v), digits=3)
            se = round(std(v)/√n, digits=3)
            @info "$m ($(ref[i])) ± $se"
            @test abs(m - ref[i]) < 3*se
        end
    end
end

@testset "Nowhere extinct condition" begin
    # This is a tricky business
    Random.seed!(13)
    t = deepcopy(Whale.extree)
    l = length(getleaves(t))
    λ = 0.1
    for μ in [0.05, 0.15]
        θ = ConstantDLWGD(λ=λ, μ=μ, η=0.66)
        r = Whale.RatesModel(θ, fixed=(:η, :p))
        w = WhaleModel(r, t, 0.05, condition=Whale.NowhereExtinctCondition(t))
        ts, ps = dlsimbunch(Whale.root(w), w.rates, 100, condition=:all)
        ale = aleobserve(ts)

        w = WhaleModel(r, t, 0.05, condition=Whale.NowhereExtinctCondition(t))
        ccd = read_ale(ale, w)
        f, g! = mlfuns(w, ccd, 0.66)
        results = optimize(f, g!, randn(2), LBFGS())
        l1, m1 = exp.(results.minimizer)

        w = WhaleModel(r, t, 0.05, condition=Whale.NonExtinctCondition())
        ccd = read_ale(ale, w)
        f, g! = mlfuns(w, ccd, 0.66)
        results = optimize(f, g!, randn(2), LBFGS())
        l2, m2 = exp.(results.minimizer)

        # obviously not guaranteed to be the case
        @test abs(λ - l1) < 0.1*λ
        @test abs(μ - m1) < abs(μ - m2)
        @test abs(μ - m1) < 0.1*μ
        @test m2 < m1  # this should be generally the case, applying the wrong
        # condition tends to lead to underestimation of μ
    end
end
