# A test suite for maximum likelihood estimation, which is at the same time
# a simulation 'study' for the core algorithm.
using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, FakeFamily
using Optim, ForwardDiff
using Test, Random, Distributions
import FakeFamily: dlsimbunch, aleobserve

function mlfuns(model, ccd, η, l)
    function f(x::Vector{T}) where T
        rates = (λ=exp(x[1]), μ=exp(x[2]), q=T[], η=T(η), p=zeros(T,l))
        -logpdf(model(rates), ccd)
    end
    g!(G, x) = G .= ForwardDiff.gradient(f, x)
    return f, g!
end

@testset "ML constant rates" begin
    Random.seed!(105)
    t = Whale.extree
    l = length(getleaves(t))
    θ = ConstantDLWGD(λ=.5, μ=.4, η=0.66)
    r = Whale.RatesModel(θ, fixed=(:η, :p))
    w = WhaleModel(r, t, 0.05)
    ts, ps = dlsimbunch(Whale.root(w), w.rates, 200)
    ale = aleobserve(ts)
    ccd = read_ale(ale, w)
    f, g! = mlfuns(w, ccd, 0.66, l)
    # And now optimize
    results = optimize(f, g!, randn(2), LBFGS())
    λ, μ = exp.(results.minimizer)
    @info λ, μ
    @test abs(λ - θ.λ) < 0.1*θ.λ
    @test abs(μ - θ.μ) < 0.1*θ.μ
end

@testset "ML a bunch" begin
    Random.seed!(179)
    t = Whale.extree
    l = length(getleaves(t))
    n = 10
    η = 0.75
    for λ = -3:1:0
        θ = ConstantDLWGD(λ=exp(λ), μ=exp(λ), η=η)
        r = Whale.RatesModel(θ, fixed=(:η, :p))
        w = WhaleModel(r, t, 0.05)
        x = []
        for i=1:n
            ts, ps = dlsimbunch(Whale.root(w), w.rates, 20)
            ale = aleobserve(ts)
            ccd = read_ale(ale, w)
            f, g! = mlfuns(w, ccd, η, l)
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
