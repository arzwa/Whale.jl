using Test
using Distributed
addprocs(3)
@everywhere using Pkg
@everywhere Pkg.activate("./test/")
@everywhere begin
    using DistributedArrays, SharedArrays, Random, Flux, ForwardDiff
end

using BenchmarkTools

M = Matrix{Float64}[rand(100,100) for i = 1:100];

@everywhere function testfun(M, θ)
    sum(pmap(x->θ'*x*θ, M))/length(θ)
end

function grad(M, θ)
    f = (x) -> testfun(M, x)
    ForwardDiff.gradient(f, θ)
end
