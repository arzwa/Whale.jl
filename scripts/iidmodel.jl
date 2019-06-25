#=
A Turing model definition for a hierarchical independent DL rates model.
=#
using Whale
using Turing

st = Whale.example_tree()
ccd = read_ale("/home/arzwa/Whale.jl/example/example-ale/", st, d=false)

@model iidwhale(x) = begin
    ν ~ InverseGamma(100.)
    η ~ Beta(10, 2)
    q = Vector{Real}(undef, nwgd(st))
    for i in eachindex(q)
        q[i] ~ Beta(1, 1)
    end
    r ~ Exponential(0.2)
    θ ~ MvLogNormal([log(r), log(r)], [.5 0.45 ; 0.45 0.5])
    λ ~ MvLogNormal(repeat([log(θ[1])], nrates(st)), ones(nrates(st)))
    μ ~ MvLogNormal(repeat([log(θ[2])], nrates(st)), ones(nrates(st)))
    x ~ [WhaleModel(st, λ, μ, float.(q), η)]  # vectorized
end
chain = sample(turingmodel, HMC(1000, 0.0001, 1))
