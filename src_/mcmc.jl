# Whale MCMC engine
# The specific nature of the problem (especially the tree structure) makes
# inference using generic engines (Turing, Mamba) rather cumbersome.

#=
ν ~ InverseGamma
η ~ Beta
q ~ [Beta]
θ ~ MvLogNormal
λ ~ GBM
μ ~ GBM

How can we implement this in e.g. Turing, with the same speed-ups as the custom
implementation?

We might need to define a different Sampler object for the rates that uses the
partial recomputation scheme. This should implement adaptation.

Similarly for η
=#

@model gbmwhale(x::CCDArray, s::SlicedTree) = begin
    ν ~ InverseGamma(10.)
    η ~ Beta(10, 2)
    q ~ [Beta(1, 1) for i=1:nwgd(s)]
    θ ~ MvLogNormal([log(0.2), log(0.2)])
end
