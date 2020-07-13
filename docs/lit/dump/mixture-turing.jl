# mixture
@model mixture(model, ccd, ::Type{T}=Float64) where T = begin
    K = 2
    r  ~ MvLogNormal(ones(K))
    η  ~ Beta(3,1)
    p  ~ Dirichlet(K, 1.0)
    z = Vector{Int64}(undef, length(ccd))
    models = [model((λ=r[i], μ=r[i], η=η, q=T[])) for i=1:K]
    for i=1:length(ccd)
        z[i]  ~ Categorical(p)
        ccd[i] ~ models[z[i]]
    end
end

bmodel = mixture(w, ccd)
g = Gibbs(HMC(0.1, 5, :r, :η, :p), PG(20, :z))
chain = sample(bmodel, g, 100)
