# # Bayesian inference using `Turing.jl`
using Whale, NewickTree, Distributions, Turing, DataFrames, LinearAlgebra

# ## Using a constant-rates model
# Get the tree
t = deepcopy(Whale.extree)
n = length(postwalk(t))  # number of internal nodes

# Now we add two WGD nodes to the tree. We do this by specifying
# the last common ancestor node for the lineages that share the
# hypothetical WGD. By default, the added node is halfway between
# the specified node and its parent.
insertnode!(getlca(t, "PPAT", "PPAT"), name="wgd_1")
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd_2")

# and we obtain a reference model object, here we will use a constant-rates
# model
θ = ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2, 0.1], η=0.9)
r = Whale.RatesModel(θ, fixed=(:p,))
w = WhaleModel(r, t, .1)

# next we get the data (we need a model object for that)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)

# Now we define the Turing model
@model constantrates(model, ccd) = begin
    r  ~ MvLogNormal(ones(2))
    η  ~ Beta(3,1)
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r[1], μ=r[2], η=η, q=[q1, q2]))
end

model = constantrates(w, ccd)
chain = sample(model, NUTS(0.65), 500)

# ## Using a branch-specific rates model
# We'll use the same tree as above. The relevant model now is
# the DLWGD model:

params = DLWGD(λ=randn(n), μ=randn(n), q=rand(2), η=rand())
r = Whale.RatesModel(params, fixed=(:p,))
w = WhaleModel(r, t, 0.5)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)

# Note that the duplication and loss rates should here be specified on a
# log-scale for the DLWGD model.

@model branchrates(model, ccd, ::Type{T}=Matrix{Float64}) where {T} = begin
    η ~ Beta(3,1)
    ρ ~ Uniform(-1, 1.)
    τ ~ truncated(Cauchy(0, 1), 0, Inf)
    S = [τ 0. ; 0. τ]
    R = [1. ρ ; ρ 1.]
    Σ = S*R*S
    !isposdef(Σ) && return -Inf
    r = T(undef, 2, n)
    r[:,1] ~ MvNormal(zeros(2), ones(2))
    for i=2:n
        r[:,i] ~ MvNormal(r[:,1], Σ)
    end
    q1 ~ Beta()
    q2 ~ Beta()
    ccd ~ model((λ=r[1,:], μ=r[2,:], η=η, q=[q1, q2]))
end

model = branchrates(w, ccd)
chain = sample(model, NUTS(0.65), 100)

# !!! warning
#     Of course such a chain should be run much longer than in this example! Here a very short chain is presented to ensure reasonable build times for this documentation.

# Let's obtain reconciled trees
pdf = DataFrame(chain)
fun = (m, x)-> Array(x) |> x->m((λ=x[3:2:36], μ=x[4:2:36], η=x[end-2], q=x[1:2]))
tt = TreeTracker(w, ccd[end-1:end], pdf, fun)
trees = track(tt)

# Let's have a look at the first family
trees[1].trees

# Or maybe all the gene pairs
ps = Whale.getpairs(trees, w);

# Now let's look at the gene pairs which have a non-zero posterior probability of being derived from WGD node 18 (the *Arabidopsis* WGD, execute `@show w` to check the model structure)
p = filter(x->x[Symbol("18_wgd")] > 0.0, ps)[!,:pair]

# The full (approximate) probability distribution over reconciliation events for this gene pair is
row = ps[ps[!,:pair] .== p[1],1:end-2]
for (k,v) in zip(names(row), Array(row))
    v > 0. && println(k, ": ", v)
end

# The following can also be helpful
tables = Whale.getwgdtables(trees, ccd, w)
tables[1]
