
# ## Inference for the constant rates model
# Required libraries
using Whale, NewickTree, Distributions, Turing

# Get the tree
t = deepcopy(Whale.extree)
n = length(postwalk(t))  # number of internal nodes
l = (n+1)÷2  # number of leaf nodes

# Now we add two WGD nodes to the tree. We do this by specifying
# the last common ancestor node for the lineages that share the
# hypothetical WGD. By default, the added node is halfway between 
# the specified node and its parent.
insertnode!(getlca(t, "ATHA", "ATHA"), name="wgd")
insertnode!(getlca(t, "ATHA", "ATRI"), name="wgd")

# and we obtain a reference model object
r = Whale.RatesModel(
    ConstantDLWGD(λ=0.1, μ=0.2, q=[0.2, 0.1], η=0.9, p=zeros(l)),
    fixed=(:p,))
w = WhaleModel(r, t)

# next we get the data (we need a model object for that)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)

# Now we define the Turing model
@model constantrates(model, ccd) = begin
    r ~ MvLogNormal(ones(2))
    q1 ~ Beta()
    q2 ~ Beta()
    η ~ Beta(3,1)
    ccd ~ model((λ=r[1], μ=r[2], η=η, q=[q1, q2]))
end

model = constantrates(w, ccd)
chain = sample(model, NUTS(0.65), 1000)

# ## Inference for the branch-rates model

# We'll use the same tree as above. The relevant model now is
# the DLWGD model:
r = Whale.RatesModel(
        DLWGD(λ=zeros(n), μ=zeros(n), 
            q=[0.2, 0.1], η=0.9, p=zeros(l)),
        fixed=(:p,))
w = WhaleModel(r, t)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), w)

# Note that the duplication and loss rates should here be specified on a
# log-scale for the DLWGD model.

@model branchrates(model, ccd, ::Type{T}=Matrix{Float64}) where {T} = begin
    η ~ Beta(3,1)
    Σ ~ InverseWishart(3, [1. 0. ; 0. 1.0])
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
chain = sample(model, NUTS(0.65), 1000)
