# # Maximum likelihood estimation - simulated data

# Here I'll show a quick example on how to perform Maximum likelihood parameter
# estimation under a constant-rates duplication and loss model. The data was
# simulated under a constant rates model with λ = μ = 1, so we now what we
# should find

using Whale, Optim, ForwardDiff

# Get species tree and data
t = readnw(readline(joinpath(@__DIR__, "../../example/example-2/tree.nw")))
l = length(getleaves(t))

params = ConstantDLWGD(λ=0.1, μ=0.2, q=Float64[], η=0.9, p=zeros(l))
r = Whale.RatesModel(params, fixed=(:p,))
model = WhaleModel(r, t, .1)
ccd = read_ale(joinpath(@__DIR__, "../../example/example-2/ale"), model)

# Define objective and gradient (note that Optim looks for minima, so we use
# -ℓ as objective)
function objective(x::Vector{T}, η=0.9) where T
    rates = (λ=exp(x[1]), μ=exp(x[2]), q=T[], η=T(η), p=zeros(T,l))
    -logpdf(model(rates), ccd)
end
f = (x) -> objective(x, 0.9)  # we fix η = 0.8
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)

# And now optimize
init = rand(2)
lower = [-Inf, -Inf]   # lower bounds λ, μ and q
upper = [Inf, Inf]    # upper bounds λ, μ and q
results = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results.minimum, results.minimizer

@info exp.(results.minimizer)

# Note that we can already retrieve the true values quite nicely for this rather
# small data set (100 families)
