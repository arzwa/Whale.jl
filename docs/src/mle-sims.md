
# Maximum likelihood estimation - simulated data

Here I'll show a quick example on how to perform Maximum likelihood parameter
estimation under a constant-rates duplication and loss model. The data was
simulated under a constant rates model with λ = μ = 1, so we now what we
should find

```@example mle-sims
using Whale, Optim, ForwardDiff
```

Get species tree and data

```@example mle-sims
wm = WhaleModel(readline(joinpath(@__DIR__, "../../example/example-2/tree.nw")))
ccd = read_ale(joinpath(@__DIR__, "../../example/example-2/ale"), wm)
```

Define objective and gradient (note that Optim looks for minima, so we use
-ℓ as objective)

```@example mle-sims
function objective(x)
    rates = ConstantRates((λ=x[1], μ=x[2], q=x[3:2], η=x[3]))
    -logpdf(wm(ConstantRates(rates)), ccd)
end
f = (x) -> objective(vcat(promote(x..., 0.8)...))  # we fix η = 0.8
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)
```

And now optimize

```@example mle-sims
init = rand(2)
lower = [-Inf, -Inf]   # lower bounds λ, μ and q
upper = [Inf, Inf]    # upper bounds λ, μ and q
results = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results.minimum, results.minimizer

@info exp.(results.minimizer)
```

Note that we can already retrieve the true values quite nicely for this rather
small data set (100 families)

We can obtain standard error estimates from the inverse of the observed Fisher
information matrix (i.e. the negative Hessian matrix at the MLE, recall that
the Fisher information matrix is an estimator of the asymptotic covariance
matrix).

```@example mle-sims
.√ ForwardDiff.hessian(f, results.minimizer)^(-1)
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

