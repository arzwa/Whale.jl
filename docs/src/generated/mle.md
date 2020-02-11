
# Maximum likelihood estimation

```@example mle
using Whale
using Optim
using ForwardDiff
```

Get the model

```@example mle
wm = WhaleModel(Whale.extree)
```

Introduce WGD at distance 0.25 from LCA node of ATHA and ATRI (last common
ancestor of Arabidopsis and Amborella)

```@example mle
node = Whale.lcanode(wm, ["ATHA", "ATRI"])
addwgd!(wm, wm[node], 0.25, rand())
```

Get the data

```@example mle
ccd = read_ale(joinpath(@__DIR__, "../../../example/example-ale"), wm)
```

Define objective and gradient (note that Optim looks for minima, so we use
-ℓ as objective)

```@example mle
f(x) = -logpdf(wm(ConstantRates(λ=x[1], μ=x[2], q=x[3:3],
    η=promote(0.80, x[1])[1])), ccd)
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)
```

And optimize

```@example mle
init = rand(3)
lower = zeros(3)
upper = [Inf, Inf, 1.]
results = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results.minimum, results.minimizer
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

