
# Maximum likelihood estimation

Here I'll show a quick example on how to perform Maximum likelihood parameter
estimation under a constant-rates duplication and loss model with a specific
WGD hypothesis (as in e.g. Rabier et al. 2014) using Whale.

```@example mle
using Whale
using Optim
using ForwardDiff
```

We'll use the example species tree in the Whale module (note, this is simply a
newick string)

```@example mle
Whale.extree
```

Get the model

```@example mle
wm = WhaleModel(Whale.extree)
```

We introduce a WGD at distance 0.25 from LCA node of ATHA and ATRI (last common
ancestor of Arabidopsis and Amborella)

```@example mle
node = Whale.lcanode(wm, ["ATHA", "ATRI"])
addwgd!(wm, wm[node], 0.25, rand())  # random initial retention rate
wm, wm[18].event
```

and another one just in Arabidopsis

```@example mle
node = Whale.lcanode(wm, ["ATHA"])
addwgd!(wm, wm[node], 0.3, rand())  # random initial retention rate
wm, wm[19].event
```

Get the data (we use the example data available in the git repository)

```@example mle
ccd = read_ale(joinpath(@__DIR__, "../../../example/example-1/ale"), wm)
```

Define objective and gradient (note that Optim looks for minima, so we use
-ℓ as objective)

```@example mle
function objective(x)
    rates = ConstantRates((λ=x[1], μ=x[2], q=x[3:4], η=x[5]))
    -logpdf(wm(ConstantRates(rates)), ccd)
end
f = (x) -> objective(vcat(promote(x..., 0.8)...))  # we fix η = 0.8
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)
```

!!! note
    We use both the `f` and `objective` function to conveniently allow fixing
    the η parameter without running into type conversion issues.

And now optimize

```@example mle
init = rand(4)
lower = zeros(4)        # lower bounds λ, μ and q
upper = [Inf, Inf, 1., 1.]  # upper bounds λ, μ and q
results = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results.minimum, results.minimizer
```

Clearly, with the retention rate (q) estimate vanishing to zero, there is no
evidence for the hypothetical WGD in the ancestor of flowering plants in this
tiny data set assuming a constant duplication and loss rates model (but note
the caveats associated with any such analysis). The retention rate estimate
for the Arabidopsis WGD event does however suggest that the data supports that
WGD hypothesis. To formally test this, one could redo the analysis without the
Arabidopsis WGD and perform a likelihood ratio test (LRT). However, inference
under the constant-rates model is very sensitive to model violations, so I
don't think such LRT-based inference is very insightful.

```@example mle
wm = WhaleModel(Whale.extree)
node = Whale.lcanode(wm, ["ATHA", "ATRI"])
addwgd!(wm, wm[node], 0.25, rand())  # random initial retention rate
ccd = read_ale(joinpath(@__DIR__, "../../../example/example-1/ale"), wm)

function objective(x)
    rates = ConstantRates((λ=x[1], μ=x[2], q=x[3:3], η=x[4]))
    -logpdf(wm(ConstantRates(rates)), ccd)
end
f = (x) -> objective(vcat(promote(x..., 0.8)...))  # we fix η = 0.8
g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)

init = rand(3)
lower = zeros(3)        # lower bounds λ, μ and q
upper = [Inf, Inf, 1.]  # upper bounds λ, μ and q
results2 = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results2.minimum, results2.minimizer
```

The LRT is

```@example mle
ℓ1, ℓ0 = -results.minimum, -results2.minimum
Λ = -2*(ℓ0 - ℓ1)
```

which would result in a rejection of th null hypothesis ate the α = 0.05
level.

!!! warning
    This is purely illustrative, and I do not recommend an LRT based approach,
    especially when using simple models of duplication and loss rates across
    the species tree. For more information refer to Zwaenepoel & Van de Peer
    (2019, MBE)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

