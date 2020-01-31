
# Maximum likelihood estimation

Below an example is shown on how to perform maximum likelihood estimation for
a constant rates model of duplication and loss with one WGD and a geometric
prior with p 0.80 (and thus mean 1.25) on the number of lineages at the root.
This will use forward difff for computing the gradients with automatic
differentiation and the LBFGS algorithm to find the optimum

```@example
using Whale
using Optim
using ForwardDiff
using DistributedArrays

# get the model
wm = WhaleModel(Whale.extree)
# introduce WGD at distance 0.25 from LCA node of ATHA and ATRI (last common
# ancestor of Arabidopsis and Amborella)
addwgd!(wm, Whale.lcanode(wm, ["ATHA", "ATRI"]), 0.25)

# get the data
ccd = CCDArray(read_ale("./example/example-ale", wm))

# define objective and gradient
# note that Optim looks for minima, so we use -ℓ as objective
f(x) = -logpdf(wm(ConstantRates(λ=x[1], μ=x[2], q=x[3:3],
    η=promote(0.80, x[1])[1])), ccd)

g = (x) -> ForwardDiff.gradient(f, x)
g!(G, x) = G .= g(x)

init = rand(3)
lower = zeros(3)
upper = [Inf, Inf, 1.]
results = optimize(f, g!, lower, upper, init, Fminbox(LBFGS()))
-results.minimum, results.minimizer
```
