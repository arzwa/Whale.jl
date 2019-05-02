#= It should be possible to use Mamba as MCMC engine, since we are actually not
doing a very special MCMC algorithm like e.g. in tree inference (where the
tree structure requires a lot of non-standard stuff in the algorithms). In the
(WH)ALE case however, the MCMC itself is more amenable to implementation in
external MCMC engines.

What we need would be the following:

- A struct analogous to a UnivariateDistribution but where the variate is a
CCD. This 'distribution' object should have a `logpdf` method which returns
the likelihood of a CCD given S, λ, μ, q, ... (which are the 'parameters' of
the distribution)

- Similar structs for the rate priors (GBM, IID).

That will be some work to implement but it's a nice opportunity to have alook
at the core code again and we might get a more efficient/reliable/maintainable
result in the end. =#
