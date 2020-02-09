
# Reimplementation goals

A more flexible, hopefully performant implementation. What would we like.

1. we should get rid of the dicts
2. the model should be a tree data structure itself (but do check performance)
3. WGDs should be easy to consider randomly located
4. we should be able to introduce and remove WGDs efficiently, enabling rjMCMC
5. reimplement the CCD data structure properly

The trick with the CCD object, which I also used in Beluga, was to have a single
struct that is distributed, containing all the computed stuff that has to be
performant and is updated in MCMC like applictions. This essentially means
keeping a copy at all times of the computed PGM.

I still think this is quite good, I don't see another approach how we can
efficiently recompute these DP matrices and restore the upon rejection.

thinking about (4), we could only introduce WGDs at particular slice boundaries
(different from original Whale, where this affected the slices). That way we
should also be able to leave the DP matrices intact, without having to extend or
shrink the matrices (as in Beluga for instance), i.e. we don't use different row
or columns or whatever for the WGDs in the DP matrix.

To do this we only have to make sure the t field for the WGD nodes is a slice
boundary of the original branch. At the model level, the branch gets really
separated in two branches with independent slices, however the total number of
slices stays the same, and in the DP matrix we should offset computations for a
WGD node by the number of slices until the next speciation below the WGD node.

TO DO:
- implement GBM model (univariate and/or bivariate?)

# Performance notes

Currently, the model is copied entirely everytime parameters change, this is
wasteful, it would probably be better to have functions like:

```
logpdf(wm::WhaleModel{R}, ccd, θ)
```

where the RatesModel type defines how to get rates from a parameter vector θ,
θ being for instance a namedtuple from TransformVariables.


## Parallel gradients: sources of overhead

(1) model copying and setting of slices happens for each CCD
(2) usage of logpdf instead of logpdf! (Dual type issue)
possible performance hacks:
it would be more efficient to differentiate the sum for each worker

Now I did the latter, for each worker we now differentiate the sum, instead
of differentiating all CCDs and summing. In other words, the gradient closure
works on a vector now but is still in parallel, and the model is copied and
reset once per worker. It gave a slight speed-up.

# Summarizing reconciliations

This remains a tricky part. We can of course abstract the reconciliation and
compute consensus topologies, no problem, but what happens to the
reconciliations?

The topology can be separated from the reconciliation, so if we have a consensus
topology, we can get for each clade in the consensus topology its approximate
pmf for reconciliations as a separate table, e.g. for the MAP reconcilation:

```
clade maprec  mapevent      pp    cladep
γ1    e       :speciation   0.8   0.9
γ2    f       :sploss       0.6   0.8
γ3    f       :duplication  0.3   0.9
...
```
In the above, pp could be the posterior probability of the MAP reconciliation of
the `clade`, and `cladep` could be the posterior probability that the clade is in
the reconciled tree? This gives the most important info, but it would be nicer
if we could include information on the other reconciliations of each clade as
well instead of only showing the MAP event. Ideally would have a more
comprehensive summary from which such a table could be easily obtained...
