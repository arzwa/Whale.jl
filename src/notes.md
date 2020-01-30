some thinking.

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
