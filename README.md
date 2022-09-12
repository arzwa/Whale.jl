[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)
[![build](https://github.com/arzwa/Whale.jl/actions/workflows/workflow.yaml/badge.svg)](https://github.com/arzwa/Whale.jl/actions/workflows/workflow.yaml)

**Note:** Do not hesitate to contact me when things are unclear, I am very happy to help.   
 
# Whale: Bayesian gene tree reconciliation and whole-genome duplication inference by amalgamated likelihood estimation

```julia
#
#           .-------------'```'----....,,__                        _,
#          |                               `'`'`'`'-.,.__        .'(
#          |                                             `'--._.'   )
#          |                                                   `'-.<
#          \               .-'`'-.                            -.    `\
#           \               -.o_.     _                     _,-'`\    |
#            ``````''--.._.-=-._    .'  \            _,,--'`      `-._(
#              (^^^^^^^^`___    '-. |    \  __,,..--'                 `
#               `````````   `'--..___\    |`
#                                     `-.,'
```

Whale.jl is a julia library for **joint inference of gene tree topologies and
their reconciliations to a species tree**. Whale uses the **amalgamation**
method of Szollosi et al. (2014) to efficiently compute the marginal likelihood
of the gene family under a duplication-loss model of gene family evolution over
a distribution of tree topologies. Whale also implements a duplication-loss and
whole-genome duplication (DLWGD) model (Rabier et al. 2014, Zwaenepoel et al.
2019). The latter can be used for the inference of ancient whole-genome
duplications (WGDs) from gene trees while taking into account gene tree and
reconciliation uncertainty.

More specifically, this library can be used to

- Statistically assess hypothetical whole-genome duplication (WGD) events in a
  species phylogeny.
- Infer lineage-specific gene duplication and loss rates for a species
  phylogeny.
- Infer high-quality (reconciled) gene trees given a known species tree using
  Bayesian gene tree reconciliation [cf. Szöllősi *et
  al.*](https://academic.oup.com/sysbio/article/64/1/e42/1634124)
- Conduct Bayesian orthology inference.
- All of the above at once.

For more information and guidance, please consult the
[docs](https://arzwa.github.io/Whale.jl/dev/index.html). 
If you are not a julia coder, you might want to get some minimal familiarity
with the Julia REPL and its package manager when using Whale, see [the julia
docs](https://docs.julialang.org/en/v1/).

Note that the scripts in the `scripts` directory might be helpful to prepare
data for Whale analyses.

## Citation

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

