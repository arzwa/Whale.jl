[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev/index.html)

**Note:** If you're interested in using the methods implemented in this library for research purposes, please consider contacting me, as there are some noteworthy developments that have not yet been integrated with the master branch and are not (well-)documented yet. For the methods as used in Zwaenepoel & Van de Peer (2019), consider the [v0.3 release](https://github.com/arzwa/Whale.jl/releases/tag/v0.3).

# Whale: whole genome duplication inference by amalgamated likelihood estimation

```

                             .-------------'```'----....,,__                        _,
                            |                               `'`'`'`'-.,.__        .'(
                            |                                             `'--._.'   )
                            |                                                   `'-.<
                            \               .-'`'-.                            -.    `\
                             \               -.o_.     _                     _,-'`\    |
                              ``````''--.._.-=-._    .'  \            _,,--'`      `-._(
                                (^^^^^^^^`___    '-. |    \  __,,..--'                 `
                                 `````````   `'--..___\    |`
                                                       `-.,'
```

- This library implements the duplication, loss and whole genome duplication (DL + WGD) model for performing joint gene tree - reconciliation inference using amalgamated likelihood estimation (ALE). By using amalgamation, uncertainty in the gene tree topology is taken into account during reconciliation.

- This method, called Whale, can be used to assess WGD hypotheses using gene family phylogenetic trees. It can also be used to estimate branch-specific duplication and loss rates for a species tree under different models of rate evolution.

- To install `Whale`, you will need a julia installation (v1.x). You should fire up the julia REPL (typically by typing `julia`), once in the julia REPL, you should type `]` to enter the package manager CLI and run the following commands:

```
(v1.1) pkg> add https://github.com/arzwa/PhyloTrees.jl
(v1.1) pkg> add https://github.com/arzwa/ConsensusTrees.jl
(v1.1) pkg> add https://github.com/arzwa/Whale.jl
```

Please have a look at the [docs](https://arzwa.github.io/Whale.jl/dev/index.html) for usage instructions and documentation. You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

Note that the scripts in the `scripts` directory might be helpful to prepare data for Whale analyses.

If you use Whale, please cite:

>[Zwaenepoel, A. and Van de Peer, Y., 2019. Inference of Ancient Whole-Genome Duplications and the Evolution of Gene Duplication and Loss Rates. *Molecular biology and evolution*, 36(7), pp.1384-1404.](https://academic.oup.com/mbe/article-abstract/36/7/1384/5475503)
