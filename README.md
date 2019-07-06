[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://arzwa.github.io/Whale.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://azwa.github.io/PACKAGE_NAME.jl/dev)

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

**Note**: The docs web page is not yet functional, but you can find the documentation in fairly readable format [here](https://github.com/arzwa/Whale.jl/tree/master/docs/src), in the files `index.md` and `manual.md`.

- This library implements the duplication, loss and whole genome duplication (DL + WGD) model for performing joint gene tree - reconciliation inference using amalgamated likelihood estimation (ALE). By using amalgamation, uncertainty in the gene tree topology is taken into account during reconciliation.

- This method, called Whale, can be used to assess WGD hypotheses using gene family phylogenetic trees. It can also be used to estimate branch-specific duplication and loss rates for a species tree under different models of rate evolution.

- To install `Whale`, you will need a julia installation (v1.x). You should fire up the julia REPL (typically by typing `julia`), once in the julia REPL, you should type `]` to enter the package manager CLI and run the following commands:

```
(v1.1) pkg> add https://github.com/arzwa/PhyloTrees.jl
(v1.1) pkg> add https://github.com/arzwa/ConsensusTrees.jl
(v1.1) pkg> add https://github.com/arzwa/Whale.jl
```

You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).
