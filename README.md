[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://arzwa.github.io/Whale.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://arzwa.github.io/Whale.jl/dev)

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

This library implements the duplication, loss and whole genome duplication (DL + WGD) model for performing joint gene tree - reconciliation inference using amalgamated likelihood estimation (ALE). By using amalgamation, uncertainty in the gene tree topology is taken into account during reconciliation inference.

This method, called Whale, can be used to assess WGD hypotheses using gene family phylogenetic trees. It can also be used to estimate branch-specific duplication and loss rates for a species tree under different models of rate evolution.

To install `Whale`, you will need a julia installation (v1.x). You should fire up the julia REPL (typically by typing `julia`), once in the julia REPL, you should type `]` and enter `add https://github.com/arzwa/Whale.jl`. It should look somewhat like below:

```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.0.0 (2018-08-08)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

(v1.0) pkg> add https://github.com/arzwa/Whale.jl
   Cloning git-repo `https://github.com/arzwa/Whale.jl`
  Updating git-repo `https://github.com/arzwa/Whale.jl`
  ...
```

You might want to get some minimal familiarity with the Julia REPL and its package manager when using Whale, see [the julia docs](https://docs.julialang.org/en/v1/).

For more usage instructions and documentation please refer to the docs.
