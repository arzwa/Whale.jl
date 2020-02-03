
I often have the situation where I want to compute a log-likelihood for iid
data in parallel. For instance if `X` is some `DArray` or `SharedArray` I might
use the following

```
logpdf(model, X) = mapreduce(x->logpdf(model, x), +, X)
```

In principle
