using Whale
using PhyloTrees

# simple readdict function
function readdict(fname::String; sep=",")
    lines = open(fname) do file ; readlines(file) ; end
    return Dict(string(split(line, sep)[1]) => string(split(line, sep)[2]) for line in lines)
end

sptree = read_sp_tree("/home/arzwa/whale/data/morris.trees/1701170.nw")
for (n, s) in sptree.species; sptree.species[n] = join(split(s, "_"), " "); end
for (n, s) in sptree.species; sptree.species[n] = " "; end
Whale.drawtree(sptree, height=1200, width=800, linewidth=2, fname="/home/arzwa/tmp/largetree.svg")

# 5 taxa -------------------------------------------------------------------------------------------
sptree = read_sp_tree("example/morris-5taxa.nw")
dffile = "/home/arzwa/whale/data/ortho5-whale3/gbm2-1.csv"
leaves = readdict("example/species.txt")
conf = read_whaleconf("/home/arzwa/whale/data/ortho5-whale2/gbm1.conf")
q, ids = mark_wgds!(sptree, conf["wgd"])
for (n, s) in sptree.species; sptree.species[n] = leaves[s]; end

# get mean rates (should it be mean?)
sumry = diagnostics(dffile, burnin=1000)
λ = sumry[[startswith(string(var), "l") for var in sumry[:variable]], :mean][1:end-1]
μ = sumry[[startswith(string(var), "m") for var in sumry[:variable]], :mean][1:end]
q = sumry[[startswith(string(var), "q") for var in sumry[:variable]], :mean][1:end]

Whale.coltree(sptree, λ, q=q)
Whale.drawtree(sptree.tree, nodelabels=true)
λ[6] = 0.25
μ[6] = 0.50

# 6 taxa -------------------------------------------------------------------------------------------
leaves = readdict("example/species.txt")
sptree = read_sp_tree("example/morris-6taxa.nw")
dffile = "/home/arzwa/whale/data/ortho4-whale3/iid1-1.csv"
conf = read_whaleconf("example/whale6.conf")
q, ids = mark_wgds!(sptree, conf["wgd"])
for (n, s) in sptree.species; sptree.species[n] = leaves[s]; end

# get mean rates (should it be mean?)
sumry = diagnostics(dffile, burnin=1000)
λ = sumry[[startswith(string(var), "l") for var in sumry[:variable]], :mean][1:end-1]
μ = sumry[[startswith(string(var), "m") for var in sumry[:variable]], :mean][1:end]
q = sumry[[startswith(string(var), "q") for var in sumry[:variable]], :mean][1:end]

Whale.drawtree(sptree)
Whale.drawtree(sptree.tree, nodelabels=true, linewidth=2)
Whale.coltree(sptree, λ, q=q, fname="/home/arzwa/tmp/la.svg")

# 9 taxa -------------------------------------------------------------------------------------------
leaves = readdict("example/species.txt")
sptree = read_sp_tree("example/morris-9taxa.nw")
conf = read_whaleconf("../data/ortho3-whale3/gbm1.conf")
q, ids = mark_wgds!(sptree, conf["wgd"])
for (n, s) in sptree.species; sptree.species[n] = leaves[s]; end

using CSV
dffile = "/home/arzwa/whale/data/ortho3-whale3/gbm1-1.csv"
df = CSV.read(dffile)
sumry_ = diagnostics(df[1001:end,:])

λ_ = sumry_[[startswith(string(var), "l") for var in sumry_[:variable]], :mean][1:end-1]
μ_ = sumry_[[startswith(string(var), "m") for var in sumry_[:variable]], :mean][1:end]
q_ = sumry_[[startswith(string(var), "q") for var in sumry_[:variable]], :mean][1:end]

# get mean rates (should it be mean?)
dffile = "/home/arzwa/whale/data/ortho3-whale3/iid1-1.csv"
sumry = diagnostics(dffile, burnin=1000)

λ = sumry[[startswith(string(var), "l") for var in sumry[:variable]], :mean][1:end-1]
μ = sumry[[startswith(string(var), "m") for var in sumry[:variable]], :mean][1:end]
q = sumry[[startswith(string(var), "q") for var in sumry[:variable]], :mean][1:end]

Whale.coltree(sptree, λ, q=q, fname="/home/arzwa/tmp/ortho3-whale3-gbm2-l.svg")
Whale.drawtree(sptree.tree, nodelabels=true, linewidth=2)
Whale.drawtree(sptree)

# 12 taxa ------------------------------------------------------------------------------------------
leaves = readdict("example/species.txt")
sptree = read_sp_tree("../data/ortho6/morris-12taxa.nw")
dffile = "../data/ortho6/gbm1-500ale.csv"
conf = read_whaleconf("../data/ortho6/gbm1.conf")
q, ids = mark_wgds!(sptree, conf["wgd"])
for (n, s) in sptree.species; sptree.species[n] = leaves[s]; end

# get mean rates (should it be mean?)
sumry = diagnostics(dffile, burnin=1000)
λ = sumry[[startswith(string(var), "l") for var in sumry[:variable]], :mean][1:end-1]
μ = sumry[[startswith(string(var), "m") for var in sumry[:variable]], :mean][1:end]
q = sumry[[startswith(string(var), "q") for var in sumry[:variable]], :mean][1:end]

Whale.drawtree(sptree)
Whale.drawtree(sptree.tree, nodelabels=true, linewidth=2)
Whale.coltree(sptree, μ, q=q, fname="/home/arzwa/tmp/mu.svg")


# 12 taxa ------------------------------------------------------------------------------------------
leaves = readdict("example/species.txt")
sptree = read_sp_tree("../data/ortho6/morris-12taxa.nw")
dffile = "../data/ortho6/gbm1-500ale.csv"
conf = read_whaleconf("../data/ortho6/gbm1.conf")
q, ids = mark_wgds!(sptree, conf["wgd"])
for (n, s) in sptree.species; sptree.species[n] = leaves[s]; end

# get mean rates (should it be mean?)
sumry = diagnostics(dffile, burnin=1000)
λ = sumry[[startswith(string(var), "l") for var in sumry[:variable]], :mean][1:end-1]
μ = sumry[[startswith(string(var), "m") for var in sumry[:variable]], :mean][1:end]
q = sumry[[startswith(string(var), "q") for var in sumry[:variable]], :mean][1:end]

Whale.drawtree(sptree)
Whale.drawtree(sptree.tree, nodelabels=true, linewidth=2)
Whale.coltree(sptree, μ, q=q, fname="/home/arzwa/tmp/mu.svg")
