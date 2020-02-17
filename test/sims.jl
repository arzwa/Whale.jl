# We use Beluga to simulate
using Beluga, NewickTree

# And use the following 9 taxon species tree
tree  = "((vvi:0.11000000000,((ath:0.06820000000,cpa:0.06820000000):0.0378,(mtr:0.1000000000,ptr:0.1000000000):0.006000000):0.00400000000):0.00702839392,((bvu:0.04330000000,cqu:0.04330000000):0.06820651982,(ugi:0.08350000000,sly:0.08350000000):0.02800651982):0.00552187411);"

# This is a set of branch-wise rates, not extremely variable
rates = [1.1 0.9 0.6 1.0 1.0 0.7 0.7 1.3 0.7 0.8 0.7 0.9 0.8 0.9 1.0 0.8 0.7;
         1.2 0.9 0.6 0.9 1.1 0.9 0.5 1.5 0.6 0.8 0.7 0.9 0.8 0.9 1.0 0.9 0.6]

# Constant rates
rates = ones(2, 17)

# The DLWGD model
model, _ = DLWGD(tree)

# Simulate trees (and profiles, but not really of interest here)
profile, trees = rand(model, 100)

# Get ale files
aledir = "/home/arzwa/dev/Whale.jl/example/example-2/ale"
mkpath(aledir)
for (i,tree) in enumerate(trees)
    Beluga.pruneloss!(tree)
    nw = tonw(tree, model)
    write(joinpath(aledir, "$i.nw"), nw)
    run(`ALEobserve $(joinpath(aledir, "$i.nw"))`)
    rm(joinpath(aledir, "$i.nw"))
end
