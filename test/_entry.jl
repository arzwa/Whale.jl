# It makes sense to have a kind of entry point for Whale, requiring  user to specify a model and some input files to run an analysis, because (1) potential users are likely not to have experience with julia, (2) analyses are not likely to be run interactively anyway.

# It seems a good idea to write the actual julia script that gets run to a file.

# Fire is an interesting package
function whaleturing(treefile, wgdconf, modelfile, datafile)
    tree = readnw(readline(tree))
    # add wgds from wgdconf
    # model = WhaleModel
    # include(modelfile)  # Turing model and some settings
end
