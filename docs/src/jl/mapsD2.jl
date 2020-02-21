# # Posterior analytics (not by Aristotle) example and MAPS D2
using CSV, Plots, DataFrames

# We first define some functions to parse the dumped posterior file.
function parsepost(df)
    f = (x) -> eval.(Meta.parse.(x))
    cols = []
    for (col, x) in eachcol(df, true)
        y = try; f(x); catch; x; end
        push!(cols, unpack.(y, col))
    end
    DataFrame(merge.(cols...))
end

unpack(x::T, sym::Symbol) where T<:Real = (; sym=>x)

unpack(x::Matrix, sym::Symbol) = (; [Symbol("$(sym)$(i)_$(j)")=>x[i,j]
    for i=1:size(x)[1], j=1:size(x)[2]]...)

unpack(x::Vector, sym::Symbol) =
    (; [Symbol("$(sym)$(i)")=>x[i] for i=1:length(x)]...)

# ## Whale posterior for MAPS D2 from the 1KP study
# Load the file and parse it
post = CSV.read(joinpath(@__DIR__, "../../../example/example-3/hmc-D2.3184654.csv"))
df = parsepost(post);

# A function to easily collect trace plots
traces(df; kwargs...) = [plot(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]

# ... and let's show some trace plots using it:
ps = traces(df, grid=false, legend=false, xticks=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))

# The same for marginal distributions (we could define a plot recipe to combine
# these with the trace plots)
marginalhists(df; kwargs...) = [stephist(x, title=col; kwargs...)
    for (col, x) in eachcol(df, true)]

ps = marginalhists(df, grid=false, legend=false, yticks=false,
    color=:black, linewidth=0.2, title_loc=:left, titlefont=7)
plot(ps..., size=(700,600))

# For this example, which is a reanalysis of a gymnosperm data set from the 1KP
# paper (using ML trees, so no gene tree uncertainty is taken into account), we
# clearly see a consistent inflation of the loss rate relative to the duplication
# rate. This is rather suspicious, since, if we would take this at face value, it
# would entail persistent genome contraction, which is no known feature of
# gymnosperm evolution. It seems the birth-death process is wandering of in rather
# unlikely regions of parameter space.

# ## Constraining duplication and loss rates

# A way to constrain this without sacrificing the flexibility of separate duplication
# and loss rates is by using a prior on the expected degree of expansion/contraction
# on each branch.
