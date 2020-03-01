"""
    parsepost!(df::DataFrame)

Parse the dumped posterior.
"""
function parsepost!(df::DataFrame)
    f = (x) -> eval.(Meta.parse.(x))
    for (col, x) in eachcol(df, true)
        df[!,col] = try; f(x); catch; x; end
    end
end

"""
    unpack(df::DataFrame)

Unpack a data frame so that all columns contain scalars.
"""
function unpack(df::DataFrame)
    cols = []
    for (col, x) in eachcol(df, true)
        push!(cols, unpack.(x, col))
    end
    DataFrame(merge.(cols...))
end

unpack(x::Vector) = unpack(DataFrame(x))
unpack(x::T, sym::Symbol) where T<:Real = (; sym=>x)
unpack(x::AbstractMatrix, sym::Symbol) = (; [Symbol("$(sym)_$(i)_$(j)")=>x[i,j] for i=1:size(x)[1], j=1:size(x)[2]]...)
unpack(x::Vector, sym::Symbol) = (; [Symbol("$(sym)_$(i)")=>x[i] for i=1:length(x)]...)

# function pack(df::DataFrame)
#     varinfo = DataFrame(parsename.(names(df)))
#     template = []
#     gdf = groupby(varinfo, 1)
#     for (i, k) in enumerate(keys(gdf))
#
#     [pack(row) for row in eachrow(df)]
#
# function pack(x::DataFrameRow)
#     for n in names(x)
#

function parsename(s::Symbol)
    ss = split(string(s), "_")
    i = length(ss) >= 2 ? parse(Int, ss[2]) : -1
    j = length(ss) >= 3 ? parse(Int, ss[3]) : -1
    Symbol(ss[1]), i, j
end


df2vec(df) = [(; [x=>y[x] for x in names(y)]...) for y in eachrow(df)]
