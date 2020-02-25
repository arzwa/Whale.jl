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

unpack(x::T, sym::Symbol) where T<:Real = (; sym=>x)
unpack(x::Matrix, sym::Symbol) = (; [Symbol("$(sym)$(i)_$(j)")=>x[i,j] for i=1:size(x)[1], j=1:size(x)[2]]...)
unpack(x::Vector, sym::Symbol) = (; [Symbol("$(sym)$(i)")=>x[i] for i=1:length(x)]...)

df2vec(df) = [(; [x=>y[x] for x in names(y)]...) for y in eachrow(df)]
