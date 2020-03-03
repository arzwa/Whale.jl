"""
    unpack(df::DataFrame)
    unpack( x::Vector)

Unpack a data frame so that all columns contain scalars. Can be repacked using
`pack`. Supports only columns that are either scalars, 1D or 2D arrays.
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
df2vec(df) = [(; [x=>y[x] for x in names(y)]...) for y in eachrow(df)]

"""
    pack(df::DataFrame)

Pack a data frame into a vector of named tuples. Column names should
be as those used by `unpack`.
"""
function pack(df::DataFrame)
    temp = template(df)
    [pack(x, temp) for x in eachrow(df)]
end

# Some ugly and inefficient code here.
function pack(x::DataFrameRow, temp::Dict)
    d = deepcopy(temp)
    for (k,(s,i,j)) in enumerate((parsename.(names(x))))
        if i > 0 && j > 0
            d[s][i, j] = x[k]
        elseif i > 0
            d[s][i] = x[k]
        else
            d[s] = x[k]
        end
    end
    (; d...)
end

function template(df::DataFrame)
    d = Dict()
    for (s, i, j) in parsename.(names(df))
        d[s] = !haskey(d,s) ? (i,j) : (max(d[s][1], i), max(d[s][2], j))
    end
    Dict(k=>v[1] == 0 && v[2] == 0 ? 0. :
            v[1]  > 0 && v[2] == 0 ? zeros(v[1]) :
            zeros(v...) for (k, v) in d)
end

function parsename(s::Symbol)
    ss = split(string(s), "_")
    i = length(ss) >= 2 ? parse(Int, ss[2]) : 0
    j = length(ss) >= 3 ? parse(Int, ss[3]) : 0
    Symbol(ss[1]), i, j
end
