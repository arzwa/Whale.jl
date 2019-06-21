struct MvBeta{T<:Real} <: ContinuousMultivariateDistribution
    n::Int64
    a::T
    b::T
    d::Beta

    MvBeta{T}(n, a::T, b::T) where T = new{T}(n, a, b, Beta{T}(T(a), T(b)))
end

MvBeta(n::Int64, a::Int, b::Int) = MvBeta{Float64}(n, float(a), float(b))
MvBeta(n::Int64, a::T, b::T) where {T<:Real} = MvBeta{T}(n, a, b)
MvBeta(n::Int64, a::Real, b::Real) = MvBeta(n, promote(a, b)...)

Base.rand(d::MvBeta) = rand(d.d, d.n)

Base.length(d::MvBeta) = d.n

function Base.convert(::Type{MvBeta{T}}, n::Int64, a, b) where T<:Real
    MvBeta{T}(n, a, b)
end

function Base.convert(::Type{MvBeta{T}}, d::MvBeta{S}) where {T<:Real,S<:Real}
    MvBeta{T}(d.n, d.a, d.b)
end

logpdf(d::MvBeta, v::AbstractArray{T,1}) where T<:Real = sum(log.(pdf.(d.d, v)))
