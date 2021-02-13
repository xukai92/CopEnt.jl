module CopEnt

using DocStringExtensions
using SpecialFunctions: logabsgamma, digamma
using Distances: pairwise, Euclidean, Chebyshev
using StatsBase: tiedrank

abstract type AbstractCDF end
struct NonParametricCDF <: AbstractCDF end

univraite_cdf(x::AbstractVecOrMat) = 
    univraite_cdf(NonParametricCDF(), x)

"""
$(SIGNATURES)

Emperical estimation of univraite cumulative distribution function values.
"""
univraite_cdf(::NonParametricCDF, x::AbstractVector) = tiedrank(x) / length(x)
function univraite_cdf(::NonParametricCDF,x::AbstractMatrix)
    d, n = size(x)
    c = similar(x)
    for i in 1:d
        c[i,:] = tiedrank(x[i,:]) / n
    end
    return c
end

function _entropy_knn_info(x::AbstractVector, dist)
    pwd = pairwise(dist, x)
    return 1, length(x), pwd
end
function _entropy_knn_info(x::AbstractMatrix, dist)
    d, n = size(x)
    pwd = pairwise(dist, x, dims=2)
    return d, n, pwd
end

_entropy_knn_constant(::Euclidean, d) = 
    d * log(π) / 2 - d * log(2) - first(logabsgamma(1 + d / 2))
_entropy_knn_constant(::Chebyshev, d) = 0

"""
$(SIGNATURES)

Estimate the entropy using the Kraskov method [1]

# References

1. Alexander Kraskov, Harald Stögbauer and Peter Grassberger. "Estimating mutual information." Physical review, 2004.
"""
function entropy_knn(
    x::AbstractVecOrMat; k=3, dist=Euclidean()
)
    d, n, pwd = _entropy_knn_info(x, dist)
    logd = map(1:n) do i
        log(2 * sort(pwd[i,:])[k+1])
    end |> sum
    return digamma(n) - digamma(k) + _entropy_knn_constant(dist, d) + logd * d / n
end

"""
$(SIGNATURES)

Estimate the copula entropy by

1. Compute the emperical copula (see also [`univraite_cdf`](@ref));
2. Estimate the entropy using the Kraskov method (see [`entropy_knn`](@ref) for keyword parameters) [1].

The returned copula entropy is the negative mutual information [2].

# References

1. Alexander Kraskov, Harald Stögbauer and Peter Grassberger. "Estimating mutual information." Physical review, 2004.
2. Jian Ma and Zengqi Sun. "Mutual information is copula entropy." Tsinghua Science & Technology, 2011.
"""
function copula_entropy(x::AbstractVecOrMat; cdf=NonParametricCDF(), k=3, dist=Euclidean())
    c = univraite_cdf(cdf, x)
    return entropy_knn(c; k=k, dist=dist)
end

export univraite_cdf, entropy_knn, copula_entropy
export NonParametricCDF
export Euclidean, Chebyshev # reexport two distance structs

end # module
