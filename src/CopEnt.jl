module CopEnt

using DocStringExtensions
using SpecialFunctions: logabsgamma, digamma
using Distances: pairwise, Euclidean, Chebyshev
using StatsBase: tiedrank

"""
$(SIGNATURES)

Compute the emperical copula.
"""
empirical_copula(x::AbstractVector) = tiedrank(x) / length(x)
function empirical_copula(x::AbstractMatrix)
    d, n = size(x)
    c = similar(x)
    for i in 1:d
        c[i,:] = tiedrank(x[i,:]) / n
    end
    return c
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
function entropy_knn(x; k::Int=3, dist::Union{Euclidean, Chebyshev}=Euclidean())
    d, n = size(x)
    pwd = pairwise(dist, x, dims=2)
    logd = map(1:n) do i
        log(2 * sort(pwd[i,:])[k+1])
    end |> sum
    return digamma(n) - digamma(k) + _entropy_knn_constant(dist, d) + logd * d / n
end

"""
$(SIGNATURES)

Estimate the copula entropy by

1. Compute the emperical copula (see also [`empirical_copula`](@ref));
2. Estimate the entropy using the Kraskov method (see [`entropy_knn`](@ref) for keyword parameters) [1].

The returned copula entropy is the negative mutual information [2].

# References

1. Alexander Kraskov, Harald Stögbauer and Peter Grassberger. "Estimating mutual information." Physical review, 2004.
2. Jian Ma and Zengqi Sun. "Mutual information is copula entropy." Tsinghua Science & Technology, 2011.
"""
function copula_entropy(x; kwargs...)
    c = empirical_copula(x)
    return entropy_knn(c; kwargs...)
end

export empirical_copula, entropy_knn, copula_entropy
export Euclidean, Chebyshev # reexport two distance structs

end # module
