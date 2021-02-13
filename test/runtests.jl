using Test, CopEnt
using CopEnt: univraite_cdf, entropy_knn
using LinearAlgebra: det

@testset "Entropy of univraite normal" begin
    for σ in [1, 2, 3]
        x = randn(1_000) * σ
        ent = 1 / 2 * log(2 * π * ℯ * σ^2)
        @test entropy_knn(x) ≈ ent atol=0.1
    end
end

@testset "Entropy of bivariate normal" begin
    for σ1 in [1, 2, 3], σ2 in [1, 2, 3]
        x = randn(2, 1_000) .* [σ1, σ2] 
        ent = 1 / 2 * log(det(2 * π * ℯ * [σ1^2 0; 0 σ2^2]))
        @test entropy_knn(x) ≈ ent atol=0.1
    end
end

@testset "Checking against pycopent" begin
    x = hcat(
        [-0.89395472, -1.48611224],
        [ 0.82243227, -1.5848612 ],
        [-1.26209653, -0.24610746],
        [-0.20907981,  1.51772724],
        [-0.59596898, -0.24640368],
        [-0.28688517,  0.6399274 ],
        [-1.32899876, -1.13641078],
        [-0.36206995, -0.42210696],
        [ 0.21879637,  1.21598259],
        [-0.48603486, -1.84245597],
    )

    upy = hcat(
        [0.3, 0.3],
        [1. , 0.2],
        [0.2, 0.7],
        [0.8, 1. ],
        [0.4, 0.6],
        [0.7, 0.8],
        [0.1, 0.4],
        [0.6, 0.5],
        [0.9, 0.9],
        [0.5, 0.1],
    )
    entpy_euclidean = 2.884534529955401
    entpy_chebyshev = 2.9414966444360235
    copentpy_euclidean = 0.7167856845750248
    copentpy_chebyshev = 0.6760921005096859

    @test univraite_cdf(x) == upy
    @test entropy_knn(x; dist=Euclidean()) ≈ entpy_euclidean
    @test entropy_knn(x; dist=Chebyshev()) ≈ entpy_chebyshev
    @test copula_entropy(x; dist=Euclidean()) ≈ copentpy_euclidean
    @test copula_entropy(x; dist=Chebyshev()) ≈ copentpy_chebyshev
end
