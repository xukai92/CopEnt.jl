using Test, CopEnt

@testset "Validating against pycopent" begin
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

    cpy = hcat(
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

    @test empirical_copula(x) == cpy
    @test entropy_knn(x; dist=Euclidean()) ≈ entpy_euclidean
    @test entropy_knn(x; dist=Chebyshev()) ≈ entpy_chebyshev
    @test copula_entropy(x; dist=Euclidean()) ≈ copentpy_euclidean
    @test copula_entropy(x; dist=Chebyshev()) ≈ copentpy_chebyshev
end
