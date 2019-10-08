using Test
using Distances
using TripletEmbeddings
using EuclideanDistanceMatrices

@testset "EuclideanDistanceMatrices.jl" begin
    # Write your own tests here.
end

@testset "EDMs.jl" begin
    # EuclideanDistanceMatrix
    d = 20
    n = 10
    @test_throws AssertionError EuclideanDistanceMatrix(d, n)
    @test_throws DomainError EuclideanDistanceMatrix(-1, n)
    @test_throws DomainError EuclideanDistanceMatrix(d, -1)

    d = 2
    n = 10
    D = pairwise(SqEuclidean(), rand(d, n), dims=2)
    @test_throws DomainError EuclideanDistanceMatrix(D + rand(n, n))

    D = Matrix{Float64}(undef, 1, 1)
    D[1,1] = 1
    @test_throws DomainError EuclideanDistanceMatrix(D)

    # NoisyEuclideanDistanceMatrix
    D = EuclideanDistanceMatrix(2,10)
    N = rand(2,2)
    X = Embedding(2,10)

    @test_throws DimensionMismatch NoisyEuclideanDistanceMatrix(D, N)
    @test_throws DimensionMismatch NoisyEuclideanDistanceMatrix(X, N)

    # NoisyMaskedEuclideanDistanceMatrix
    X = Embedding(2,10)
    N = rand(10,10)
    mask = rand(3,4)
    @test_throws DimensionMismatch NoisyMaskedEuclideanDistanceMatrix(X, N, mask)

    D = EuclideanDistanceMatrix(2,9)
    N = rand(10,10)
    mask = rand(3,4)
    @test_throws DimensionMismatch NoisyMaskedEuclideanDistanceMatrix(X, N, mask)

    # isedm
    X = rand(3,10)
    D = pairwise(SqEuclidean(), X, dims=2)
    @test isedm(D) == true
    @test isedm(X) == false

    X = rand(1,2)
    D = pairwise(SqEuclidean(), X, dims=2)
    @test isedm(D) == true

    D = pairwise(SqEuclidean(), X, dims=1)
    @test_throws DomainError isedm(D)
end