using Test
using Distances
using Distributions
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

@testset "masks.jl" begin
    n = 20
    deletions = 20
    mask = randommask(n, deletions)
    @test count(x -> x == 0, mask) == 2 * deletions

    @test_throws DomainError randommask(-1, 10)
    @test_throws DomainError randommask(-1, Bernoulli(0.5))
    @test_throws DomainError randommask(2, 0)

    n = 20
    fraction = 0.2
    mask = randommask(n, fraction)
    @test count(x -> x == 0, mask)/(prod(size(mask)) - n) == fraction

    # idxmask
    D = EuclideanDistanceMatrix(2, 10)
    @test_throws BoundsError idxmask(D, CartesianIndices((9:11, 9:11)))

    D = EuclideanDistanceMatrix(2, 10)
    mask = [1  1  1  1  1  1  1  1  1  1;
            1  1  0  0  1  1  1  1  1  1;
            1  0  1  0  1  1  1  1  1  1;
            1  0  0  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1]
    @test all(mask .== idxmask(D, CartesianIndices((2:4, 2:4))))

    mask = [1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  0  0  0  0;
            1  1  1  1  1  1  0  0  0  0;
            1  1  1  1  1  1  0  0  0  0;
            1  1  1  1  1  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1]
    @test all(mask .== idxmask(D, CartesianIndices((7:10, 3:5)), symmetric=true))
    @test all(mask .== idxmask(D, CartesianIndices((7:10, 3:5))))

    mask = [1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  1  1  1  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1;
            1  1  0  0  0  1  1  1  1  1]
    @test all(mask .== idxmask(D, CartesianIndices((7:10, 3:5)), symmetric=false))
end