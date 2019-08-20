using EuclideanDistanceMatrices
using Distances
using Test

@testset "EuclideanDistanceMatrices.jl" begin
    # Write your own tests here.
end

@testset "EDMs.jl" begin
    X = rand(3,10)
    D = pairwise(SqEuclidean(), X, dims=2)
    @test isedm(D) == true
    @test isedm(X) == false
end