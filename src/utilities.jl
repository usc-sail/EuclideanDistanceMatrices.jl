"""
   randommask(D, p[ symmetric=false])

Generate a random Bernoulli mask of size(D).

# Examples

    julia> X = rand(2,3)
    2×3 Array{Float64,2}:
     0.78353   0.90907   0.130756
     0.102162  0.829066  0.246455

    julia> D = pairwise(SqEuclidean(), X, dims=2)
    3×3 Array{Float64,2}:
     0.0       0.54415   0.446934
     0.54415   0.0       0.945209
     0.446934  0.945209  0.0

    julia> mask = randommask(D, 0.5)
    3×3 Symmetric{Integer,Array{Integer,2}}:
         0   true  false
      true      0  false
     false  false      0

"""
function randommask(D::AbstractMatrix{T}, p::T; symmetric=true) where T <: Real
    LinearAlgebra.checksquare(D)
    @assert 0 < p ≤ 1 "p must be in the interval (0,1]."

    if symmetric
        # Generates a random lower triangular matrix and then symmetrizes it
        return Symmetric([i[1] > i[2] ? rand(Bernoulli(p)) : 0 for i in CartesianIndices(D)], :L)
    else
        # Generates random entries for all entries
        return [rand(Bernoulli(p)) for i in CartesianIndices(D)]
    end
end

function relativeerror(D::AbstractEuclideanDistanceMatrix{T}, D̂::AbstractEuclideanDistanceMatrix{T}) where T <: Real
    return norm(D - D̂)/norm(D) * 100
end