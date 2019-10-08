"""
   randommask(D, p[ symmetric=true])

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
function randommask(n::Int, distribution::Bernoulli{T}; symmetric=true) where T
    if symmetric
        # Generates a random lower triangular matrix and then symmetrizes it
        return Symmetric([i[1] > i[2] ? rand(distribution) : 0 for i in CartesianIndices(zeros(n,n))], :L)
    else
        # Generates random entries for all entries
        return [rand(distribution) for i in CartesianIndices(D)]
    end
end

"""
    randommask(n::Int, deletions::Int)

Randomly and uniformly create a mask with 2 × "deletions" number of deletions. The
2 × multiplier is due to the deletions done only on the entries below the diagonal,
and then symmetrized.
"""
function randommask(n::Int, deletions::Int)
    # We create a lower diagonal matrix from which we can sample CartesianIndices
    # uniformly at random without replacement
    mask = UnitLowerTriangular(ones(Int, n, n)) - Matrix(I, n, n)
    indices = sample(findall(x -> x == 1, mask), deletions, replace=false)

    mask[indices] .= 0
    mask = mask + Matrix(I, n, n)

    return Symmetric(mask, :L)
end

function relativeerror(D::MaskedEuclideanDistanceMatrix{T}, D̂::EuclideanDistanceMatrix{T}) where T <: Real
    return norm(D.D - D̂.D)/norm(D.D) * 100
end