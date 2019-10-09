"""
   randommask(n, Bernoulli(p)[ symmetric=true])

Generate a random Bernoulli mask of n × n.
"""
function randommask(n::Int, distribution::Bernoulli{T}; symmetric=true) where T
    n ≥ 2 || throw(DomainError(n, "n < 2"))

    if symmetric
        # Generates a random lower triangular matrix and then symmetrizes it
        return Symmetric([i[1] > i[2] ? convert(Int, rand(distribution)) : 1 for i in CartesianIndices((1:n,1:n))], :L)
    else
        # Generates random entries for all entries
        return [i[1] != i[2] ? convert(Int, rand(distribution)) : 1 for i in CartesianIndices((1:n,1:n))]
    end
end

"""
   randommask(D, Bernoulli(p)[ symmetric=true])

Generate a random Bernoulli mask of size(D, 1).
"""
function randommask(D::EDM, distribution::Bernoulli{T}; symmetric=true) where T
    return randommask(size(D, 1), distribution, symmetric=symmetric)
end

"""
    randommask(n::Int, number_of_deletions::Int)

Randomly and uniformly create a mask with 2 × "number_of_deletions" number of deletions. The
2 × multiplier is due to the deletions done only on the entries below the diagonal,
and then symmetrized.
"""
function randommask(n::Int, number_of_deletions::Int)
    n ≥ 2 || throw(DomainError(n, "n < 2"))
    number_of_deletions > 0 || throw(DomainError(number_of_deletions, "deletions < 1"))

    # We create a lower diagonal matrix from which we can sample CartesianIndices
    # uniformly at random without replacement using sample from StatsBase
    mask = UnitLowerTriangular(ones(Int, n, n)) - Matrix(I, n, n)
    indices = sample(findall(x -> x == 1, mask), number_of_deletions, replace=false)

    # We set the values of indices to zero, and symmetrize
    mask[indices] .= 0
    mask = mask + Matrix(I, n, n)

    return Symmetric(mask, :L)
end

"""
    randommask(D::EDM, number_of_deletions::Int)

Randomly and uniformly create a mask with 2 × "number_of_deletions" number of deletions for D.
The 2 × multiplier is due to the deletions done only on the entries below the diagonal,
and then symmetrized.
"""
function randommask(D::EDM, number_of_deletions::Int)
    return randommask(size(D, 1), number_of_deletions)
end

"""
    randommask(n::Int, fraction_of_deletions::Real)

Randomly and uniformly create a mask with n * (n - 1) * fraction_of_deletions number of deleted
elements.
"""
function randommask(n::Int, fraction_of_deletions::Float64)
    n ≥ 2 || throw(DomainError(n, "n < 2"))
    1 ≥ fraction_of_deletions > 0 || throw(DomainError(fraction_of_deletions, "We need 0 < fraction ≤ 1 of entries."))

    deletions = round(Int, n * (n-1) / 2 * fraction_of_deletions)
    return randommask(n, deletions)
end

"""
    randommask(D::EDM, fraction_of_deletions::Real)

Randomly and uniformly create a mask with n * (n - 1) * fraction_of_deletions number of deleted
elements.
"""
function randommask(D::EDM, fraction_of_deletions::Real)
    return randommask(size(D, 1), fraction_of_deletions)
end

"""
    idxmask(D::EDM, indices::CartesianIndices)

Generate a mask from CartesianIndices.

The indices along the diagonal are ignored.
"""
function idxmask(D::EDM, indices::CartesianIndices; symmetric::Bool=true)
    mask = ones(Int, size(D))
    
    # We set values to zero for all non-diagonal indices
    mask[[idx for idx in indices if idx[1] != idx[2]]] .= 0
    
    if symmetric && issymmetric(mask)
        return mask
    elseif symmetric && !issymmetric(mask)
        # If there are more zeros in the lower triangle
        if count(x -> x == 0, LowerTriangular(mask)) > count(x -> x == 0, UpperTriangular(mask))
            return Symmetric(mask, :L)
        else # It's not symmetric, so the zeros must be above the diagonal
            return Symmetric(mask, :U)
        end
    else
        return mask
    end

end