"""
    abstract type AbstractEuclideanDistanceMatrix{T} <: AbstractMatrix{T} end
"""
abstract type AbstractEuclideanDistanceMatrix{T} <: AbstractMatrix{T} end

"""
    EuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T} <: AbstractMatrix{T}

Euclidean Distance Matrix (EDM).

# Examples
Generate a random EDM:

    julia> D = EuclideanDistanceMatrix(2, 3)
    3×3 EuclideanDistanceMatrix{Float64}:
     0.0        0.0588086  0.361541
     0.0588086  0.0        0.635088
     0.361541   0.635088   0.0

---

Generate an EDM from a Matrix or TripletEmbeddings.Embedding:

    julia> X = Embedding(2,3)
    2×3 Embedding{Float64}:
     0.584104  0.372789  0.389338
     0.54629   0.547285  0.609064

    julia> D = EuclideanDistanceMatrix(X)
    3×3 EuclideanDistanceMatrix{Float64}:
     0.0        0.0446553   0.0418744
     0.0446553  0.0         0.00409058
     0.0418744  0.00409058  0.0
"""
mutable struct EuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T}
    D::Matrix{AbstractFloat} # Euclidean distances matrix

    function EuclideanDistanceMatrix(d::Int, n::Int)
        @assert d ≤ n "We need d ≤ n (dimension ≤ # of points)"

        X = rand(d, n)
        
        # This is equivalent to D = diag(G) * ones(1,n) - 2G + ones(n) * diag(G)',
        # where G is the Gram matrix of (centered) X.
        D = pairwise(SqEuclidean(), X, dims=2) 

        new{eltype(X)}(D)
    end

    function EuclideanDistanceMatrix(X::AbstractMatrix{T}) where T       
        # This is equivalent to D = diag(G) * ones(1,n) - 2G + ones(n) * diag(G)',
        # where G is the Gram matrix of (centered) X.
        D = pairwise(SqEuclidean(), X, dims=2)
        new{T}(D)
    end
end

mutable struct MaskedEuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T}
    D::Matrix{AbstractFloat} # Euclidean distances matrix
    mask::Matrix{Int}

    function MaskedEuclideanDistanceMatrix(
        X::AbstractMatrix{T},
        p::T;
        symmetric=false) where T <: Real

        @assert 0 < p ≤ 1 "p must be in the interval (0,1]"
        D = EuclideanDistanceMatrix(X)
        new{T}(D, randommask(D, p; symmetric=symmetric))
    end

    function MaskedEuclideanDistanceMatrix(
        D::EuclideanDistanceMatrix{T},
        p::T) where T <: Real

        @assert 0 < p ≤ 1 "p must be in the interval (0,1]"
        new{T}(D.D, rand(Bernoulli(p), size(D)))
    end

    function MaskedEuclideanDistanceMatrix(
        D::EuclideanDistanceMatrix{T},
        distribution::S) where {T <: Real, S <: Distribution{Univariate,Discrete}}
        
        new{T}(D.D, rand(distribution, size(D)))
    end

    function MaskedEuclideanDistanceMatrix(
        D::EuclideanDistanceMatrix{T},
        mask::Matrix{Int}) where T <: Real
        
        @assert size(D) == size(mask)
        new{T}(D.D, mask)
    end

end

mask(D::MaskedEuclideanDistanceMatrix) = D .* D.mask

Base.size(D::AbstractEuclideanDistanceMatrix) = size(D.D)
Base.getindex(D::AbstractEuclideanDistanceMatrix, inds...) = getindex(D.D, inds...)

function Base.:*(s::Number, D::AbstractEuclideanDistanceMatrix)
    D.D = s * D.D
    return D
end

LinearAlgebra.rank(D::AbstractEuclideanDistanceMatrix) = rank(D.D)

"""
    function isedm(D[, tol=sqrt(eps())])

Check whether a matrix is an EDM.
"""
function isedm(D::Matrix{T}; tol=sqrt(eps())) where T
    n = try
        LinearAlgebra.checksquare(D)
    catch DimensionMismatch
        return false
    end
    
    λs = eigvals(-J(n) * D * J(n) / 2)
    
    # isposdef doesn't work in this case for eigvals that are slightly negative (i.e. -1e-16).
    return all((isapprox.(λs, zeros(n), atol=tol)) .| (λs .≥ 0.0))
end

function J(n::Int)
    return Matrix{Float64}(I, n, n) - ones(n,n)/n
end