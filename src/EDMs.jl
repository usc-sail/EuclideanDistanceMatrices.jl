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
        d ≥ 1 || throw(DomainError("d < 1"))
        n ≥ 1 || throw(DomainError("n < 1"))
        @assert d ≤ n "We need d ≤ n (dimension ≤ # of points)"

        X = rand(d, n)
        
        # This is equivalent to D = diag(G) * ones(1,n) - 2G + ones(n) * diag(G)',
        # where G is the Gram matrix of (centered) X.
        D = pairwise(SqEuclidean(), X, dims=2) 

        new{Float64}(D)
    end

    function EuclideanDistanceMatrix(D::Matrix{T}) where T <: Real
        isedm(D) || throw(DomainError(D, "D is not an EDM."))
        new{T}(D)
    end

    function EuclideanDistanceMatrix(X::TripletEmbeddings.Embedding{T}) where T
        D = pairwise(SqEuclidean(), X, dims=2)
        new{T}(D)
    end

end

mutable struct NoisyEuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T}
    D::EuclideanDistanceMatrix{T} # Data
    N::Matrix{AbstractFloat}      # Noise

    function NoisyEuclideanDistanceMatrix(
        D::EuclideanDistanceMatrix{T},
        N::Matrix{T}) where T <: AbstractFloat

        size(N) == size(D) || throw(DimensionMismatch("D and N must be the same size."))

        new{T,S}(D, N)
    end

    function NoisyEuclideanDistanceMatrix(
        X::TripletEmbeddings.Embedding{T},
        N::Matrix{T}) where T <: AbstractFloat

        D = EuclideanDistanceMatrix(X)
        size(N) == size(D) || throw(DimensionMismatch("D and N must be the same size."))

        new{T}(D, N)
    end
end

mutable struct NoisyMaskedEuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T}
    D::EuclideanDistanceMatrix{T}
    N::Matrix{<:Real}
    mask::AbstractMatrix{<:Real}

    function NoisyMaskedEuclideanDistanceMatrix(
        X::TripletEmbeddings.Embedding{T},
        N::Matrix{S},
        mask::AbstractMatrix{P}) where {T <: Real, S <: AbstractFloat, P <: Real}
        
        D = EuclideanDistanceMatrix(X)

        size(mask) == size(D) || throw(DimensionMismatch("D and mask must be the same size."))
        size(mask) == size(N) || throw(DimensionMismatch("N and mask must be the same size."))
        
        new{T,S,P}(D, N, mask)
    end

    function NoisyMaskedEuclideanDistanceMatrix(
        D::EuclideanDistanceMatrix{T},
        N::Matrix{S},
        mask::AbstractMatrix{P}) where {T <: Real, S <: AbstractFloat, P <: Real}
        
        size(mask) == size(D) || throw(DimensionMismatch("D and mask must be the same size."))
        size(mask) == size(N) || throw(DimensionMismatch("N and mask must be the same size."))
        
        new{T,S,P}(D, N, mask)
    end

    function NoisyMaskedEuclideanDistanceMatrix(
        D::NoisyEuclideanDistanceMatrix{T},
        mask::AbstractMatrix{P}) where {T <: Real, S <: AbstractFloat, P <: Real}
        
        size(mask) == size(D) || throw(DimensionMismatch("D and mask must be the same size."))
        size(mask) == size(D.N) || throw(DimensionMismatch("N and mask must be the same size."))
        
        new{T,S,P}(D.D, D.N, mask)
    end
end

mutable struct MaskedEuclideanDistanceMatrix{T} <: AbstractEuclideanDistanceMatrix{T}
    D::Matrix{AbstractFloat} # Euclidean distances matrix
    mask::AbstractMatrix{<:Real}

    function MaskedEuclideanDistanceMatrix(
        X::TripletEmbeddings.Embedding{T},
        p::T;
        symmetric=false) where T <: Real

        @assert 0 < p ≤ 1 "p must be in the interval (0,1]"
        D = EuclideanDistanceMatrix(X)
        new{T}(D, randommask(D, p; symmetric=symmetric))
    end

    function MaskedEuclideanDistanceMatrix(
        X::TripletEmbeddings.Embedding{T},
        mask::AbstractMatrix{S}) where {T <: Real, S <: Real}
    
        D = pairwise(SqEuclidean(), X, dims=2)
        size(D) == size(mask) || throw(DimensionMismatch("D and mask myust be the same size"))
        new{T}(D, mask)
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
        mask::AbstractMatrix{S}) where {T <: Real, S <: Real}
        
        size(D) == size(mask) || throw(DimensionMismatch("D and mask myust be the same size"))
        new{T}(D.D, mask)
    end
end

masked(D::MaskedEuclideanDistanceMatrix) = D .* D.mask

Base.size(D::AbstractEuclideanDistanceMatrix) = size(D.D)
Base.getindex(D::AbstractEuclideanDistanceMatrix, inds...) = getindex(D.D, inds...)

function Base.show(io::IO, ::MIME"text/plain", D::MaskedEuclideanDistanceMatrix)
    show(io, "text/plain", D.D .* D.mask)
end

function Base.:*(s::Number, D::AbstractEuclideanDistanceMatrix)
    D.D = s * D.D
    return D
end

LinearAlgebra.rank(D::AbstractEuclideanDistanceMatrix) = rank(D.D)

"""
    isedm(D[, atol=sqrt(eps())])

Check whether a matrix is an EDM. atol is passed to isapprox's atol. 
"""
function isedm(D::Matrix{T}; atol=sqrt(eps())) where T <: Real
    n = try
        LinearAlgebra.checksquare(D)
    catch DimensionMismatch
        return false
    end

    n > 1 || throw(DomainError(D, "D must have dimensions n > 1"))
    
    λs = eigvals(-J(n) * D * J(n) / 2)
    
    # isposdef doesn't work in this case for eigvals that are slightly negative (i.e. -1e-16).
    return all((isapprox.(real(λs), zeros(n), atol=atol)) .| (real(λs) .≥ 0.0)) .& # Real part
           all(isapprox.(imag(λs), zeros(n), atol=atol))                           # Imag part
end

function J(n::Int)
    return Matrix{Float64}(I, n, n) - ones(n,n)/n
end