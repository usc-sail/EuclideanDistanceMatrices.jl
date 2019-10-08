"""
    Triplets(D::EuclideanDistanceMatrix{T}) where T

Compute the triplets from D.
"""
function TripletEmbeddings.Triplets(D::EuclideanDistanceMatrix{T}) where T <: Real
  
    n = size(D,1)
    triplets = Vector{Tuple{Int,Int,Int}}(undef, n*binomial(n-1, 2))
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k
            if D[i,j] < D[i,k]
                counter +=1
                @inbounds triplets[counter] = (i, j, k)
            elseif D[i,j] > D[i,k]
                counter += 1
                @inbounds triplets[counter] = (i, k, j)
            end
        end
    end

    return Triplets(triplets[1:counter])
end

function TripletEmbeddings.Triplets(D::MaskedEuclideanDistanceMatrix{T}) where T <: Real
    
    n = size(D,1)
    Dm = masked(D)
    triplets = Vector{Tuple{Int,Int,Int}}(undef, n*binomial(n-1, 2))
    counter = 0   

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k
            if Dm[i,j] != 0 && Dm[i,k] != 0
                if Dm[i,j] < Dm[i,k]
                    counter +=1
                    @inbounds triplets[counter] = (i, j, k)
                elseif Dm[i,j] > Dm[i,k]
                    counter += 1
                    @inbounds triplets[counter] = (i, k, j)
                end
            end
        end
    end

    return Triplets(triplets[1:counter])
end