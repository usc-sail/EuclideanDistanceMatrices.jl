function relativeerror(D::MaskedEuclideanDistanceMatrix{T}, D̂::EuclideanDistanceMatrix{T}) where T <: Real
    return norm(D.D - D̂.D)/norm(D.D) * 100
end