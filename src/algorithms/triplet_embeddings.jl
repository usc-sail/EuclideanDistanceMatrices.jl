function complete(D::MaskedEuclideanDistanceMatrix{T}, loss::TripletEmbeddings.AbstractLoss, d::Int; kwargs...) where T <: Real
    @assert d ≥ 1 "d must be ≥ 1"

    triplets = Triplets(D)
    X = Embedding(d, size(D, 1))
    _ = fit!(loss, triplets, X; kwargs...)

    D̂ = EuclideanDistanceMatrix(X)
    s = mean(filter(!isnan, D./D̂))

    return s * D̂
end