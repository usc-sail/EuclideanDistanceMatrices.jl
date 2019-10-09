function relativeerror(D::MaskedEDM{T}, D̂::EDM{T}) where T <: Real
    return norm(D.D - D̂.D)/norm(D.D) * 100
end