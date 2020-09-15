###############################################################################
## Logistic Problem
#    min_x  1/m * ∑_{i=1}^m log(1 + exp(-y_i dot(A_i, x))) + g(reg, x)
#
#    x : R^n
#    y : R^m (observations)
#    A : mxn matrix of samples

mutable struct LogisticPb{Tr, Tm} <: CompositeProblem
    A::Matrix{Float64}
    y::Vector{Float64}
    regularizer::Tr
    n::Int64
    x0::Vector{Float64}
    M_x0::Tm
end

problem_dimension(pb::LogisticPb) = pb.n

## f
# 0th order
function f(pb::LogisticPb, x)
    m = size(pb.A, 1)

    Ax = pb.A * x
    fval = 0.0
    @inbounds @simd for i in 1:m
        fval += log(1 + exp(-pb.y[i] * Ax[i]))
    end

    return fval / m
end

# 1st order
σ(x) = 1/(1+exp(-x))
function ∇f!(pb::LogisticPb, res, x)
    m = size(pb.A, 1)

    σyAx = σ.(-pb.y .* (pb.A * x))
    res .= transpose(pb.A) * (σyAx .* -pb.y)
    res ./= m

    return res
end

# 2nd order
∇σ(x) = σ(x) * σ(-x)
function ∇²f_h!(pb::LogisticPb, res, x, h)
    m = size(pb.A, 1)

    yAx = -pb.y .* (pb.A * x)
    Ah = pb.A * h

    res .= transpose(pb.A) * (Ah .* ∇σ.(yAx))
    res ./= m

    return res
end

# conditioning
function get_gradlips(pb::LogisticPb)
    m = size(pb.A, 1)
    return opnorm(pb.A)^2 / m
end
# get_μ_cvx(pb::LogisticPb) = (svdvals(pb.A)[end])^2
