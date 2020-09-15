###############################################################################
## Logistic Problem
#    min_x  1/m * ∑_{i=1}^m log(1 + exp(-y_i dot(A_i, x))) + g(reg, x)
#
#    x : R^n
#    y : R^m observations, *values -1 and 1*
#    A : mxn matrix of samples

mutable struct LogisticPb{Tr, Tm} <: CompositeProblem
    A::Matrix{Float64}
    y::Vector{Float64}
    regularizer::Tr
    n::Int64
    x0::Vector{Float64}
    M_x0::Tm
    function LogisticPb(
            A::Matrix{Float64},
            y::Vector{Float64},
            regularizer::Tr,
            n::Int64,
            x0::Vector{Float64},
            M_x0::Tm
        ) where {Tr, Tm}
        @assert Set(y) == Set([-1.0, 1.0]) "Logistic rhs vector shoudl take values -1.0, 1.0, here: $(Set(y))."
        return new{Tr, Tm}(A, y, regularizer, n, x0, M_x0)
    end
end

problem_dimension(pb::LogisticPb) = pb.n


"""
    logsig(t)

Compute the logarithm of sigmoid `-log(1+exp(-t))` with higher precision than plain
implementation.

Reference:
- F. Pedragosa's blog post http://fa.bianp.net/blog/2019/evaluate_logistic/
"""
@inline function logsig(t)
    if t < -33.3
        return t
    elseif t <= -18
        return t - exp(t)
    elseif t <= 37
        return -log1p(exp(-t))
    else
        return -exp(-t)
    end
end

## f
# 0th order
function f(pb::LogisticPb, x)
    m = size(pb.A, 1)

    Ax = pb.A * x
    fval = 0.0
    @inbounds @simd for i in 1:m
        fval -= logsig(pb.y[i] * Ax[i])
    end

    return fval / m
end

# 1st order
σ(x) = 1/(1+exp(-x))
function ∇f!(pb::LogisticPb, res, x)
    m = size(pb.A, 1)

    σyAx = pb.A * x
    σyAx .*= -pb.y
    σyAx .= σ.(σyAx)
    res .= transpose(pb.A) * (σyAx .* pb.y)
    res ./= -m

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
