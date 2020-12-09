###############################################################################
## Adrian Lewis' OWOS problem
#    min_(x,y) = 2x² + y² + |x²-y|

struct ALfunction <: CompositeProblem
end

problem_dimension(pb::LogisticPb) = 2

## f
# 0th order
f(pb::LogisticPb, x)
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
