###############################################################################
## Adrian Lewis' OWOS problem
#    min_(x,y) = 2x² + y² + |x²-y|

struct quadmaxquadAL <: CompositeProblem
    regularizer::maxquadAL
end

quadmaxquadAL() = quadmaxquadAL(maxquadAL())

problem_dimension(::quadmaxquadAL) = 2

## f
# 0th order
function f(::quadmaxquadAL, xy)
    x, y = xy
    return 2*x^2 + y^2
end

# 1st order
function ∇f!(::quadmaxquadAL, res, xy)
    x, y = xy
    return res .= [4x, 2y]
end

# 2nd order
function ∇²f_h!(::quadmaxquadAL, res, x, h)
    res[1] = 4*h[1]
    res[2] = 2*h[2]
    return res
end

# conditioning
# get_gradlips(::quadmaxquadAL)
# get_μ_cvx(::quadmaxquadAL) = (svdvals(pb.A)[end])^2

function firstorder_optimality_tangnorm(pb::quadmaxquadAL, x, M::PlaneParabola, ∇f_x)
    return norm(project(M, x, ∇f_x)), -1
end

function firstorder_optimality_tangnorm(::quadmaxquadAL, x, M::Euclidean, ∇f_x)
    ∇g_x = x[1]^2-x[2] > 0 ? [2*x[1], -1] : [-2*x[1], 1]
    return norm(∇f_x + ∇g_x), -1
end
