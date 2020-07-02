module CompositeProblems

using LinearAlgebra
using StructuredProximalOperators
using Random
using Distributions

# Write your package code here.

abstract type CompositeProblem end



##
problem_dimension(pb::CompositeProblem) = error("problem_dimension not implemented for a problem $(typeof(pb)).")


## Smooth side
# 0th order
f(pb::CompositeProblem, x) = error("f not implemented for a problem $(typeof(pb)) and point $(typeof(x)).")

# 1st order
function ∇f!(pb::CompositeProblem, res, x)
    error("∇f! not implemented for a problem $(typeof(pb)), result $(typeof(res)) and point $(typeof(x)).")
end
function ∇f(pb::CompositeProblem, x)
    error("∇f not implemented for a problem $(typeof(pb)) and point $(typeof(x)).")
end

# 2nd order
function ∇²f_h!(pb::CompositeProblem, res, x, h)
    error("∇²f_h! not implemented for a problem $(typeof(pb)), result $(typeof(res)), point $(typeof(x)) and direction $(typeof(h)).")
end
function ∇²f_h(pb::CompositeProblem, x, h)
    error("∇²f_h not implemented for a problem $(typeof(pb)), point $(typeof(x)) and direction $(typeof(h)).")
end

# conditioning
get_gradlips(pb::CompositeProblem) = error("get_gradlips not implemented for a problem $(typeof(pb)).")
get_μ_cvx(pb::CompositeProblem) = error("get_μ_cvx not implemented for a problem $(typeof(pb)).")


## Nonsmooth side
g(pb::CompositeProblem, x) = (pb.regularizer)(x)

prox_αg!(pb::CompositeProblem, res, x, α) = prox_αg!(pb.regularizer, res, x, α)
prox_αg(pb::CompositeProblem, x, α) = prox_αg(pb.regularizer, x, α)

∇M_g!(pb::CompositeProblem, M, ∇M_g, x) = ∇M_g!(pb.regularizer, M, ∇M_g, x)
∇M_g(pb::CompositeProblem, M, x) = ∇M_g(pb.regularizer, M, x)

∇²M_g_ξ(pb::CompositeProblem, M, x, ξ) = ∇²M_g_ξ(pb.regularizer, M, x, ξ)
∇²M_g_ξ!(pb::CompositeProblem, M, res, x, ξ) = ∇²M_g_ξ!(pb.regularizer, M, res, x, ξ)


export f, ∇f!, ∇f, ∇²f_h!, ∇²f_h
export get_gradlips, get_μ_cvx
export g, prox_αg!, prox_αg, ∇M_g!, ∇M_g, ∇²M_g_ξ!, ∇²M_g_ξ
export problem_dimension


include("problems/Lasso.jl")
export get_lasso


end
