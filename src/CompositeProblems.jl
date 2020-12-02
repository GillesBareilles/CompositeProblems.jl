module CompositeProblems

using LinearAlgebra
import StructuredProximalOperators:
    Manifold, g, prox_αg!, prox_αg, ∇M_g!, ∇M_g, ∇²M_g_ξ!, ∇²M_g_ξ
using StructuredProximalOperators
using Random
using Distributions
using DelimitedFiles
using JLD

using JuMP, MosekTools

abstract type CompositeProblem end

const instances_dir = joinpath(dirname(pathof(CompositeProblems)), "..", "instances")

##
function problem_dimension(pb::CompositeProblem)
    return error("problem_dimension not implemented for a problem $(typeof(pb)).")
end

function F(pb::CompositeProblem, x)
    return f(pb, x) + g(pb, x)
end


## Smooth side
# 0th order
function f(pb::CompositeProblem, x)
    return error("f not implemented for a problem $(typeof(pb)) and point $(typeof(x)).")
end

# 1st order
function ∇f!(pb::CompositeProblem, res, x)
    return error("∇f! not implemented for a problem $(typeof(pb)), result $(typeof(res)) and point $(typeof(x)).")
end
function ∇f(pb::CompositeProblem, x)
    res = zeros(size(x))
    ∇f!(pb, res, x)
    return res
end

# 2nd order
function ∇²f_h!(pb::CompositeProblem, res, x, h)
    return error("∇²f_h! not implemented for a problem $(typeof(pb)), result $(typeof(res)), point $(typeof(x)) and direction $(typeof(h)).")
end
function ∇²f_h(pb::CompositeProblem, x, h)
    res = zeros(size(h))
    ∇²f_h!(pb, res, x, h)
    return res
end

# conditioning
function get_gradlips(pb::CompositeProblem)
    return error("get_gradlips not implemented for a problem $(typeof(pb)).")
end
function get_μ_cvx(pb::CompositeProblem)
    return error("get_μ_cvx not implemented for a problem $(typeof(pb)).")
end


## Nonsmooth side
g(pb::CompositeProblem, x) = g(pb.regularizer, x)

prox_αg!(pb::CompositeProblem, res, x, α) = prox_αg!(pb.regularizer, res, x, α)
prox_αg(pb::CompositeProblem, x, α) = prox_αg(pb.regularizer, x, α)

∇M_g!(pb::CompositeProblem, M, ∇M_g, x) = ∇M_g!(pb.regularizer, M, ∇M_g, x)
∇M_g(pb::CompositeProblem, M, x) = ∇M_g(pb.regularizer, M, x)

∇²M_g_ξ(pb::CompositeProblem, M, x, ξ) = ∇²M_g_ξ(pb.regularizer, M, x, ξ)
∇²M_g_ξ!(pb::CompositeProblem, M, res, x, ξ) = ∇²M_g_ξ!(pb.regularizer, M, res, x, ξ)


export CompositeProblem
export F
export f, ∇f!, ∇f, ∇²f_h!, ∇²f_h
export get_gradlips, get_μ_cvx
export g, prox_αg!, prox_αg, ∇M_g!, ∇M_g, ∇²M_g_ξ!, ∇²M_g_ξ
export problem_dimension


include("problems/Leastsquares.jl")
include("problems/Logistic.jl")
include("problems/Logistic_instances.jl")
include("problems/Tracenorm.jl")

include("utils_instances.jl")
include("optimality_checkers.jl")

export LeastsquaresPb
export get_random_qualifiedleastsquares, get_random_qualifiedlasso, get_lasso_MLE

export LogisticPb
export get_logit_ionosphere, get_logit_gisette, get_random_logit, get_logit_MLE

export TracenormPb
export get_tracenorm_MLE

export firstorder_optimality_tangnorm
end
