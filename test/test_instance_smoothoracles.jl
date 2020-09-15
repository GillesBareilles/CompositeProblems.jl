using CompositeProblems
using StructuredProximalOperators
using Random
using LinearAlgebra
using BenchmarkTools
using ForwardDiff

function check_problem_smoothoracles(pb, x, η)

    f_x = f(pb, x)
    ∇f_x = ∇f(pb, x)
    ∇²f_x_η = ∇²f_h(pb, x, η)

    ηgradfx = dot(η, ∇f_x)
    ηHessf_xη = dot(η, ∇²f_x_η)

    @show ηgradfx, ηHessf_xη

    fgrad(t) = abs(f(pb, x + t*η) - (f_x + t * ηgradfx))
    fhess(t) = abs(f(pb, x + t*η) - (f_x + t * ηgradfx + 0.5 * t^2 * ηHessf_xη))

    return compare_curves(fgrad, fhess)
end


function main()
    println("Least square test")
    n, m = 20, 15
    pb = get_random_qualifiedlasso(n, m, 0.5)

    Random.seed!(1234)
    x = rand(n)
    Random.seed!(4321)
    η = rand(n)

    res = check_problem_smoothoracles(pb, x, η)


    println("\nLogistic test")
    n, m = 200, 150
    pb = get_random_logit(n=n, m=m, sparsity=0.5)

    Random.seed!(1234)
    x = rand(n)
    Random.seed!(4321)
    η = rand(n)

    @show f(pb, x)
    @btime f($pb, $x)
    # @time f(pb, x)

    println()
    println()

    ∇fx_AD = ForwardDiff.gradient(x->f(pb, x), x)
    ∇fx_∇f = zeros(n); CompositeProblems.∇f!(pb, ∇fx_∇f, x)

    @show norm(∇fx_AD), norm(∇fx_∇f)
    @show norm(∇fx_AD - ∇fx_∇f)

    # @time ForwardDiff.gradient(x->f(pb, x), x)
    @btime CompositeProblems.∇f!($pb, $∇fx_∇f, $x)

    println()
    println()
    ∇²f_h_base = zeros(n)
    @btime CompositeProblems.∇²f_h!($pb, $∇²f_h_base, $x, $η)
    @assert false

    ∇²f_h_ADhess = ForwardDiff.hessian(x->f(pb, x), x) * η

    CompositeProblems.∇²f_h!(pb, ∇²f_h_base, x, η)

    @show norm(∇²f_h_ADhess)
    @show norm(∇²f_h_base)
    @show norm(∇²f_h_ADhess - ∇²f_h_base)

    @btime CompositeProblems.∇²f_h!($pb, $∇²f_h_base, $x, $η)

    @assert false
    @btime CompositeProblems.∇²f_h!($pb, $∇²f_h_base, $x, $η)

    res = check_problem_smoothoracles(pb, x, η)
    return display_curvescomparison(res)
end

main()
