using CompositeProblems
using Test
using ForwardDiff
using LinearAlgebra

@testset "CompositeProblems.jl" begin
    # Write your tests here.

    n, m = 5, 3
    testcollection_pb = [
        ("Least squares", get_random_qualifiedlasso(n, m, 0.5)),
        ("Logistic", get_random_logit(n=n, m=m)),                   # Beware, hessian of logistic is often null for n, m > 10, 10
    ]

    @testset "$testcollection" for (testcollection, pb) in testcollection_pb
        x = rand(n)
        h = rand(n)

        # First order
        ∇f_x = zeros(n)
        ∇f!(pb, ∇f_x, x)

        @test isapprox(∇f_x, ∇f(pb, x))
        @test isapprox(∇f_x, ForwardDiff.gradient(x->f(pb, x), x))

        # Second order
        ∇²f_xh = zeros(n)
        ∇²f_h!(pb, ∇²f_xh, x, h)

        @test isapprox(∇²f_xh, ∇²f_h(pb, x, h))

        ∇²f_xh_AD = ForwardDiff.hessian(x->f(pb, x), x) * h
        @test isapprox(∇²f_xh, ∇²f_xh_AD)
    end
end
