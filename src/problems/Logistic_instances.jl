function get_logit_ionosphere(;λ=0.001)
    checkdownload_ionosphere()

    rawdata = readdlm(joinpath(instances_dir, "ionosphere.data"), ',')
    A = Matrix{Float64}(rawdata[:, 1:end-1])
    n = size(A, 2)

    y = map(x-> x=="g" ? 1.0 : -1.0, rawdata[:, end])

    return LogisticPb(A, y, regularizer_l1(λ), n, zeros(1), l1Manifold(zeros(1)))
end


function get_logit_gisette(λ = 6.67e-4)
    sampledim = 5000

    if isfile(joinpath(instances_dir, "gisette_scale.jld"))
        d = load(joinpath(instances_dir, "gisette_scale.jld"))
        A = d["A"]
        y = d["y"]
    else
        println("Reading ", joinpath(instances_dir, "gisette_scale"))
        rawdata = readdlm(joinpath(instances_dir, "gisette_scale"), ' ')

        println("Assembling problem matrix, vector.")
        nsamples = size(rawdata, 1)

        A = zeros(Float64, nsamples, sampledim)
        for i in 1:size(rawdata, 1), j in 2:size(rawdata, 2)
            s = rawdata[i, j]
            s == "" && continue

            indcol = parse.(Int, split(s, ":")[1])
            val = parse.(Float64, split(s, ":")[2])
            A[i, indcol] = val
        end

        y = Vector{Float64}(rawdata[:, 1])

        println("")
        save(joinpath(instances_dir, "gisette_scale.jld"), "A", A, "y", y)
    end

    return LogisticPb(A, y, regularizer_l1(λ), sampledim, zeros(1), l1Manifold(zeros(1)))
end


## Random sparse logit
function get_random_logit(;n=80, m=85, sparsity=0.5, λ=0.1, seed=1234)

    Random.seed!(seed)
    A = rand(m, n)*10
    Random.seed!(seed+3)
    y = Vector([rand() > sparsity ? 1.0 : -1.0 for i in 1:m])

    return LogisticPb(A, y, regularizer_l1(λ), n, zeros(1), l1Manifold(zeros(1)))
end

function get_logit_MLE(;n=20, m=15, sparsity=0.5, seed=1234, λ=0.01)
    @assert 0 ≤ sparsity ≤ 1

    Random.seed!(seed)
    A = rand(Normal(), m, n)

    Random.seed!(seed+1)
    x0 = rand(Normal(), n)
    Random.seed!(seed+2)
    optstructure = l1Manifold(rand(Bernoulli(1-sparsity), n))
    x0 = project(optstructure, x0)

    y = zeros(m)
    for i in 1:m
        Random.seed!(seed+i)
        if rand(Bernoulli(σ(dot(A[i, :], x0))))
            y[i] = 1.0
        else
            y[i] = -1.0
        end
    end

    return LogisticPb(A, y, regularizer_l1(λ), n, x0, optstructure)
end