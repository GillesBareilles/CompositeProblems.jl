function get_logit_ionosphere(μ=0.001)
    checkdownload_ionosphere()

    rawdata = readdlm("instances/ionosphere.data", ',')
    A = Matrix{Float64}(rawdata[:, 1:end-1])
    n = size(A, 2)

    y = map(x-> x=="g" ? 1.0 : -1.0, rawdata[:, end])

    return LogisticPb(A, y, regularizer_l1(μ), n, zeros(1), l1Manifold(zeros(1)))
end


function get_logit_gisette(μ = 6.67e-4)
    sampledim = 5000

    if isfile("instances/gisette_scale.jld")
        d = load("instances/gisette_scale.jld")
        A = d["A"]
        y = d["y"]
    else
        rawdata = readdlm("instances/gisette_scale", ' ')
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
        save("instances/gisette_scale.jld", "A", pb.A, "y", pb.y)
    end

    return LogisticPb(A, y, regularizer_l1(μ), sampledim, zeros(1), l1Manifold(zeros(1)))
end


## Random sparse logit
function get_random_logit(;n=80, m=85, sparsity=0.5, μ=0.1, seed=1234)

    Random.seed!(seed)
    A = rand(m, n)*10
    Random.seed!(seed+3)
    y = Vector([rand() > sparsity ? 1.0 : -1.0 for i in 1:m])

    return LogisticPb(A, y, regularizer_l1(μ), n, zeros(1), l1Manifold(zeros(1)))
end
