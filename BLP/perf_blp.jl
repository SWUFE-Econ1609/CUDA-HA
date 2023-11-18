
using CUDA
using Distributions
using StatsFuns: softmax!
using Statistics: norm, mean
using Optim
using Printf
using .Threads
using Random
using DataFrames, CSV
T = Float64
Random.seed!(20230309)
include("utils_blp.jl")
include("nmsmax.jl")

Grid = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000]
tcpu = zeros(1, length(Grid))
tgpu = zeros(1, length(Grid))
for (i, nI) in enumerate(Grid)
    @show nI
    nJ = 20
    nR = 100
    x1 = rand(Uniform(1, 3), nJ)
    x2 = rand(Normal(), nJ)
    xh = hcat(x1, x2)
    beth = [-6.0, 1.0, 1.0]
    xi = rand(Normal(0, 0.5), nJ)
    delta = hcat(ones(nJ), xh) * beth + xi

    z1 = rand(Normal(-1, 1), nI)
    z2 = rand(Normal(1, 0.5), nI)
    zh = hcat(z1, z2)
    thetaz = [1.5, 0.75, 1.25]

    nu1 = rand(Normal(), nI)
    nu2 = rand(Normal(), nI)
    nu = hcat(nu1, nu2)
    theta_v = [1.5, 0.2]

    pr = zeros(nI, nJ)
    mu_z = zeros(nI, nJ)
    mu_v = zeros(nI, nJ)
    chosen = zeros(Int, nI)

    for i in 1:nI
        esum = 0.0
        for j in 1:nJ
            mu_z[i, j] = 0.75 * z1[i] * x1[j] + 1.25 * z2[i] * x1[j]
            mu_v[i, j] = 1.5 * nu1[i] * x1[j] + 3.5 * nu2[i] * x2[j]
            pr[i, j] = delta[j] + mu_z[i, j] + mu_v[i, j]
        end
        softmax!(view(pr, i, :))
        chosen[i] = rand(Categorical(pr[i, :]))
    end
    pr0 = repeat((1:nJ)', nI, 1) .== chosen
    mtruth = hcat(mean(pr, dims=1), mean(pr .* (z1 .- mean(z1)), dims=1), mean(pr .* z2, dims=1), mean(pr .* z1 .* x1', dims=1), mean(pr .* z2 .* x1', dims=1), mean(pr .* x1', dims=1), mean(pr .* x2', dims=1))

    (vh, vd) = draw_shock(nR, nI, 2)

    parh = vcat(0.6, 1.0, log(1.1), log(4.5))
    probh = zeros(nI, nJ)
    prh = zeros(nI, nR, nJ)
    deltah = copy(delta)
    mtruthh = copy(mtruth)
    msimh = similar(mtruthh)

    zbarh = dropdims(mean(zh, dims=1), dims=1)
    mnth = zeros(nI, nJ * 7)
    struthh = vec(mean(pr, dims=1))
    deltah0 = zeros(T, nJ)
    deltah1 = zeros(T, nJ)
    soldh = zeros(T, nJ)
    deltah0 .= deltah
    soldh .= struthh
    funh(par) = mnt_obj(par, mnth, probh, prh, zh, zbarh, xh, vh, deltah0, deltah1, nI, nJ, nR, struthh, mtruth, msimh)
    optimize(funh, parh, iterations=1)
    tcpu[i] = mean([@elapsed optimize(funh, parh, iterations=10) for _ in 1:10])
    if nthreads() == 1
        CUDA.memory_status()
        CUDA.reclaim()
        pard = vcat(0.6, 1.0, log(1.1), log(4.5))
        pardd = CuArray(pard)
        probd = CuArray(probh)
        prd = CuArray(prh)
        deltad = CuArray(deltah)
        zd = CuArray(zh)
        xd = CuArray(xh)
        mtruthd = CuArray(mtruth)
        msimd = CuArray(msimh)
        zbard = CuArray(zbarh)
        mntd = CUDA.zeros(T, nI, nJ * 7)
        struthd = CuArray(struthh)
        deltad0 = CUDA.zeros(T, nJ)
        deltad1 = CUDA.zeros(T, nJ)
        soldd = CUDA.zeros(T, nJ)
        deltad0 .= deltad
        soldd .= struthd
        fund(par) = mnt_obj(par, mntd, probd, prd, zd, zbard, xd, vd, deltad0, deltad1, nI, nJ, nR, struthd, mtruthd, msimd)
        nmsmax(fund, vcat(0.6, 1.0, log(1.1), log(4.5)), max_its=1)
        tgpu[i] = mean([@elapsed nmsmax(fund, vcat(0.6, 1.0, log(1.1), log(4.5)), max_its=10) for _ in 1:10])
    end

end

