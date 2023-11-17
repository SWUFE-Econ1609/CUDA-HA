
using Adapt
using CUDA
using .Threads
using LinearAlgebra: mul!
using Statistics
using DataFrames
using CSV
T = Float64

include("utils_dcegm.jl")
Grid = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# Grid = [100, 200]
names = vcat(:thread, Symbol.("Grid" .* string.(Grid)))
Time_CPU = zeros(1, length(Grid))
Time_GPU = zeros(1, length(Grid))
THREADS = ones(Int, length(Grid) + 1) * nthreads()
for (i, nM) in enumerate(Grid)
    @show nM
    # nM = 2000
    nV = div(nM, 10)
    nA = nM - nV
    nQ = 13
    nD = 2
    nT = 20
    cl = 1.0e-6
    R = 1.04e0
    eta = sqrt(0.5)

    beta = 0.98
    rho = 2.0e0
    sig = 0.005
    delta = -1.0
    par = Parameter(beta, rho, sig, delta, R, cl, nM, nV, nA, nQ, nD, nT)

    xl = 1.0e-6
    xu = 400.e0
    mvech = nonlinspace(nM, xl, xu * 1.5e0, 1.1e0)
    avech = nonlinspace(nA, xl, xu, 1.1e0)


    nodes, wgt = log_hermite(nQ, eta)
    yvech = nodes .* 2.0
    ywgth = wgt


    polh = [Policyh(zeros(T, nM, nD), zeros(T, nM, nD), zeros(T, nM, nD)) for _ in 1:nT]


    prih = Primeh(zeros(T, nA, nQ, nD), zeros(T, nA, nQ, nD), zeros(T, nA, nQ, nD), zeros(T, nA, nQ, nD), zeros(T, nA, nQ, nD), zeros(T, nA, nQ), zeros(T, nA, nQ), zeros(T, nA), zeros(T, nA))


    gdh = Gridh(yvech, ywgth, avech, mvech)
    solve_egm!(par, polh, prih, gdh)
    Time_CPU[i] = mean([@elapsed solve_egm!(par, polh, prih, gdh) for _ in 1:20])


    if isone(nthreads())
        mvecd = CuArray(mvech)
        avecd = CuArray(avech)
        yvecd = CuArray(yvech)
        ywgtd = CuArray(ywgth)
        pold = [Policyd(CUDA.zeros(T, nM, nD), CUDA.zeros(T, nM, nD), CUDA.zeros(T, nM, nD)) for _ in 1:nT]
        prid = Primed(CUDA.zeros(T, nA, nQ, nD), CUDA.zeros(T, nA, nQ, nD), CUDA.zeros(T, nA, nQ, nD), CUDA.zeros(T, nA, nQ, nD), CUDA.zeros(T, nA, nQ, nD), CUDA.zeros(T, nA, nQ), CUDA.zeros(T, nA, nQ), CUDA.zeros(T, nA), CUDA.zeros(T, nA))
        gdd = Gridd(yvecd, ywgtd, avecd, mvecd)
        solve_egm!(par, pold, prid, gdd)
        Time_GPU[i] = mean([CUDA.@elapsed solve_egm!(par, pold, prid, gdd) for _ in 1:20])
    end

end

names = vcat(:thread, Symbol.(string.(Grid)))

tcpu = hcat(nthreads(), Time_CPU)
df_cpu = DataFrame(tcpu, :auto)
rename!(df_cpu, Symbol.(names))

tgpu = hcat(0, Time_GPU)
df_gpu = DataFrame(tgpu, :auto)
rename!(df_gpu, Symbol.(names))

# df_CPU = rename(DataFrame(Time_CPU,:auto), Symbol.(names))
# df_GPU = rename(DataFrame(Time_GPU,:auto), Symbol.(names))

if isone(nthreads())
    CSV.write("performance_dcegm.csv", df_gpu, header=true, append = true)
    CSV.write("performance_dcegm.csv", df_cpu, header=false, append=true)
else
    CSV.write("performance_dcegm.csv", df_cpu, header=false, append=true)
end