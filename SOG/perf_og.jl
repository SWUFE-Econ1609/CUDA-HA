using QuantEcon: tauchen, MarkovChain
using LinearAlgebra
using .Threads
using CUDA
include("utils_og.jl")
using Random
using BenchmarkTools
using CSV
using DataFrames

BLAS.set_num_threads(nthreads())
Grid = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
names = vcat(:thread, Symbol.("Grid" .* string.(Grid)))
Time_CPU = zeros(1, length(Grid))
Time_GPU = zeros(1, length(Grid))
THREADS = ones(Int, length(Grid) + 1) * nthreads()
for (i, nK) in enumerate(Grid)
    @show nK
    gamma = 1.50
    delta = 0.10
    beta = 0.95
    alpha = 0.30
    rho = 0.90
    se = 0.1
    par = Parameter(beta, gamma, delta, alpha)

    nZ = 4
    # nK = 1000

    mc = tauchen(nZ, rho, se)
    PIh = mc.p * beta
    zvech = collect(mc.state_values)

    kmin = 0.2
    kmax = 6.0
    kvech = collect(range(kmin, kmax, nK))
    gdh = Gridh(nK, nZ, kvech, zvech, PIh)


    uh = zeros(T, nK, nZ, nK)
    vh = zeros(T, nK, nZ)
    Tvh = zeros(T, nK, nZ)
    Evh = zeros(T, nK, nZ)


    Time_CPU[i] = @belapsed value_iter!($uh, $vh, $Tvh, $Evh, $gdh, $par)
    if isone(nthreads())
        PId = CuArray(PIh)
        zvecd = CuArray(zvech)
        kvecd = CuArray(kvech)
        gdd = Gridd(nK, nZ, kvecd, zvecd, PId)
        ud = CuArray(uh)
        vd = CuArray(vh)
        Tvd = CuArray(Tvh)
        Evd = CuArray(Evh)
        Time_GPU[i] = @belapsed value_iter!($ud, $vd, $Tvd, $Evd, $gdd, $par)
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
    CSV.write("performance_og.csv", df_gpu, header=true, append = true)
    CSV.write("performance_og.csv", df_cpu, header=false, append=true)
else
    CSV.write("performance_og.csv", df_cpu, header=false, append=true)
end
