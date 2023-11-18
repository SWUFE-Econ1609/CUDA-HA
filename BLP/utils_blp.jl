
using Sobol
using StatsFuns:norminvcdf
function draw_shock(nR::Int, nI::Int, dims::Int=1, cut::Int=80)
    sbseq = skip(SobolSeq(dims), cut)
    sb = reduce(hcat, [next!(sbseq) for i = 1:nR*nI])
    sb = norminvcdf.(sb)
    sb = [collect(eachcol(reshape(sb[i, :], nR, nI))) for i in axes(sb, 1)]
    sbh = zeros(nI, nR, dims)
    view(sbh, :, :, 1) .= hcat(sb[1]...)'
    view(sbh, :, :, 2) .= hcat(sb[2]...)'
    sbd = CuArray(sbh)
    return (sbh, sbd)
end


function individual_prob!(par::Vector{T}, pr::Array{T,3}, z::Matrix{T}, x::Matrix{T}, v, delta::Vector{T}, nI::Int, nJ::Int, nR::Int)
    @threads for i in 1:nI
        for r in 1:nR
            for j in 1:nJ
                mu_z = par[1] * z[i, 1] * x[j, 1] + par[2] * z[i, 2] * x[j, 1]
                mu_v = exp(par[3]) * v[i,r,1] * x[j, 1] + exp(par[4]) * v[i,r,2] * x[j, 2]
                pr[i, r, j] = delta[j] + mu_z + mu_v
            end
            softmax!(view(pr, i, r, :))
        end
    end
end

function mktshare(par::Vector{T}, pr::Array{T,3}, z::Matrix{T}, x::Matrix{T}, v, delta::Vector{T}, nI::Int, nJ::Int, nR::Int)
    individual_prob!(par, pr, z, x, v, delta, nI, nJ, nR)
    snew = vec(mean(mean(pr, dims=2), dims=1))
    return snew
end

function mnt_matrix!(mnt::Matrix{T}, prob::Matrix{T}, z::Matrix{T}, zbar::Vector{Float64},x::Matrix{T}, nI::Int, nJ::Int)
    @threads for i in 1:nI
        for j in 1:nJ
            mnt[i, j] = prob[i, j]
            mnt[i, nJ+j] = prob[i, j] * (z[i, 1]-zbar[1])
            mnt[i, nJ*2+j] = prob[i, j] * (z[i, 2])
            mnt[i, nJ*3+j] = prob[i, j] * z[i, 1] * x[j, 1]
            mnt[i, nJ*4+j] = prob[i, j] * z[i, 2] * x[j, 1]
            mnt[i, nJ*5+j] = prob[i, j] * x[j, 1]
            mnt[i, nJ*6+j] = prob[i, j] * x[j, 2]
        end
    end
end

function mnt_obj(par::Vector{T}, mnt::Matrix{T}, prob::Matrix{T}, pr::Array{T,3}, z::Matrix{T}, zbar::Vector{T}, x::Matrix{T}, v::Array{T, 3}, delta0::Vector{T}, delta1::Vector{T}, nI::Int, nJ::Int, nR::Int, struth::Vector{T}, mtruth::Matrix{T}, msim::Matrix{T})
    for _ in 1:10
        snew = mktshare(par, pr, z, x, v, delta0, nI, nJ, nR)
        delta1 .= delta0 .- log.(snew) .+ log.(struth)
        # tolin = maximum(abs.(deltah0 .- deltah1))
        delta0 .= delta1
    end
    individual_prob!(par, pr, z, x, v, delta1, nI, nJ, nR)
    prob .= dropdims(mean(pr, dims=2), dims=2)
    mnt_matrix!(mnt, prob, z, zbar, x, nI, nJ)
    msim .= mean(mnt, dims=1)
    res = norm(msim .- mtruth)
    # @printf("----------------------------\n")
    # @printf("res:%2.8f\n", res)
    # @printf("par1:%2.8f\n", par[1])
    # @printf("par2:%2.8f\n", par[2])
    # @printf("par3:%2.8f\n", exp(par[3]))
    # @printf("par4:%2.8f\n", exp(par[4]))
    # @printf("----------------------------\n")
    return res
end

function individual_xb_kernel!(par::CuDeviceVector{T}, pr::CuDeviceArray{T,3}, z::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, v::CuDeviceArray{T,3}, delta::CuDeviceVector{T}, nI::Int, nJ::Int, nR::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    r = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    j = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if (i <= nI) && (r <= nR) && (j <= nJ)
        mu_z = par[1] * z[i, 1] * x[j, 1] + par[2] * z[i, 2] * x[j, 1]
        mu_v = exp(par[3]) * v[i, r, 1] * x[j, 1] + exp(par[4]) * v[i, r, 2] * x[j, 2]
        pr[i, r, j] = delta[j] + mu_z + mu_v
    end
    return
end

function individual_xb!(par::CuVector{T}, pr::CuArray{T,3}, z::CuMatrix{T}, x::CuMatrix{T}, v::CuArray{T,3}, delta::CuVector{T}, nI::Int, nJ::Int, nR::Int)
    kernel = @cuda launch = false individual_xb_kernel!(par, pr, z, x, v, delta, nI, nJ, nR)
    config = launch_configuration(kernel.fun)
    threads = min(nI * nJ * nR, config.threads)
    threads = (fld(threads, 2^4), 2^3, 2)
    blocks = (cld(nI, threads[1]), cld(nR, threads[2]), cld(nJ, threads[3]))
    CUDA.@sync begin
        kernel(par, pr, z, x, v, delta, nI, nJ, nR; threads, blocks)
    end
    synchronize()
end

function individual_prob!(par::CuVector{T}, pr::CuArray{T,3}, z::CuMatrix{T}, x::CuMatrix{T}, v::CuArray{T,3}, delta::CuVector{T}, nI::Int, nJ::Int, nR::Int)
    individual_xb!(par, pr, z, x, v, delta, nI, nJ, nR)
    softmax!(pr, dims=3)
end
function mktshare(par::CuVector{T}, pr::CuArray{T,3}, z::CuMatrix{T}, x::CuMatrix{T}, v::CuArray{T,3}, delta::CuVector{T}, nI::Int, nJ::Int, nR::Int)
    individual_prob!(par, pr, z, x, v, delta, nI, nJ, nR)
    snew = vec(mean(mean(pr, dims=2), dims=1))
    return snew
end

function mnt_matrix_kernel!(mnt::CuDeviceMatrix{T}, prob::CuDeviceMatrix{T}, z::CuDeviceMatrix{T}, zbar::CuDeviceVector{T}, x::CuDeviceMatrix{T}, nI::Int, nJ::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if (i <= nI) && (j <= nJ)
        mnt[i, j] = prob[i, j]
        mnt[i, nJ+j] = prob[i, j] * (z[i, 1] - zbar[1])
        mnt[i, nJ*2+j] = prob[i, j] * z[i, 2]
        mnt[i, nJ*3+j] = prob[i, j] * z[i, 1] * x[j, 1]
        mnt[i, nJ*4+j] = prob[i, j] * z[i, 2] * x[j, 1]
        mnt[i, nJ*5+j] = prob[i, j] * x[j, 1]
        mnt[i, nJ*6+j] = prob[i, j] * x[j, 2]
    end
    return
end

function mnt_matrix!(mnt::CuMatrix{T}, prob::CuMatrix{T}, z::CuMatrix{T}, zbar::CuVector{T}, x::CuMatrix{T}, nI::Int, nJ::Int)
    kernel = @cuda launch=false mnt_matrix_kernel!(mnt, prob, z, zbar, x, nI, nJ)
    config = launch_configuration(kernel.fun)
    threads = min(nI*nJ, config.threads)
    threads = (fld(threads, 2^2), 2^2)
    blocks = (cld(nI, threads[1]), cld(nJ, threads[2]))
    CUDA.@sync begin
        kernel(mnt, prob, z, zbar, x, nI, nJ;threads, blocks)
    end
    synchronize()
end

function mnt_obj(par::Vector{T}, mnt::CuMatrix{T}, prob::CuMatrix{T}, pr::CuArray{T,3}, z::CuMatrix{T}, zbar::CuVector{T}, x::CuMatrix{T}, v::CuArray{T,3}, delta0::CuVector{T}, delta1::CuVector{T}, nI::Int, nJ::Int, nR::Int, struth::CuVector{T}, mtruth::CuMatrix{T}, msim::CuMatrix{T})
    par = CuArray(par)
    for _ in 1:10
        snew = mktshare(par, pr, z, x, v, delta0, nI, nJ, nR)
        delta1 .= delta0 .- log.(snew) .+ log.(struth)
        delta0 .= delta1
    end
    individual_prob!(par, pr, z, x, v, delta1, nI, nJ, nR)
    prob .= dropdims(mean(pr, dims=2), dims=2)
    mnt_matrix!(mnt, prob, z, zbar, x, nI, nJ)
    msim .= mean(mnt, dims=1)
    synchronize()
    res = - norm(msim .- mtruth)
    # GC.gc(true)
    return res
end
