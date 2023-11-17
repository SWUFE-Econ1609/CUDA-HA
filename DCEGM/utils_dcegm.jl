

struct Parameter{T}
    beta::T
    rho::T
    sig::T
    delta::T
    R::T
    cl::T
    nM::Int
    nV::Int
    nA::Int
    nQ::Int
    nD::Int
    nT::Int
end
Adapt.Adapt.@adapt_structure Parameter

struct Gridh{T}
    yvec::Array{T,1}
    ywgt::Array{T,1}
    avec::Array{T,1}
    mvec::Array{T,1}
end

struct Policyh{T}
    m::Array{T,2}
    c::Array{T,2}
    v::Array{T,2}
end

struct Primeh{T}
    c::Array{T,3}
    m::Array{T,3}
    v::Array{T,3}
    g::Array{T,3}
    p::Array{T,3}
    Euq::Array{T, 2}
    Evq::Array{T, 2}
    Eu::Array{T, 1}
    Ev::Array{T, 1}
end


struct Gridd{T}
    yvec::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    ywgt::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    avec::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    mvec::CuArray{T,1,CUDA.Mem.DeviceBuffer}
end

struct GridDevice{T}
    yvec::CuDeviceArray{T,1,1}
    ywgt::CuDeviceArray{T,1,1}
    avec::CuDeviceArray{T,1,1}
    mvec::CuDeviceArray{T,1,1}
end
Adapt.adapt_structure(to, s::Gridd) = GridDevice(adapt(to, s.yvec), adapt(to, s.ywgt), adapt(to, s.avec), adapt(to, s.mvec))

struct Policyd{T}
    m::CuArray{T,2,CUDA.Mem.DeviceBuffer}
    c::CuArray{T,2,CUDA.Mem.DeviceBuffer}
    v::CuArray{T,2,CUDA.Mem.DeviceBuffer}
end
struct PolicyDevice{T}
    m::CuDeviceArray{T,2,1}
    c::CuDeviceArray{T,2,1}
    v::CuDeviceArray{T,2,1}
end
Adapt.adapt_structure(to, s::Policyd) = PolicyDevice(adapt(to, s.m), adapt(to, s.c), adapt(to, s.v))


struct Primed{T}
    m::CuArray{T,3,CUDA.Mem.DeviceBuffer}
    c::CuArray{T,3,CUDA.Mem.DeviceBuffer}
    v::CuArray{T,3,CUDA.Mem.DeviceBuffer}
    g::CuArray{T,3,CUDA.Mem.DeviceBuffer}
    p::CuArray{T,3,CUDA.Mem.DeviceBuffer}
    Euq::CuArray{T,2,CUDA.Mem.DeviceBuffer}
    Evq::CuArray{T,2,CUDA.Mem.DeviceBuffer}
    Eu::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    Ev::CuArray{T,1,CUDA.Mem.DeviceBuffer}
end
struct PrimeDevice{T}
    m::CuDeviceArray{T,3,1}
    c::CuDeviceArray{T,3,1}
    v::CuDeviceArray{T,3,1}
    g::CuDeviceArray{T,3,1}
    p::CuDeviceArray{T,3,1}
    Euq::CuDeviceArray{T,2,1}
    Evq::CuDeviceArray{T,2,1}
    Eu::CuDeviceArray{T,1,1}
    Ev::CuDeviceArray{T,1,1}
end
Adapt.adapt_structure(to, s::Primed) = PrimeDevice(adapt(to, s.m), adapt(to, s.c), adapt(to, s.v), adapt(to, s.g), adapt(to, s.p), adapt(to, s.Euq), adapt(to, s.Evq), adapt(to, s.Eu), adapt(to, s.Ev))

function util(x::T, par::Parameter, id::Int)
    if (abs(par.rho) - 1.0e0) < 1e-6
        log(x) + par.delta * (id == 1) -20.0
    else
        x^(1.0e0 - par.rho) / (1.0e0 - par.rho) + par.delta * (id == 1) -20.0
    end
end

function mutil(x::T, par::Parameter)
    x^(-par.rho)
end

function mutil(xs::Vector{T}, par::Parameter)
    out = similar(xs)
    for (i, x) in enumerate(xs)
        out[i] = mutil(x, par)
    end
    return out
end

function imutil(x::T, par::Parameter)
    x^(-1.0 / par.rho)
end

function nonlinspace(n::Int, xl=1e-6, xu=40, phi=1.1)
    out = zeros(T, n)
    out[1] = xl
    for i in 2:n
        out[i] = out[i-1] + (xu - out[i-1]) / (n - i + 1)^phi
    end
    return out
end

using FastGaussQuadrature: gausshermite
function log_hermite(nQ::Int, eta::T, mu::T=-eta^2 / 2)
    (x, w) = gausshermite(nQ)
    x = exp.(sqrt(2) .* eta .* x .+ mu)
    w = w ./ sqrt(pi)
    return (x, w)
end

using Interpolations
function interp(xs::Vector{Float64}, ys::Vector{Float64}, xp::Vector{Float64})
    itp = extrapolate(interpolate((vcat(0.0, xs),), vcat(0.0, ys), Gridded(Linear())), Line())
    itp(xp)
end

function policy_terminal!(par::Parameter, pol::Policyh{T}, gd::Gridh{T})
    @threads for ix in 1:par.nM
        for id in 1:par.nD
            pol.m[ix, id] = gd.mvec[ix]
            pol.c[ix, id] = pol.m[ix]
            pol.v[ix, id] = -1.0e0 / util(pol.c[ix, id], par, par.nD)
        end
    end
end

function policy_terminal_kernel!(par::Parameter, pol::PolicyDevice{T}, gd::GridDevice{T})
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if ix <= par.nM
        for id = 1:par.nD
            pol.m[ix, id] = gd.mvec[ix]
            pol.c[ix, id] = pol.m[ix, id]
            pol.v[ix, id] = -1.0e0 / util(pol.c[ix, id], par, par.nD)
        end
    end
    return
end

function policy_terminal!(par::Parameter, pol::Policyd{T}, gd::Gridd{T})
    kernel = @cuda launch = false policy_terminal_kernel!(par, pol, gd)
    config = launch_configuration(kernel.fun)
    threads = min(par.nM, config.threads)
    blocks = cld(par.nM, threads)
    CUDA.@sync begin
        kernel(par, pol, gd; threads, blocks)
    end
end

function wealth_dynamic!(par::Parameter, iid::Int, pri::Primeh, pol::Policyh, gd::Gridh)

    @threads for ix in 1:par.nA
        pri.m[ix, :, iid] .= par.R * gd.avec[ix] .+ gd.yvec .* (iid==1)
        replace!(x -> x < par.cl ? par.cl : x, view(pri.m, ix, :, iid))
        pri.c[ix, :, iid] .= interp(pol.m[:, iid], pol.c[:, iid], pri.m[ix, :, iid])
        pri.c[ix, :, iid]
        pri.v[ix, :, iid] .= -1.0 ./ interp(pol.m[:, iid], pol.v[:, iid], pri.m[ix, :, iid])
        pri.g[ix, :, iid] .= mutil(pri.c[ix, :, iid], par)
    end
end
function upper_envelope(m::Vector{Float64}, v::Vector{Float64}, c::Vector{Float64})
    comM = sort(m)
    fall = Int64[]
    increase = Int64[]
    push!(increase, 1)
    i = 2
    nM = length(m)
    while i <= nM - 1
        if ((m[i+1] < m[i]) && (m[i] > m[i-1])) || ((v[i] < v[i-1]) && (m[i] > m[i-1]))
            push!(fall, i)
            k = i
            while (m[k+1] < m[k])
                k = k + 1
            end
            increase = push!(increase, k)
            i = k
        end
        i = i + 1
    end
    fall = push!(fall, nM)
    nK = length(fall)
    nP = length(comM)
    comV = ones(nP, nK) .* typemin(Float64)
    comC = ones(nP, nK) .* typemin(Float64)
    for j = 1:nK
        below = m[increase[j]] .≥ comM
        above = m[fall[j]] .≤ comM
        inrange = (above .+ below) .== 0
        comV[inrange, j] = interp(m[increase[j]:fall[j]], v[increase[j]:fall[j]], comM[inrange])
        comC[inrange, j] = interp(m[increase[j]:fall[j]], c[increase[j]:fall[j]], comM[inrange])
    end

    (uv, idl) = findmax(comV, dims=2)
    uv = vec(uv)
    um = comM
    uc = vec(comC[idl])
    if isinf(uv[1])
        #uv[1] = 1.0e6
        #uc[1] = 1e-6
    end
    ininf = isinf.(uv)
    if length(sum(ininf)) > 0
        uv[ininf] .= interp(um[.!ininf], uv[.!ininf], um[ininf])
        uc[ininf] .= interp(um[.!ininf], uc[.!ininf], um[ininf])
    end
    uc[1] = c[1]
    uv[1] = v[1]
    return um, uv, uc
end
function wealth_dynamic_kernel!(par::Parameter, iid::Int, pri::PrimeDevice, pol::PolicyDevice, gd::GridDevice)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    iq = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if (ix <= par.nA) && (iq <= par.nQ)
        pri.m[ix, iq, iid] = par.R * gd.avec[ix] + gd.yvec[iq] * (iid==1)
        if pri.m[ix, iq, iid] < par.cl
            pri.m[ix, iq, iid] = par.cl
        end
        if (pri.m[ix, iq, iid] < pol.m[1, iid])
            pri.c[ix, iq, iid] = pol.c[1, iid] / pol.m[1, iid] * pri.m[ix, iq, iid]
            pri.v[ix, iq, iid] = pol.v[1, iid] / pol.m[1, iid] * pri.m[ix, iq, iid]
        elseif (pri.m[ix, iq, iid] >= pol.m[end, iid])
            pri.c[ix, iq, iid] = (pol.c[end, iid] - pol.c[end-1, iid]) / (pol.m[end, iid] - pol.m[end-1, iid]) * pri.m[ix, iq, iid] + pol.c[end-1, iid]
            pri.v[ix, iq, iid] = (pol.v[end, iid] - pol.v[end-1, iid]) / (pol.m[end, iid] - pol.m[end-1, iid]) * pri.m[ix, iq, iid] + pol.v[end-1, iid]
        else
            pri.v[ix, iq, iid] = -40.0f0
            for iix in 1:(par.nM-1)
                if (pol.m[iix, iid] < pri.m[ix, iq, iid]) && (pol.m[iix+1, iid] >= pri.m[ix, iq, iid])
                    tmp = (pol.v[iix+1, iid] - pol.v[iix, iid]) / (pol.m[iix+1, iid] - pol.m[iix, iid]) * (pri.m[ix, iq, iid] - pol.m[iix, iid]) + pol.v[iix, iid]
                    if tmp > pri.v[ix, iq, iid]
                        pri.v[ix, iq, iid] = tmp
                        pri.c[ix, iq, iid] = (pol.c[iix+1, iid] - pol.c[iix, iid]) / (pol.m[iix+1, iid] - pol.m[iix, iid]) * (pri.m[ix, iq, iid] - pol.m[iix, iid]) + pol.c[iix, iid]
                    end
                end
            end
        end
        pri.g[ix, iq, iid] = mutil(pri.c[ix, iq, iid], par)
        pri.v[ix, iq, iid] = -1.0e0 / pri.v[ix, iq, iid]
    end
    return
end

function wealth_dynamic!(par::Parameter, iid::Int, pri::Primed, pol::Policyd, gd::Gridd)
    kernel = @cuda launch = false wealth_dynamic_kernel!(par, iid, pri, pol, gd)
    config = launch_configuration(kernel.fun)
    threads = (min(par.nA, config.threads), 1)
    blocks = (cld(par.nA, threads[1]), par.nQ)
    CUDA.@sync begin
        kernel(par, iid, pri, pol, gd; threads, blocks)
    end
    synchronize()
end

function expect_shock!(par::Parameter, pri::Primeh)
    @threads for ix in 1:par.nA
        vmax = maximum(pri.v[ix, :,:], dims=2)
        expsum = sum(exp.((pri.v[ix, :,:] .- vmax)./par.sig), dims=2)
        logsum = vmax .+ par.sig .* log.(expsum)
        pri.Evq[ix, :] .= logsum
        for iid in 1:par.nD
            pri.p[ix, :, iid] .= exp.((pri.v[ix, :, iid] .- logsum)./par.sig)
        end
        pri.Euq[ix, :] .= sum(pri.p[ix, :, :] .* pri.g[ix, :, :], dims=2) .* par.beta .* par.R
    end
end
function expect_shock_kernel!(par::Parameter, pri::PrimeDevice)
    ix = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    iq = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if (ix <= par.nA) && (iq <= par.nQ)
        vmax = pri.v[ix, iq, 1]
        for iid in 2:par.nD
            if pri.v[ix, iq, iid] > vmax
                vmax = pri.v[ix, iq, iid]
            end
        end
        expsum = 0.0f0
        for iid in 1:par.nD
            expsum += exp((pri.v[ix, iq, iid] - vmax) / par.sig)
        end
        logsum = vmax + par.sig * log(expsum)
        pri.Evq[ix, iq] = logsum
        pri.Euq[ix, iq] = 0.0e0
        for iid in 1:par.nD
            pri.p[ix, iq, iid] = exp((pri.v[ix, iq, iid] - logsum) / par.sig)
            pri.Euq[ix, iq] += pri.p[ix, iq, iid] * pri.g[ix, iq, iid] * par.beta * par.R
        end
    end
    return
end
function expect_shock!(par::Parameter, pri::Primed)
    kernel = @cuda launch = false expect_shock_kernel!(par, pri)
    config = launch_configuration(kernel.fun)
    threads = (min(par.nA, config.threads), 1)
    blocks = (cld(par.nA, threads[1]), par.nQ)
    CUDA.@sync begin
        kernel(par, pri; threads, blocks)
    end
    synchronize()
end

function expect_inv!(id::Int, par::Parameter, pri::Primeh, gd::Gridh, pol::Policyh)
    @threads for ix in 1:par.nM
        if ix <= par.nV
            pol.c[ix, id] = imutil(pri.Eu[1], par) / (par.nV+1) *ix
            pol.m[ix, id] = pol.c[ix, id]
            pol.v[ix, id] = -1.0e0 / (util(pol.c[ix, id], par, id) + par.beta*pri.Ev[1])
        elseif ix <= par.nM
            pol.c[ix, id] = imutil(pri.Eu[ix-par.nV], par)
            pol.m[ix, id] = pol.c[ix, id] + gd.avec[ix-par.nV]
            pol.v[ix, id] = -1.0e0 /(util(pol.c[ix, id], par, id) + par.beta*pri.Ev[ix-par.nV])
        end
    end
end

function expect_inv_kernel!(id::Int, par::Parameter, pri::PrimeDevice, gd::GridDevice, pol::PolicyDevice)
    ix = threadIdx().x + (blockIdx().x -1)*blockDim().x
    if ix <= par.nV
        pol.c[ix, id] = imutil(pri.Eu[1], par) / (par.nV+1) * ix 
        pol.m[ix, id] = pol.c[ix, id]
        pol.v[ix, id] = -1.0e0 / (util(pol.c[ix, id], par, id) + par.beta*pri.Ev[1])
    elseif ix<=par.nM
        pol.c[ix, id] = imutil(pri.Eu[ix-par.nV], par)
        pol.m[ix, id] = pol.c[ix, id] + gd.avec[ix-par.nV]
        pol.v[ix, id] = -1.0e0 / (util(pol.c[ix, id], par, id) + par.beta*pri.Ev[ix-par.nV])
    end
    return
end
function expect_inv!(id::Int, par::Parameter, pri::Primed, gd::Gridd, pol::Policyd)
    kernel = @cuda launch = false expect_inv_kernel!(id, par, pri, gd, pol)
    config = launch_configuration(kernel.fun)
    threads = min(par.nM, config.threads)
    blocks = cld(par.nM, config.threads)
    CUDA.@sync begin
        kernel(id, par, pri, gd, pol;threads, blocks)
    end
end

function solve_egm!(par::Parameter, pol::Vector{Policyh{T}}, pri::Primeh, gd::Gridh)
    policy_terminal!(par, pol[par.nT], gd)
    for it in par.nT-1:-1:1
        for id in 1:par.nD
            for iid in 1:par.nD
                wealth_dynamic!(par, iid, pri, pol[it+1], gd)
            end
            expect_shock!(par, pri)
            mul!(pri.Eu, pri.Euq, gd.ywgt)
            mul!(pri.Ev, pri.Evq, gd.ywgt)
            expect_inv!(id, par, pri, gd, pol[it])
            irange = par.nV+1:par.nM
            m, v, c = upper_envelope(pol[it].m[irange, id], pol[it].v[irange, id], pol[it].c[irange, id])
            pol[it].m[irange, id] .= m
            pol[it].c[irange, id] .= c
            pol[it].v[irange, id] .= v
        end
    end
end

function solve_egm!(par::Parameter, pol::Vector{Policyd{T}}, pri::Primed, gd::Gridd)
    policy_terminal!(par, pol[par.nT], gd)
    for it in par.nT-1:-1:1
        for id in 1:par.nD
            for iid in 1:par.nD
                wealth_dynamic!(par, iid, pri, pol[it+1], gd)
            end
            expect_shock!(par, pri)
            mul!(pri.Eu, pri.Euq, gd.ywgt)
            mul!(pri.Ev, pri.Evq, gd.ywgt)
            expect_inv!(id, par, pri, gd, pol[it])
        end
    end
end
