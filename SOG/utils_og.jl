using Adapt
T = Float64

struct Parameter{T}
    beta::T
    gamma::T
    delta::T
    alpha::T
end
Adapt.@adapt_structure Parameter

struct Gridh{T}
    nK::Int
    nZ::Int
    kvec::Array{T,1}
    zvec::Array{T,1}
    PI::Array{T,2}
end

struct Gridd{T}
    nK::Int
    nZ::Int
    kvec::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    zvec::CuArray{T,1,CUDA.Mem.DeviceBuffer}
    PI::CuArray{T,2,CUDA.Mem.DeviceBuffer}
end
struct GridDevice{T}
    nK::Int
    nZ::Int
    kvec::CuDeviceArray{T,1,1}
    zvec::CuDeviceArray{T,1,1}
    PI::CuDeviceArray{T,2,1}
end
Adapt.adapt_structure(to, s::Gridd) = GridDevice(adapt(to, s.nK), adapt(to, s.nZ), adapt(to, s.kvec), adapt(to, s.zvec), adapt(to, s.PI))

function value_iter!(u::Array{T,3}, v::Matrix{T}, Tv::Matrix{T}, Ev::Matrix{T}, gd::Gridh{T}, par::Parameter{T})
    for _ in 1:10
        mul!(Ev, v, gd.PI)
        @threads for ik in 1:gd.nK
            for iz in 1:gd.nZ
                for jk in 1:gd.nK
                    c = exp(gd.zvec[iz]) * gd.kvec[ik]^par.alpha + (1.0 - par.delta) * gd.kvec[ik] - gd.kvec[jk]
                    if c < 0.0
                        u[ik, iz, jk] = -1e12
                    else
                        u[ik, iz, jk] = c^(1.0 - par.gamma) / (1.0 - par.gamma) + Ev[jk, iz]
                    end
                end
            end
        end
        maximum!(Tv, u)
        v .= Tv
    end
end

function value_bellman_kernel!(u::CuDeviceArray{T,3}, Ev::CuDeviceArray{T,2}, gd::GridDevice{T}, par::Parameter{T})
    ik = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    jk = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if (ik <= gd.nK) && (jk <= gd.nK)
        for iz in 1:gd.nZ
            c = exp(gd.zvec[iz]) * gd.kvec[ik]^par.alpha + (1.0 - par.delta) * gd.kvec[ik] - gd.kvec[jk]
            if c < 0.0e0
                u[ik, iz, jk] = -1e12
            else
                u[ik, iz, jk] = c^(1.0 - par.gamma) / (1.0 - par.gamma) + Ev[jk, iz]
            end
        end
    end
    return
end

function value_bellman!(u::CuArray{T,3}, Ev::CuArray{T,2}, gd::Gridd{T}, par::Parameter{T})
    kernel = @cuda launch = false value_bellman_kernel!(u, Ev, gd, par)
    config = launch_configuration(kernel.fun)
    thread = min(gd.nK^2, Int(floor(sqrt(config.threads))))
    threads = (thread, thread)
    blocks = (cld(gd.nK, threads[1]), cld(gd.nK, threads[2]))
    CUDA.@sync begin
        kernel(u, Ev, gd, par; threads, blocks)
    end
    synchronize()
end

function value_iter!(u::CuArray{T,3}, v::CuArray{T,2}, Tv::CuArray{T,2}, Ev::CuArray{T,2}, gd::Gridd{T}, par::Parameter{T})
    for _ in 1:10
        mul!(Ev, v, gd.PI)
        value_bellman!(u, Ev, gd, par)
        maximum!(Tv, u)
        v .= Tv
    end
end