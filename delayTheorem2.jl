using DifferentialEquations
using Parameters
using LinearAlgebra
using Random
using NLsolve
using Plots
using Statistics

#pairwise delay and given A
# Parameters
@with_kw mutable struct Params
    N::Int = 3
    ω::Float64 = 1.0
    K::Float64 = 1.0
    d::Float64 = 1.0           
    v::Float64 = 1.0            
    A::Matrix{Float64}          #a_ij = a(dist(i,j))
    τ::Matrix{Float64}          #τ_ij = dist(i,j)*d/v
    k::Int = N                 
end


#define
f(ϕ) = sin(ϕ)
fp(ϕ) = cos(ϕ)
a_of_dist(x) = 0.5 * exp(-abs(x))  
ring_dist(i, j, N) = min(abs(i - j), N - abs(i - j))


function build_A_τ(N; d=1.0, v=1.0)
    D = [ring_dist(i, j, N) for i in 1:N, j in 1:N]
    A = a_of_dist.(D)
    τ = (d / v) .* D
    return A, τ
end


#solve Omega
function find_Ω(ω, K, A, τ, k; i_ref=1)
    function F!(x)  
        Ω = x[1]
        s = 0.0
        @inbounds @simd for j in axes(A, 2)
            s += A[i_ref, j] * f(-Ω * τ[i_ref, j])
        end
        return Ω - ω - (K / k) * s
    end
    res = nlsolve(x -> [F!(x)], [ω])  # use ω as initial guess
    return res.zero[1]
end



# DDE system with pairwise delays
function PhaseDelayNetwork(du, u, h, p::Params, t)
    @unpack N, ω, K, A, τ, k = p
    @inbounds for i in 1:N
        acc = 0.0
        @simd for j in 1:N
            # history evaluated at (t - τ_ij), take j-th component
            θj_delay = h(p, t - τ[i, j])[j]
            acc += A[i, j] * f(θj_delay - u[i])
        end
        du[i] = ω + (K / k) * acc
    end
end



# Scan over K
function scan_K(N; ω=1.0, d=1.0, v=1.0, Klist=0.0:0.2:6.0, T=200.0, saveat=0.5, seed=42)
    Random.seed!(seed)
    A, τ = build_A_τ(N; d=d, v=v)
    k = N                          # all-to-all normalization
    u0 = 2π .* rand(N)             

   
    lags = unique(vec(τ))
    lags = sort(unique(filter(>(0.0), lags)))  

    R_final = Float64[]
    stab_pred = Float64[]

    for K in Klist
        # synchronized Ω and simple stability predictor
        Ω = find_Ω(ω, K, A, τ, k)
        # linearization sum S
        S = (K / k) * sum(A[1, j] * fp(-Ω * τ[1, j]) for j in 1:N)
        push!(stab_pred, S)

        # simulate
        P = Params(N=N, ω=ω, K=K, d=d, v=v, A=A, τ=τ, k=k)
        hfun(p, t) = u0
        prob = DDEProblem(PhaseDelayNetwork, u0, hfun, (0.0, T), P; constant_lags=lags)
        sol = solve(prob, MethodOfSteps(Tsit5()); saveat=saveat)

        θ = reduce(hcat, sol.u)                
        Rt = [abs(mean(exp.(im .* θ[:, i]))) for i in axes(θ, 2)]
        push!(R_final, mean(Rt[end-20:end]))   
    end

    return Klist, R_final, stab_pred
end




#try
N = 50
ω = 1.0
d = 1.0
v = 1.0
Klist, R_final, stab_pred = scan_K(N; ω=ω, d=d, v=v, Klist=0.0:0.2:6.0)

plot(Klist, R_final, lw=2, label="Simulated R", xlabel="K", ylabel="R")
plot!(Klist, [S > 0 ? 1.0 : 0.0 for S in stab_pred], lw=2, ls=:dash, label="Theory Stable (S>0)")
