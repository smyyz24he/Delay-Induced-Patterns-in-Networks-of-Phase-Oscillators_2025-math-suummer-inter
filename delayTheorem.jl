using DifferentialEquations
using Parameters
using LinearAlgebra
using Random
using NLsolve
using Plots
using Statistics



#parameters
@with_kw mutable struct Params
    N::Int = 50             
    ω::Float64 = 1.0        
    K::Float64 = 1.0        
    τ::Float64 = 0.5        
    A::Matrix{Float64}      
    k::Int = N - 1          
end

#define function
f(ϕ) = sin(ϕ)
fp(ϕ) = cos(ϕ)

#solve Omega
function find_Ω(ω, K, τ)
    f_Ω(x) = x[1] - ω - K * f(-x[1]*τ)
    try
        res = nlsolve(f_Ω, [ω])  # use ω as initial guess
        return res.zero[1]
    catch
        return NaN
    end
    
end

#DDE
function PhaseDelayNetwork(du, u, h, p, t)
    @unpack N, ω, K, A, k, τ = p
    for i in 1:N
        du[i] = ω
        for j in 1:N
            if A[i,j] != 0
                du[i] += (K/k) * f(h(p, t - τ)[j] - u[i])
            end
        end
    end
end




#differ. K
function scan_K(N; ω=1.0, τ=0.5, Klist=0.0:0.2:6.0)
    A = ones(N,N) .- Matrix{Float64}(I, N, N)    #omit self-cycled
    k = N - 1
    u0 = 2π * rand(N)  

    R_final = []
    stab_pred = []

    for K in Klist
        # synchronized R in them
        Ω = find_Ω(ω, K, τ)
        push!(stab_pred, K * fp(-Ω *τ))

        #simulation
        P = Params(N=N, ω=ω, K=K, τ=τ, A=A, k=k)
        h(p, t) = u0
        prob = DDEProblem(PhaseDelayNetwork, u0, h, (0.0, 200.0), P, constant_lags=[τ])
        sol = solve(prob, MethodOfSteps(Tsit5()), saveat=0.5)

        θ = reduce(hcat, sol.u)
        Rt = [abs(mean(exp.(im .* θ[:, i]))) for i in axes(θ, 2)]
        push!(R_final, mean(Rt[end-50:end]))  #average R
    end

    return Klist, R_final, stab_pred
end




#plots
Klist, R_final, stab_pred = scan_K(50, ω=1.0, τ=0.5)

plot(Klist, R_final, lw=2, label="Simulated R", xlabel="K", ylabel="R")
plot!(Klist, [x > 0 ? 1.0 : 0.0 for x in stab_pred], lw=2, ls=:dash, label="Theory Stable")




