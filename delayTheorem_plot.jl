using DifferentialEquations
using Parameters
using LinearAlgebra
using Random
using NLsolve
using Plots
using Statistics

#include("delayTheorem.jl")  


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

function plot_stability_region(; ω=0.5)
    Klist = -1:0.05:1
    τlist = 0:0.05:8

    #stability = zeros(length(Klist), length(τlist))
    Z_exact  = fill(Float64(NaN), length(Klist), length(τlist))

    for (i, K) in enumerate(Klist)
        #Ω_guess = ω
        for (j, τ) in enumerate(τlist)
            try
                Ω = find_Ω(ω, K, τ)
                println("K=$K, τ=$τ, Ω=$Ω")
                Z_exact[i,j] = (K * fp(-Ω*τ) > 0) ? 1.0 : 0.0
                #Ω_guess = Ω
            catch
                Z_exact[i,j] = NaN   
            end
        end
    end

    my_colors = [colorant"#ff8c3a", colorant"#4c72b0"]

    heatmap(τlist, Klist, Z_exact;
        xlabel="τ (delay)", ylabel="K (coupling strength)",
        title="Stability Region (blue=Stable, orange=Unstable)",
        c=cgrad(my_colors, categorical=true),
        colorbar=false, size=(700,500))
end


plot_stability_region()



