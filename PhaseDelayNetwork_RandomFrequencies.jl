using DifferentialEquations

using Parameters

using Plots

 

#parameters

@with_kw mutable struct Params

    N :: Real = 50     

    ω0 :: Real = 1    

    γ :: Real = .1 

    ϵ :: Real = 0.1   

    v :: Real = 1       

    σ :: Real = 4     

    τ0 :: Real = 0.1    

end

 

# Spatial distance function on a ring

function dist(i::Real,j::Real,N::Real)

    return minimum([abs(i−j), N−abs(i−j)])

end

 

# Connection weight functions

function WExp(x,σ)

    return exp(-abs(x)/σ)

end

 

function WWizard(x,σ)

    return (1-abs(x)/σ)*exp(-abs(x)/σ)

end

 

# Build connection matrix

function MakeConnections(params)

    N = params.N

    σ = params.σ

    W = zeros(N,N)

    for i = 1:N, j = 1:N

        W[i,j] = WWizard(dist(i,j,N),σ)

    end

    return W

end

 

function MakeFrequencies(params)

    N = params.N

    ω0 = params.ω0

    γ = params.γ

 

    x = rand(N,1)

    ω = zeros(N,1)

    ω .= ω0 .+γ*tan.(pi*(x.-1/2))

 

    return ω

 

end

 

# Build delay matrix

function MakeDelays(params)

    N, v, τ0 = params.N, params.v, params.τ0

    τ = zeros(N,N)

    for i = 1:N, j = 1:N

        τ[i,j] = i == j ? τ0 : dist(i,j,N)/v

    end

    return τ

end





#Delayed Kuramoto Dynamics

 

function PhaseDelayNetwork(du,u,h,p,t)

    P1, W, τ, ω = p

    @unpack N, ω0, γ, ϵ, v, σ, τ0 = P1

 

   

    for i = 1:N

        du[i] = ω[i]     #originally θ̇ᵢ = ω

        for j = 1:N

            hist = h(p, t - τ[i,j])[j]

            du[i] += ϵ * W[i,j] * sin(hist - u[i])

        end

    end

end




# Simulation

 

P1 = Params()

W = MakeConnections(P1)

τ = MakeDelays(P1)

ω = MakeFrequencies(P1)

P = (P1, W, τ, ω)

 

N = Int(P1.N)

θ0 = collect(LinRange(0,2π,N))   #Initial condition(uniformly distributed)

tend = 100 * (2π / P1.ω0)

tspan = (0.0, tend)

h(p, t) = θ0                     #Constant history function

 

prob = DDEProblem(PhaseDelayNetwork, θ0, h, tspan, P)

alg = MethodOfSteps(RK4())

Tsample = 1000

 

sol = solve(prob, alg, saveat=tend/Tsample, abstol=1e-9, reltol=1e-8)

θ = sol[1:N, :]

t = sol.t




#Synchrony Order Parameter

Z = complex(zeros(length(t)))

R = zeros(length(t))

Θ = zeros(length(t))

 

for n in 1:length(t)

    Z[n] = sum(exp.(im .* θ[:,n])) / N

    R[n] = abs(Z[n])

    Θ[n] = angle(Z[n])

end





#plots

default(show = true)

spam = LinRange(0, 2π, 200)

anim = Animation()

 

for n in 1:length(t)

    x, y = cos.(θ[:,n]), sin.(θ[:,n])

    Rx, Ry = R[n]*cos(Θ[n]), R[n]*sin(Θ[n])

 

    plt = plot(x, y, seriestype=:scatter, markersize=6, color=:green, legend=false, xlims=(-1.1,1.1), ylims=(-1.1,1.1))

    plot!(plt, cos.(spam), sin.(spam), lw=1, color=:black)

    scatter!([Rx], [Ry], color=:blue, markersize=8)

    plot!([0, Rx], [0, Ry], lw=2, color=:black)

 

    frame(anim)

end

 

gif(anim, "PhaseDelayNetwork.gif", fps=10)

 

#Plot R vs Time

plt2 = plot(t, R, lw=2, xlabel="Time", ylabel="R(t)", title="Order Parameter over Time", legend=false)

display(plt2)