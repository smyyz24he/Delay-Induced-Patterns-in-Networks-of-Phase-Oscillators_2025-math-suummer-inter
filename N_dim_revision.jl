using DifferentialEquations

using LinearAlgebra

using Plots

using LaTeXStrings

 

N = 10

omega = 1

strength = 0.1

Omega = strength*Matrix(I, N, N) #N*N matrix

 

# random initial condition

u0 = zeros(2*N,1)

u0[1:N] = randn(N)

u0[N+1:2*N] = randn(N)

 

#time

tspan = (0, 100)

 

# RHS：du = f(u, t)

function RHS!(du, u, p, t)

    v = u[1:N]

    x = u[N+1:end]

    dx=v

    dv = -omega^2 .* x + Omega* x

 

    du[1:N] = dv

    du[N+1:end] = dx

end

 

# ode

prob = ODEProblem(RHS!, u0, tspan)

 

# solve

sol = solve(prob, Tsit5(), saveat = 0.1)

 

v = sol[1:N,:]

x = sol[N+1:2*N,:]

t = sol.t

 

plot(x[1,:],x[N,:], xlabel = L"x_1", ylabel = L"x_N",

    title = (L"x_1") * "vs " * (L"x_N") * " " * (L"(N = %$N)") , legend = false)