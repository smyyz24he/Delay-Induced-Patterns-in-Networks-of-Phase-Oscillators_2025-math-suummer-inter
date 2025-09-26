using DifferentialEquations
using LinearAlgebra
using Plots

N = 100
omega = 10
Omega = Matrix(I, N, N) #N*N matrix

# random initial condition
#x0=zeros(N)
#v0=ones(N)
x0 = randn(N)          
v0 = randn(N)          
u0 = vcat(v0, x0)

#time
tspan = (0, 100)
tsteps = range(tspan[1], tspan[2], length=1000)


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
sol = solve(prob, Tsit5())

x1 = [sol[i][N+1] for i in eachindex(sol)]
x2 = [sol[i][N+2] for i in eachindex(sol)]  
x6 = [sol[i][N+6] for i in eachindex(sol)] 
v1 = [sol[i][1] for i in eachindex(sol)]
v2 = [sol[i][2] for i in eachindex(sol)]



plot(x1,x6, xlabel = "x1", ylabel = "x6",
    title = "x1 vs x6 (N=100)", legend = false)


