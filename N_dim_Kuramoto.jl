using DifferentialEquations
using LinearAlgebra
using Plots
using LaTeXStrings

# parameters
N = 10
strength = 0.1
Omega = strength * Matrix(I, N, N)  
omega = range(1.0, 1.5, length=N)


u0 = rand(N) .* 2π  # random initial angles in [0, 2π)

tspan = (0, 100)

# RHS
function RHS!(du, u, p, t)
    theta = u
    coupling_matrix = Omega.* sin.(theta.- theta')  

    coupling = sum(coupling_matrix, dims=2) 

    du .= omega.+ coupling
end

# ODE
prob = ODEProblem(RHS!, u0, tspan)

# solve
sol = solve(prob, Tsit5(), saveat = 0.1)

theta = sol[1:N, :]
t = sol.t

# plot
plot(theta[1, :], theta[N, :], xlabel = L"\theta_1", ylabel = L"\theta_N",
    title = (L"\theta_1") * " vs " * (L"\theta_N") * " " * (L"(N = %$N)"),
    legend = false)

#plot(sol.t, sol[10, :], xlabel="Time", ylabel="θ₁", title="Phase of Oscillator 1")
