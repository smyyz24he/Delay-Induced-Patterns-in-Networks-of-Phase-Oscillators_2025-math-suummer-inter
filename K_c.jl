using DifferentialEquations
using Random
using Statistics
using Plots

function kuramoto_ode!(du, u, p, t)
    N, K, omega = p
    @inbounds for i in 1:N
        du[i] = omega[i] + K / N * sum(sin(u[j] - u[i]) for j in 1:N)
    end
end

function simulate_R(N, K, omega, tfinal=100)
    u0 = 2*pi .* rand(N)
    p = (N, K, omega)
    tspan = (0, tfinal)
    prob = ODEProblem(kuramoto_ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=0.1)
    theta = sol.u[end]
    x_1 = mean(cos.(theta))
    y_1 = mean(sin.(theta))
    R = sqrt(x_1^2 + y_1^2)
    return R
end

function run_experiment(; N=100, gamma=1.0)
    Random.seed!(1234)
    x = rand(N)
    omega = gamma.* tan.(pi .* (x .- 0.5))  # Cauchy(0, γ)

    K_values = 0.1:0.1:5.0
    R_values = [simulate_R(N, K, omega) for K in K_values]

    #theorem
    R_theory = [K > 2*gamma ? sqrt(1 - 2*gamma/K) : 0.0 for K in K_values]

    plot(K_values, R_values, label="Simulated R", lw=2, marker=:circle)
    plot!(K_values, R_theory, label="Theoretical R", lw=2, linestyle=:dash, color=:red)

    hline!([1.0], linestyle=:dot, color=:black, label="y=1")

    xlabel!("K (Coupling strength)")
    ylabel!("R (Order parameter)")
    title!("Kuramoto Model: A Phase Transition")
    plot!(legend = :topright)  
end

run_experiment()

