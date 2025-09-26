using DifferentialEquations
using Random
using Plots

function kuramoto_ode!(du, u, p, t)
    N, epsilon, omega = p
    @inbounds for i in 1:N
        sum_sin = 0.0
        for j in 1:N
            sum_sin += sin(u[j] - u[i])
        end
        du[i] = omega[i] + epsilon * sum_sin/N
    end
end

function kuramoto_sim(N, epsilon, omega0, gamma; tfinal=100, dt=0.01)
    Random.seed!(123)
    x = rand(N)
    omega = omega0 .+ gamma.* tan.(pi.* (x .- 0.5))  #cauchy distribution

    u0 = rand(N)* 2*pi   #initial

    tspan = (0, tfinal)
    p = (N, epsilon, omega)

    prob = ODEProblem(kuramoto_ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dt)

    anim = @animate for i in 1:length(sol.t)
        theta = sol.u[i]
        x = cos.(theta)
        y = sin.(theta)
        X = sum(x) / N
        Y = sum(y) / N
        R = sqrt(X^2 + Y^2)
        phi = atan(Y, X)

        scatter(x, y, color=:red, label="", legend=false, markersize=6)
        plot!(cos.(2*pi .* LinRange(0,1,100)), sin.(2*pi .* LinRange(0,1,100)), lw=1, color=:blue)
        scatter!([R * cos(phi)], [R * sin(phi)], color=:purple, markersize=8)
        plot!([0, R * cos(phi)], [0, R * sin(phi)], lw=2, color=:purple)
        title!("Synchrony = $(round(R, digits=3)), Theory = $(round(sqrt(1 - 2*gamma/epsilon) * (epsilon > 2*gamma), digits=3))")
        xlims!(-1.1, 1.1); ylims!(-1.1, 1.1)
        #; aspect_ratio=:equal
    end

    gif(anim, "kuramoto.gif", fps=20)
end

# example
kuramoto_sim(100, 1.5, 0.0, 0.2)
