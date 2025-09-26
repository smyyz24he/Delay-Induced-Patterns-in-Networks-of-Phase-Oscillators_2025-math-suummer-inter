using DifferentialEquations
using Plots

omega = 1.0
u0 = [1.0, 1.0, 0.0, -1.0]  


function RHS!(du, u, p, t)
    v1, x1, v2, x2 = u
    du[1] = -omega^2 * x1 + (x1 - x2)   
    du[2] = v1                          
    du[3] = -omega^2 * x2 + (x2 - x1)   
    du[4] = v2                          
end

# time
tspan = (0, 10)

prob = ODEProblem(RHS!, u0, tspan)
sol = solve(prob, Tsit5())

v1 = sol[1, :]
x1 = sol[2, :]
v2 = sol[3, :]
x2 = sol[4, :]

#plot(x1, v1, label="Particle 1", xlabel="x", ylabel="v", title="Phase Space")
#plot!(x2, v2, label="Particle 2")
plot(x1,x2)
