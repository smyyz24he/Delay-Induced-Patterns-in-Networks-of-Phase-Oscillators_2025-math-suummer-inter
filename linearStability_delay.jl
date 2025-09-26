using Plots
using NLsolve

#目前有问题，因为图中所有都是unstable
# 参数
N = 60
ω = 1.0
σ = 0.5
d = 1.0
τ0 = 1.0

# 权重函数
w(x) = exp(-abs(x)/2)

# 定义环距离
dist(i,j,N) = min(abs(i-j), N-abs(i-j))

# 定义 τ_j
function τj(j, v, d, τ0)
    if j == 0
        return τ0
    else
        return d * j / v
    end
end

# H 函数和导数
H(x) = sin(x)
H′(x) = cos(x)

# (17) 同步态频率方程
function freq_eq(Ω, v)
    sum_term = sum(w(d*dist(0,j,N)) * H(-Ω * τj(dist(0,j,N), v, d, τ0)) for j in 0:N-1)
    return Ω - (ω + σ * sum_term)
end

# 计算 Ω
function find_Ω(v, guess=1.0)
    sol = nlsolve(x -> [freq_eq(x[1], v)], [guess]; xtol=1e-10, ftol=1e-10)
    return sol.zero[1]
end

# 计算稳定性: 检查 Eμ(λ) 在 λ=0 是否有正实部
function is_stable(Ω, v)
    for μ in 1:N
        val = sum(
            w(d*dist(0,j,N)) * H′(-Ω*τj(dist(0,j,N), v, d, τ0)) *
            (1 - exp(2π*im*(μ-1)*j/N)) for j in 0:N-1
        )
        if real(σ * val) > 0   # 如果有正的增长趋势 -> 不稳定
            return false
        end
    end
    return true
end

# 扫描 1/v
vvals = range(0.4, 1.0, length=150)    # v 的范围
Ωvals = Float64[]
stab = Bool[]

for v in vvals
    Ω = find_Ω(v)
    push!(Ωvals, Ω)
    push!(stab, is_stable(Ω, v))
end

invv = 1.0 ./ vvals

# 分别画稳定/不稳定
plot(xlabel="1/v", ylabel="Ω", lw=2)
plot!(invv[stab], Ωvals[stab], color=:blue, lw=2, label="Stable (solid)")
plot!(invv[.!stab], Ωvals[.!stab], color=:blue, lw=2, ls=:dash, label="Unstable (dashed)")
