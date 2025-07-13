using Random, StatsBase, Distributions, LinearAlgebra, Plots, Statistics

# Data structure for temporal edge
struct TemporalEdge
    from::Int
    to::Int
    time::Float64
end

# Power-law sampling function
function sample_bounded_powerlaw(N::Int, γ::Float64, xmin::Float64, xmax::Float64)
    @assert γ > 1 "γ must be > 1 for normalizable PDF"
    u = rand(N)
    a = xmin^(1 - γ)
    b = xmax^(1 - γ)
    return (a .+ u .* (b - a)).^(1 / (1 - γ))
end

# Generate activity-driven temporal network
function simulate_activity_network(N::Int, m::Int, T::Int, η::Float64, ε::Float64, γ::Float64)
    x_vals = sample_bounded_powerlaw(N, γ, ε, 1.0)
    a_vals = η .* x_vals
    avg_a = mean(a_vals)
    squared_a = mean(a_vals .^ 2)

    temporal_edges = TemporalEdge[]
    integrated_edges = Set{Tuple{Int, Int}}()

    dt = 0.1
    for t in 1:dt:T
        for i in 1:N
            if rand() < a_vals[i] * dt
                targets = rand(setdiff(1:N, [i]), m)
                for j in targets
                    push!(temporal_edges, TemporalEdge(i, j, t))
                    push!(integrated_edges, (min(i, j), max(i, j)))
                end
            end
        end
    end

    return temporal_edges, integrated_edges, avg_a, squared_a
end

# Helper to organize temporal edges
function build_edge_dict(temporal_edges::Vector{TemporalEdge})
    edge_dict = Dict{Float64, Vector{Tuple{Int, Int}}}()
    for e in temporal_edges
        push!(get!(edge_dict, e.time, []), (e.from, e.to))
    end
    return edge_dict
end

# General function to run SIR/SIS/SEIR on a temporal or aggregated graph
function run_epidemic!(states::Vector{Symbol}, edges::Vector{Tuple{Int, Int}}, N::Int, T::Int, β::Float64, μ::Float64, σ::Union{Float64, Nothing}=nothing)
    results = Dict(:I => Int[], :S => Int[], :E => Int[], :R => Int[])
    mean_degree = max(length(edges) / N, 1e-8)

    for t in 1:T
        new_S, new_E, new_I, new_R = Set{Int}(), Set{Int}(), Set{Int}(), Set{Int}()

        for (i, j) in edges
            if states[i] == :I && states[j] == :S && rand() < β / mean_degree
                push!(new_E, j)
            elseif states[j] == :I && states[i] == :S && rand() < β / mean_degree
                push!(new_E, i)
            end
        end

        for i in 1:N
            if σ !== nothing && states[i] == :E && rand() < σ
                push!(new_I, i)
            elseif states[i] == :I && rand() < μ
                push!(new_R, i)
            end
        end

        for i in new_E
            if states[i] == :S
                states[i] = (σ === nothing ? :I : :E)
            end
        end
        for i in new_I
            states[i] = :I
        end
        for i in new_R
            if epidemic_type == :SIS
                states[i] = :S  # Reinfection possible
            elseif epidemic_type == :SIR || epidemic_type == :SEIR
                states[i] = :R  # Permanent recovery
            end
        end

        for s in [:S, :E, :I, :R]
            push!(results[s], count(==(s), states))
        end
    end

    return results
end

# Initialize states
function initialize_states(N::Int, initial_infected::Int, mode::Symbol)
    states = fill(:S, N)
    for i in rand(1:N, initial_infected)
        states[i] = :I
    end
    return states
end

# Aggregate temporal edges up to τ
function time_aggregated_network(temporal_edges::Vector{TemporalEdge}, τ::Float64)
    return Set([(min(e.from, e.to), max(e.from, e.to)) for e in temporal_edges if e.time ≤ τ])
end

# Simulate epidemic on temporal network
function simulate_epidemic_temporal(N, T, β, μ, σ, initial_infected, epidemic_type, edge_dict)
    states = initialize_states(N, initial_infected, epidemic_type)
    results = Dict(:I => Int[], :S => Int[], :E => Int[], :R => Int[])

    for t in 1:T
        edges = get(edge_dict, t, Tuple{Int,Int}[])
        daily_results = run_epidemic!(states, edges, N, 1, β, μ, epidemic_type == :SEIR ? σ : nothing)
        for k in keys(daily_results)
            push!(results[k], daily_results[k][1])
        end
    end

    return results
end

# Simulate epidemic on aggregated network
function simulate_epidemic_aggregated(N, T, β, μ, σ, initial_infected, epidemic_type, aggregated_edges)
    states = initialize_states(N, initial_infected, epidemic_type)
    edges = collect(aggregated_edges)
    return run_epidemic!(states, edges, N, T, β, μ, epidemic_type == :SEIR ? σ : nothing)
end

# Unified simulation entry point (temporal)
function simulate_temporal(N, m, T, η, ε, γ, β, μ, σ, initial_infected, epidemic_type)
    temporal_edges, _, avg_a, squared_a = simulate_activity_network(N, m, T, η, ε, γ)
    edge_dict = build_edge_dict(temporal_edges)
    results = simulate_epidemic_temporal(N, T, β, μ, σ, initial_infected, epidemic_type, edge_dict)
    return results, avg_a, squared_a
end

# Unified simulation entry point (aggregated)
function simulate_aggregated(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, epidemic_type)
    temporal_edges, _, avg_a, squared_a = simulate_activity_network(N, m, T, η, ε, γ)
    aggregated_edges = time_aggregated_network(temporal_edges, τ)
    results = simulate_epidemic_aggregated(N, T, β, μ, σ, initial_infected, epidemic_type, aggregated_edges)
    return results, avg_a, squared_a
end

# Repeat simulation (temporal)
function repeat_temporal_simulation(N, m, T, η, ε, γ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
    results, avg_a, squared_a = [], [], []
    for _ in 1:num_repeats
        r, a, s = simulate_temporal(N, m, T, η, ε, γ, β, μ, σ, initial_infected, epidemic_type)
        push!(results, r); push!(avg_a, a); push!(squared_a, s)
    end
    return results, avg_a, squared_a
end

# Repeat simulation (aggregated)
function repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
    results, avg_a, squared_a = [], [], []
    for _ in 1:num_repeats
        r, a, s = simulate_aggregated(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, epidemic_type)
        push!(results, r); push!(avg_a, a); push!(squared_a, s)
    end
    return results, avg_a, squared_a
end


# Network parameters
N = 1000 # Number of nodes
m = 5   # Number of links per active node
T = 500   # Number of time steps
η = 10.0  # Rescaling factor for activity rates
ε = 0.001  # Lower cutoff for x_i 
γ = 2.1  # Power-law exponent
τ = 1.0  # Time aggregation threshold

# Epidemic parameters
β = 0.04  # Infection rate
μ = 0.01  # Recovery rate
σ = 0.00  # Exposed to Infected rate (for SEIR)
initial_infected = 10  # Number of initially infected nodes

# Simylation parameters
num_repeats = 100  # Number of simulation runs
## SIS
epidemic_type = :SIS  
results_SIS, avg_a_SIS, squared_a_SIS = repeat_temporal_simulation(N, m, T, η, ε, γ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected_matrix_SIS = reduce(hcat, [r[:I] for r in results_SIS])  # T × R matrix
avg_infected_SIS = mean(infected_matrix_SIS, dims=2)             # mean over runs
std_infected_SIS = std(infected_matrix_SIS, dims=2)

τ=20.0
results20_SIS, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected20_matrix_SIS = reduce(hcat, [r[:I] for r in results20_SIS])  
avg_infected20_SIS = mean(infected20_matrix_SIS, dims=2)            
std_infected20_SIS = std(infected20_matrix_SIS, dims=2)

τ=40.0
results40_SIS, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected40_matrix_SIS = reduce(hcat, [r[:I] for r in results40_SIS])  
avg_infected40_SIS = mean(infected40_matrix_SIS, dims=2)            
std_infected40_SIS = std(infected40_matrix_SIS, dims=2)

plot(avg_infected40_SIS, ribbon=std_infected40_SIS, lw=2, label="Aggregated T=40", color=:dodgerblue)   
plot!(avg_infected20_SIS, ribbon=std_infected20_SIS, lw=2, label="Aggregated T=20", color=:crimson) 
plot!(avg_infected_SIS, ribbon=std_infected_SIS/2 ,lw=2, title="SIS Infected Curve", xlabel="Time Steps", ylabel="Number of Infected Nodes", label="Activity-Driven", legend=:bottomright, color=:darkorange)

savefig("peak_SIS.png")
### Critical R0
T = 500   # Number of time steps
num_repeats = 10  # Number of simulation runs
R0_times = [0.1, 0.12, 0.15, 0.17, 0.2, 0.22, 0.25, 0.27, 0.3, 0.32, 0.35, 0.37, 0.4, 0.45, 0.5]
I_inf_temporal = Float64[]
I_inf_agg20 = Float64[]
I_inf_agg40 = Float64[]

sum_a = 0.0
sum_a_squared = 0.0

for R0 in R0_times
    β_R = R0 * μ
    sum_I_temporal = 0.0
    sum_I_agg20 = 0.0
    sum_I_agg40 = 0.0

    for _ in 1:num_repeats
        results, a_avg, a_sq = simulate_temporal(
            N, m, T, η, ε, γ, β_R, μ, σ, initial_infected, epidemic_type
        )
        sum_I_temporal += results[:I][end]
        sum_a += a_avg
        sum_a_squared += a_sq

        results20, _, _ = simulate_aggregated(
            N, m, T, η, ε, γ, τ, β_R, μ, σ, initial_infected, epidemic_type
        )
        sum_I_agg20 += results20[:I][end]

        results40, _, _ = simulate_aggregated(
            N, m, T, η, ε, γ, τ, β_R, μ, σ, initial_infected, epidemic_type
        )
        sum_I_agg40 += results40[:I][end]
    end

    push!(I_inf_temporal, sum_I_temporal / num_repeats)
    push!(I_inf_agg20, sum_I_agg20 / num_repeats)
    push!(I_inf_agg40, sum_I_agg40 / num_repeats)
end

avg_a = sum_a / (num_repeats * length(R0_times))
avg_a_sq = sum_a_squared / (num_repeats * length(R0_times))

R0_critical = 2 * avg_a / (avg_a + sqrt(avg_a_sq))
println("Estimated critical R₀: ", R0_critical)
# Plotting
plot(R0_times, I_inf_agg40, label="Aggregated τ = 40", lw=2, marker=:diamond, color=:dodgerblue)
plot!(R0_times, I_inf_agg20, label="Aggregated τ = 20", lw=2, marker=:square, color=:crimson)
plot!(R0_times, I_inf_temporal, label="Temporal", lw=2, marker=:circle, color=:darkorange)
scatter!([R0_critical], [0], label="Critical R₀", color=:lime, marker=:star, markersize=8)
plot!(xlabel="R₀", ylabel="I∞", title="Steady-state infected fraction vs R₀", legend=:topleft)
## SIR
epidemic_type = :SIR 
results_SIR, avg_a_SIR, squared_a_SIR = repeat_temporal_simulation(N, m, T, η, ε, γ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected_matrix_SIR = reduce(hcat, [r[:I] for r in results_SIR]) 
avg_infected_SIR = mean(infected_matrix_SIR, dims=2)          
std_infected_SIR = std(infected_matrix_SIR, dims=2)

τ=20.0
results20_SIR, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected20_matrix_SIR = reduce(hcat, [r[:I] for r in results20_SIR])  
avg_infected20_SIR = mean(infected20_matrix_SIR, dims=2)            
std_infected20_SIR = std(infected20_matrix_SIR, dims=2)

τ=40.0
results40_SIR, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected40_matrix_SIR = reduce(hcat, [r[:I] for r in results40_SIR])  
avg_infected40_SIR = mean(infected40_matrix_SIR, dims=2)            
std_infected40_SIR = std(infected40_matrix_SIR, dims=2)


plot(avg_infected40_SIR, ribbon=std_infected40_SIR, lw=2, label="Aggregated T=40", color=:dodgerblue)   
plot!(avg_infected20_SIR, ribbon=std_infected20_SIR, lw=2, label="Aggregated T=20", color=:crimson) 
plot!(avg_infected_SIR, ribbon=std_infected_SIR ,lw=2, title="SIR Infected Curve", xlabel="Time Steps", ylabel="Number of Infected Nodes", label="Activity-Driven", legend=:bottomright, color=:darkorange)
savefig("peak_SIR.png")
## SEIR
epidemic_type = :SEIR
σ  = 0.04
results_SEIR, avg_a_SEIR, squared_a_SEIR = repeat_temporal_simulation(N, m, T, η, ε, γ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected_matrix_SEIR = reduce(hcat, [r[:I] for r in results_SEIR])  
avg_infected_SEIR = mean(infected_matrix_SEIR, dims=2)             
std_infected_SEIR = std(infected_matrix_SEIR, dims=2)

τ=20.0
results20_SEIR, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected20_matrix_SEIR = reduce(hcat, [r[:I] for r in results20_SEIR])  
avg_infected20_SEIR = mean(infected20_matrix_SEIR, dims=2)            
std_infected20_SEIR = std(infected20_matrix_SEIR, dims=2)

τ=40.0
results40_SEIR, _, _ = repeat_aggregated_simulation(N, m, T, η, ε, γ, τ, β, μ, σ, initial_infected, num_repeats, epidemic_type)
infected40_matrix_SEIR = reduce(hcat, [r[:I] for r in results40_SEIR])  
avg_infected40_SEIR = mean(infected40_matrix_SEIR, dims=2)             
std_infected40_SEIR = std(infected40_matrix_SEIR, dims=2)

plot(avg_infected40_SEIR, ribbon=std_infected40_SEIR, lw=2, label="Aggregated T=40", color=:dodgerblue)   
plot!(avg_infected20_SEIR, ribbon=std_infected20_SEIR, lw=2, label="Aggregated T=20", color=:crimson) 
plot!(avg_infected_SEIR, ribbon=std_infected_SEIR/2 ,lw=2, title="SEIR Infected Curve", xlabel="Time Steps", ylabel="Number of Infected Nodes", label="Activity-Driven", legend=:bottomright, color=:darkorange)

savefig("peak_SEIR.png")
