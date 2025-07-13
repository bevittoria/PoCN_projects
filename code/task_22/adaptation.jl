using Random, StatsBase, Plots

# Struct to represent an edge at a time step
struct TemporalEdge
    from::Int
    to::Int
    time::Int
end



function sir_each_time(edge_dict::Dict{Int, Vector{Tuple{Int, Int}}}, t::Int, β::Float64, μ::Float64, states::Vector{Symbol}, N::Int)
    """    Simulates one time step of the SIR model on a temporal network.  
    # Arguments
    - edge_dict: Dictionary mapping time steps to edges
    - t: Current time step
    - β: Infection rate
    - μ: Recovery rate
    - states: Current states of nodes (S, I, R)
    - N: Total number of nodes
    # Returns
    - states: Updated states of nodes after the time step
    """
    current_edges = get(edge_dict, t, [])
    new_infected = Set{Int}()
    new_recovered = Set{Int}()

    # Infection step
    for (i, j) in current_edges
        if states[i] == :I && states[j] == :S && rand() < β
            push!(new_infected, j)
        elseif states[j] == :I && states[i] == :S && rand() < β
            push!(new_infected, i)
        end
    end

        # Recovery step
    for i in 1:N
        if states[i] == :I && rand() < μ
            push!(new_recovered, i)
        end
    end

    # Apply updates
    for i in new_infected
        states[i] = :I
    end
    for i in new_recovered
        states[i] = :R
    end

    return states
end
function sample_bounded_powerlaw(N::Int, γ::Float64, xmin::Float64, xmax::Float64)
    @assert γ > 1 "γ must be > 1 for normalizable PDF"
    u = rand(N)
    a = xmin^(1 - γ)
    b = xmax^(1 - γ)
    return (a .+ u .* (b - a)).^(1 / (1 - γ))
end

function simulate_activity_network_epidemic(
    N::Int, 
    m0::Int, 
    T::Int, 
    η::Float64, 
    ε::Float64, 
    β::Float64, 
    μ::Float64, 
    γ::Float64,
    initial_infected::Int; 
    feedback::Bool = false, 
    α::Float64 = 1.0
)
    # Activity rates
    x_vals = sample_bounded_powerlaw(N, γ, ε, 1.0)
    #x_vals = [max(rand(), ε) for _ in 1:N]
    a_vals = η .* x_vals

    # States
    states = fill(:S, N)
    infected_nodes = Set(rand(1:N, initial_infected))
    for i in infected_nodes
        states[i] = :I
    end

    infected_counts = Int[]
    recovered_counts = Int[]
    
    temporal_edges = TemporalEdge[]
    integrated_edges = Set{Tuple{Int, Int}}()
    edge_dict = Dict{Int, Vector{Tuple{Int, Int}}}()


    for t in 1:T
        I_frac = count(==(:I), states) / N
        m = feedback ? max(1, round(Int, m0 * (1 - exp(-α * I_frac)))) : m0
        #@show m, I_frac  # Debugging output
        #println("Time step: ", t, " | m: ", m, " | Infected fraction: ", I_frac)
        
        edge_dict[t] = []

        for i in 1:N
            if rand() < a_vals[i]
                targets = rand(setdiff(1:N, [i]), m)
                for j in targets
                    push!(temporal_edges, TemporalEdge(i, j, t))
                    push!(integrated_edges, (min(i, j), max(i, j)))
                    push!(edge_dict[t], (i, j))
                end
            end
        end

        # === SIR dynamics ===
        states = sir_each_time(edge_dict, t, β, μ, states, N)

        push!(infected_counts, count(==(:I), states))
        push!(recovered_counts, count(==(:R), states))
    end

    return infected_counts, recovered_counts, temporal_edges, edge_dict, states
end


function plot_sir(
    infected_counts::Vector{Float64},
    recovered_counts::Vector{Float64},
    susceptible_counts::Vector{Float64},
    T::Int
)
    plot(1:T, susceptible_counts, label="Susceptible", lw=2)
    plot!(1:T, infected_counts, label="Infected", lw=2, color=:red)
    plot!(1:T, recovered_counts, label="Recovered", lw=2, color=:green)
    xlabel!("Time")
    ylabel!("Number of Nodes")
    title!("SIR on Activity-Driven Temporal Network")
end

using Statistics


function repeated_simulation(
    N::Int, 
    m0::Int, 
    T::Int, 
    η::Float64, 
    ε::Float64, 
    β::Float64, 
    μ::Float64,
    γ::Float64, 
    initial_infected::Int; 
    feedback::Bool = false, 
    α::Float64 = 1.0,
    num_repeats::Int = 10
)
    results = []
    for _ in 1:num_repeats
        infected_counts, recovered_counts, temporal_edges, integrated_edges = simulate_activity_network_epidemic(
            N, m0, T, η, ε, β, μ,γ, initial_infected; 
            feedback = feedback,
            α = α
        )
        push!(results, (infected_counts, recovered_counts))
    end
    # i want to return the average infected, recovered, susceptible counts at each time 
    # step across all simulations
    # Filter out runs where the epidemic did not start (less than 20 recovered at end)
    filtered_results = filter(r -> last(r[2]) > 20, results)
    if isempty(filtered_results)
        filtered_results = results  # fallback to all if none pass
    end
    results = filtered_results
    avg_infected = [mean([infected_counts[t] for (infected_counts, _) in results]) for t in 1:T]
    std_infected = [std([infected_counts[t] for (infected_counts, _) in results]) for t in 1:T]
    avg_recovered = [mean([recovered_counts[t] for (_, recovered_counts) in results]) for t in 1:T]
    avg_susceptible = N .- avg_infected .- avg_recovered

    return avg_infected, avg_recovered, avg_susceptible, results, std_infected
end


Random.seed!(123)


N= 1000
m0 = 20
T = 1000
η = 5.0
ε = 0.001
β = 0.04
μ = 0.01

avg_infected, avg_recovered, avg_susceptible, _, std_infected = repeated_simulation(
    N, m0, T, η, ε, β, μ,γ, 10; 
    feedback = false,
    num_repeats = 10
)

avg_infected_fb, avg_recovered_fb, avg_susceptible_fb, _, std_infected_fb = repeated_simulation(
    N, m0, T, η, ε, β, μ,γ, 10; 
    feedback = true,
    α = 18.0,
    num_repeats = 100
)

avg_infected_fb30, avg_recovered_fb, avg_susceptible_fb, _, std_infected_fb30 = repeated_simulation(
    N, m0, T, η, ε, β, μ,γ, 10; 
    feedback = true,
    α = 30.0,
    num_repeats = 100
)

avg_infected_fb10, avg_recovered_fb, avg_susceptible_fb, _, std_infected_fb10 = repeated_simulation(
    N, m0, T, η, ε, β, μ,γ, 10; 
    feedback = true,
    α = 10.0,
    num_repeats = 100
)

plot(avg_infected,ribbon = std_infected, label="No Adaptation", lw=2)
plot!(avg_infected_fb30, ribbon = std_infected_fb30./2, label="With Adaptation (α=30)", lw=2)
plot!(avg_infected_fb, ribbon = std_infected_fb./2, label="With Adaptation (α=18)", lw=2)
plot!(avg_infected_fb10, ribbon = std_infected_fb10./2, label="With Adaptation (α=10)", lw=2)
xlabel!("Time")
ylabel!("Infected count")
plot!(legend=:topright)



infected,_,_,edge_dict = simulate_activity_network_epidemic(
    N, m0, T, η, ε, β, μ, 1; 
    feedback = true,
    α = 1.0
)

edges_per_time = [length(get(edge_dict, t, [])) for t in 1:T]
plot(1:T, infected, label="Infected")
plot!(1:T, edges_per_time, label="Edges", ylabel="Number", xlabel="Time", title="Edges vs Infected Over Time")