
#-------------------------------------------------------------------------------
# Initialization
#-------------------------------------------------------------------------------

# Threads.nthreads()
# addprocs()

###### NOTE: These packages might be needed, uncomment them accordingly
###### to the use:

# Pkg.add("Gurobi")
# Pkg.add("StatPlots")
# Pkg.add("JuMP")
# Pkg.add("GR")
# Pkg.add("Distances")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("StatsBase")

###### NOTE: Sometimes "Compat" package hampers the other packages to be
###### correctly installed
# Pkg.rm("Compat")

###### NOTE: Checking out if the packages are in place
# Pkg.build()
# Pkg.status()
# Pkg.update()

###### NOTE: Standard procedure
# using Clustering
# using GR
# using IndexedTables
# using StatPlots
using Distributed
using Gurobi
using JuMP
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using Statistics
@everywhere using Gurobi
@everywhere using JuMP
@everywhere using DataFrames
@everywhere using Distances
@everywhere using CSV

function wogrin2D_P(k::Int64,
    weight::Int64,
    a₁::Vector{Float64},
    a₂::Vector{Float64},
    parallel::Bool)
    ## Clustering modelling brought by the article named "A New Approach to
    ## Model Load Levels in Electric Power Systems With High Renewable
    ## Penetration" by Sonja Wogrin et al. (2014)

    # x: Aux Array for concatenating the DataFrame used
    # Data: DataFrame used
    # k: number of classes
    # weight: weight used to account clusters' distances
    #                                                   # Weight = 1: dif_mean
    #                                                   # Weight = 2: dif_min
    #                                                   # Weight = 3: dif_max
    # n: Number of points
    # m: Number of dimensions
    # dist: distances from k-centers (n,k)
    # class: Array of classes (n)
    # weights_ar: Dataframe of costs
    # first: Array of first indexes
    # k_cent: Array of first centroids
    # costs: Array of weighted distances
    # total_cost: Auxiliar var.
    # total_cost_new: Auxiliar var.
    # δ: Auxiliar paramenter

    ## Initial definitions
    n = length(a₁)                              # Number of points
    m = 2                                       # Number of dimensions
    dist = SharedArray(zeros(n,k))              # Array of distances (n,k)
    costs = SharedArray(zeros(n,k))             # Array of costs (n,k)
    first_s = sample(1:n,k,replace=false)       # Sampling the first centroids
    k_cent = hcat(a₁,a₂)[first_s,:]             # First centroids
    class = zeros(Int64, n)                     # Array of classes (n)
    weights_ar = [zeros(n) zeros(n) zeros(n)]   # Costs array
    Data = hcat(a₁, a₂)                         # Completing the Data with costs
    c_use = zeros(Bool,k)                       # Array with the using status
    total_cost = 0                              # Starting the auxiliar var.
    total_cost_new = 1                          # Starting the auxiliar var.
    δ = 1e-10                                   # Aux. paramenter
    tol = 1e-12                                 # Tolerance
    cont = 0                                    # Counter

    # First cost settings (Only for 2D)
    dif_mean = mean(Data[:,1]-Data[:,2])
    dif_min = minimum(Data[:,1]-Data[:,2])
    dif_max = maximum(Data[:,1]-Data[:,2])

    # Defining weights
    for i in 1:n
        weights_ar[i,1] = abs(Data[i,1]-Data[i,2]-dif_mean+δ)
        weights_ar[i,2] = abs(Data[i,1]-Data[i,2]-dif_min+δ)
        weights_ar[i,3] = abs(Data[i,1]-Data[i,2]-dif_max+δ)
    end

    if parallel

        #Assigning classes (parallel)
        while total_cost != total_cost_new
            cont = cont + 1
            total_cost = total_cost_new

            # Defining distances
            for i in 1:n
                @sync @distributed for j in 1:k
                    dist[i,j] = evaluate(Euclidean(),
                    Data[i,1:2],k_cent[j,:])
                end
            end

            # Defining costs
            for i in 1:n
                @sync @distributed for j in 1:k
                    costs[i,j] = weights_ar[i,weight]*dist[i,j]
                end
            end

            # Defining classes
            for i in 1:n
                class[i] = findmin(costs[i,:])[2]
            end

            ## Classification used / unused clusters
            for i in 1:k
                if length(class[class[:] .== i]) == 0
                    c_use[i] = false
                else
                    c_use[i] = true
                end
            end

            # Update classes (mean for those used and 0 for the unused)
            for j in 1:k
                if c_use[j]
                    k_cent[j,1] = mean(Data[class[:] .== j,:][:,1])
                    k_cent[j,2] = mean(Data[class[:] .== j,:][:,2])
                else
                    k_cent[j,1] = mean(Data[:,:][:,1])
                    k_cent[j,2] = mean(Data[:,:][:,2])
                end
            end

            total_cost_new = sum(costs[i,class[i]] for i in 1:n)
        end

    else

        #Assigning classes (non_parallel)
        while total_cost != total_cost_new
            cont = cont + 1
            total_cost = total_cost_new

            # Defining distances
            for i in 1:n
                for j in 1:k
                    dist[i,j] = evaluate(Euclidean(),
                    Data[i,1:m],k_cent[j,:])
                end
            end

            # Defining costs
            for i in 1:n
                for j in 1:k
                    costs[i,j] = weights_ar[i,weight]*dist[i,j]
                end
            end

            # Defining classes
            for i in 1:n
                class[i] = findmin(costs[i,:])[2]
            end

            ## Classification used / unused clusters
            for i in 1:k
                if length(class[class[:] .== i]) == 0
                    c_use[i] = false
                else
                    c_use[i] = true
                end
            end

            # Update classes (mean for those used and 0 for the unused)
            for j in 1:k
                if c_use[j]
                    k_cent[j,1] = mean(Data[class[:] .== j,:][:,1])
                    k_cent[j,2] = mean(Data[class[:] .== j,:][:,2])
                else
                    k_cent[j,1] = mean(Data[:,:][:,1])
                    k_cent[j,2] = mean(Data[:,:][:,2])
                end
            end

            total_cost_new = sum(costs[i,class[i]] for i in 1:n)
        end
    end


    Size_Cluster = zeros(k)

    for i in 1:k
        if c_use[i]
            Size_Cluster[i] = length(Data[class[:] .== i,:][1])
        else
            Size_Cluster[i] = 0
        end
    end

    #-------------------------------------------------------------------------------
    ### Transition Matrix
    #-------------------------------------------------------------------------------

    N_transit = zeros(k,k)
    from = 0
    to = 0
    for i in 1:n-1
    	from = class[i]
    	to = class[i+1]
    	N_transit[from,to] = N_transit[from,to] + 1
    end

    return Data, class, k_cent, N_transit, first_s, c_use
end

function wogrin2D(k::Int64,
    weight::Int64,
    a₁::Vector{Float64},
    a₂::Vector{Float64})

    ## Clustering modelling brought by the article named "A New Approach to
    ## Model Load Levels in Electric Power Systems With High Renewable
    ## Penetration" by Sonja Wogrin et al. (2014)

    # x: Aux Array for concatenating the DataFrame used
    # Data: DataFrame used
    # k: number of classes
    # weight: weight used to account clusters' distances
    #                                                   # Weight = 1: dif_mean
    #                                                   # Weight = 2: dif_min
    #                                                   # Weight = 3: dif_max
    # n: Number of points
    # m: Number of dimensions
    # dist: distances from k-centers (n,k)
    # class: Array of classes (n)
    # weights_ar: Dataframe of costs
    # first: Array of first indexes
    # k_cent: Array of first centroids
    # costs: Array of weighted distances
    # total_cost: Auxiliar var.
    # total_cost_new: Auxiliar var.
    # δ: Auxiliar paramenter

    ## Initial definitions
    
    n = length(a₁)                              # Number of points
    m = 2                                       # Number of dimensions
    dist = zeros(n,k)                           # Array of distances (n,k)
    costs = zeros(n,k)                          # Array of costs (n,k)
    first_s = sample(1:n,k,replace=false)       # Sampling the first centroids
    k_cent = [a₁ a₂][first_s,:]                 # First centroids
    class = zeros(Int64, n)                     # Array of classes (n)
    weights_ar = [zeros(n) zeros(n) zeros(n)]   # Costs array
    Data = [a₁ a₂]                              # Completing the Data with costs
    c_use = zeros(Bool,k)                       # Array with the using status
    total_cost = 0                              # Starting the auxiliar var.
    total_cost_new = 1                          # Starting the auxiliar var.
    δ = 1e-10                                   # Aux. paramenter
    tol = 1e10                                  # Tolerance
    cont = 0                                    # Counter

    # First cost settings (Only for 2D)
    dif_mean = mean(Data[:,1]-Data[:,2])
    dif_min = minimum(Data[:,1]-Data[:,2])
    dif_max = maximum(Data[:,1]-Data[:,2])

    # Defining weights
    for i in 1:n
        weights_ar[i,1] = abs(Data[i,1]-Data[i,2]-dif_mean+δ)
        weights_ar[i,2] = abs(Data[i,1]-Data[i,2]-dif_min+δ)
        weights_ar[i,3] = abs(Data[i,1]-Data[i,2]-dif_max+δ)
    end

    #Assigning classes (non_parallel)
    while total_cost != total_cost_new
        cont = cont + 1
        if cont > tol
            println("\n Does not converge! \n")
        end

        total_cost = total_cost_new

        # Defining distances
        for i in 1:n
            for j in 1:k
                dist[i,j] = evaluate(Euclidean(),
                Data[i,1:m],k_cent[j,:])
            end
        end

        # Defining costs
        for i in 1:n
            for j in 1:k
                costs[i,j] = weights_ar[i,weight]*dist[i,j]
            end
        end

        # Defining classes
        for i in 1:n
            class[i] = findmin(costs[i,:])[2]
        end

        ## Classification used / unused clusters
        for i in 1:k
            if length(class[class[:] .== i]) == 0
                c_use[i] = false
            else
                c_use[i] = true
            end
        end

        # Update classes (mean for those used and 0 for the unused)
        for j in 1:k
            if c_use[j]
                k_cent[j,1] = mean(a₁[class[:] .== j])
                k_cent[j,2] = mean(a₂[class[:] .== j])
            else
                k_cent[j,1] = mean(a₁)
                k_cent[j,2] = mean(a₂)
            end
        end

        total_cost_new = sum(costs[i,class[i]] for i in 1:n)
    end

    Size_Cluster = zeros(k)

    for i in 1:k
        if c_use[i]
            Size_Cluster[i] = length(Data[class[:] .== i,:][:,1])
        else
            Size_Cluster[i] = 0
        end
    end

    #-------------------------------------------------------------------------------
    ### Transition Matrix
    #-------------------------------------------------------------------------------

    N_transit = zeros(k,k)
    from = 0
    to = 0
    for i in 1:n-1
    	from = class[i]
    	to = class[i+1]
    	N_transit[from,to] = N_transit[from,to] + 1
    end

    return Data, class, k_cent, N_transit, first_s, c_use
end

function pineda1D(k::Int64,
	a₁::Vector{Float64})

	## Clustering modelling brought by the article named "Chronological
    ## Time-Period Clustering for Optimal Capacity Expansion Planning
    ## With Storage" by Salvador Pineda & Juan Morales (2018)

	### Initialization
	n = length(a₁)						# Size of Data
	class = collect(1:n)				# Array of classes (n)
	k_cent = copy(a₁)					# Centroids array
	dist = zeros(n)						# Array of distances (n)
	l = length(a₁)						# Number of clusters used
	counter = 0							# Control the number of iterations
	tol = 10^2							# Convergence tolerance

	while l > k
		counter = counter + 1
		if counter > tol
			println("Doesn't converge! \n")
			return
		end
		#Distances matrix designation
		for i in 1:(l-1)
			dist[i] = 2 * length(class[class[:] .== i]) *
					   	  length(class[class[:] .== i+1]) /
						  (length(class[class[:] .== i]) +
						  length(class[class[:] .== i+1])) *
						  abs(k_cent[i] - k_cent[i+1])
		end

		# Updating the last value for the max
		dist[l] = maximum(dist[1:l])

		# min: find the minimum distances
		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

		marker = zeros(n)					# To mark whenever the minimum occurs

		##Mark as 1 whenever a minimum happens
		for i in 1:(n-1)
			if (class[i] in min_dist) && (class[i] != class[i+1])
				marker[i+1] = 1
			end
		end

		##Accounts the cumulative change in clusters order
		c_change = 0						# Change in the number of clusters
		for i in 1:n
			c_change = c_change - marker[i]
			marker[i] = c_change
		end

		##Update values
		class[:] = class[:] + marker[:]

		##Update centroids
		for i in 1:l
			k_cent[i] = mean(a₁[class[:] .== i])
		end

		l = l + Int(c_change)
	end

	return class, k_cent[1:l]
end

function pineda2D(k::Int64,
	a₁::Vector{Float64},
    a₂::Vector{Float64})

	## Clustering modelling inpired by the article named "Chronological
    ## Time-Period Clustering for Optimal Capacity Expansion Planning
    ## With Storage" by Salvador Pineda & Juan Morales (2018)

    if length(a₁) != length(a₂)
        println("Different sizes input")
        return
    end

	### Initialization
	n = length(a₁)						# Size of Data
	class = collect(1:n)				# Array of classes (n)
	k_cent = copy([a₁ a₂])              # Centroids array
	dist = zeros(n)						# Array of distances (n)
	l = length(a₁)						# Number of clusters used
	counter = 0							# Control the number of iterations

	while l > k
		#Distances matrix designation
		for i in 1:(l-1)
			dist[i] = 2 * length(class[class[:] .== i]) *
					   	  length(class[class[:] .== i+1]) /
						  (length(class[class[:] .== i]) +
						  length(class[class[:] .== i+1])) *
						  evaluate(Euclidean(),k_cent[i], k_cent[i+1])
		end

		# Updating the last value for the max
		dist[l] = maximum(dist[1:l])

		# min: find the minimum distances
		min_dist = findall(x -> x == minimum(dist[1:l]),dist[1:l])

		marker = zeros(n)					# To mark whenever the minimum occurs

		##Mark as 1 whenever a minimum happens
		for i in 1:(n-1)
			if (class[i] in min_dist) && (class[i] != class[i+1])
				marker[i+1] = 1
			end
		end

		##Accounts the cumulative change in clusters order
		c_change = 0						# Change in the number of clusters
		for i in 1:n
			c_change = c_change - marker[i]
			marker[i] = c_change
		end

		##Update values
		class[:] = class[:] + marker[:]

		##Update centroids
		for i in 1:l
			k_cent[i,1] = mean(a₁[class[:] .== i])
            k_cent[i,2] = mean(a₂[class[:] .== i])
		end

		l = l + Int(c_change)
	end

	return class, k_cent[1:l,:]
end

function wassdist2D(from::Array{Float64},
    to::Array{Float64})

    for d in 1:2
        if sum(from[:,d]) != sum(to[:,d])
            println("\n Arrays with differente mass in dimension $d \n")
            # return
        end
    end
    ##__________________________________________________________________________
    ##### Probabilistic Distance based on Wasserstein distance
    ##__________________________________________________________________________

    ### Initialization
    n = length(from[:,1])

    #-------------------------------------------------------------------------------
    # Optimisation model formulation
    #-------------------------------------------------------------------------------

    model = Model(solver = GurobiSolver())

    #-------------------------------------------------------------------------------
    ##Sets
    #-------------------------------------------------------------------------------

    I = collect(1:n)            # Original distribution
    J = collect(1:n)            # Adapted distribution
    K = collect(1:2)    		# Number of dimensions

    #-------------------------------------------------------------------------------
    ##Parameters
    #-------------------------------------------------------------------------------

    P_i = copy(from)
    P_j = copy(to)

    #-------------------------------------------------------------------------------
    ##Variables
    #-------------------------------------------------------------------------------

    @variable(model,eta[1:I[end], 1:J[end], 1:K[end]] >= 0)

    #-------------------------------------------------------------------------------
    ##Constraints (Model C)
    #-------------------------------------------------------------------------------

    @constraint(model,[i in I, k in K],  P_i[i,k] == sum(eta[i,j,k] for j in J))
    @constraint(model,[j in J, k in K],  P_j[j,k] == sum(eta[i,j,k] for i in I))

    #-------------------------------------------------------------------------------
    ##Objective Function
    #-------------------------------------------------------------------------------

    @objective(model, Min, sum(abs(i-j)*eta[i,j,k] for i in I, j in J, k in K))

    status = solve(model)

    #Get the values of costs against number of clusters
    out = getobjectivevalue(model)

    return out
end
