import numpy as np
import numpy.matlib
import random

def nsga_2(pop,gen,M,V,min_range,max_range, sys_param):
    
    # % Both the input arguments need to of integer data type
    if pop == 0 or gen == 0 :
        raise Exception('Both input arguments pop and gen should be integer datatype')
    
    # % Minimum population size has to be 20 individuals
    if pop < 20 :
        raise Exception('Minimum population for running this function is 20') 
    
    if gen < 5 :
        raise Exception('Minimum number of generations is 5') 
    
    # % Make sure pop and gen are integers
    pop = round(pop)
    gen = round(gen)
    
    # % Initialize the population
    # % Population is initialized with random values which are within the
    # % specified range. Each chromosome consists of the decision variables. Also
    # % the value of the objective functions, rank and crowding distance
    # % information is also added to the chromosome vector but only the elements
    # % of the vector which has the decision variables are operated upon to
    # % perform the genetic operations like corssover and mutation.
    chromosome = initialize_variables(pop, M, V, min_range, max_range, sys_param)
    chromosome_0 = chromosome # save initialization
    
    # % Sort the initialized population
    # % Sort the population using non-domination-sort. This returns two columns
    # % for each individual which are the rank and the crowding distance
    # % corresponding to their position in the front they belong. At this stage
    # % the rank and the crowding distance for each chromosome is added to the
    # % chromosome vector for easy of computation.
    chromosome = non_domination_sort_fast(chromosome, M, V) 
    # print('chromosome',chromosome)
    
    # % Start the evolution process
    # % The following are performed in each generation
    # % * Select the parents which are fit for reproduction
    # % * Perfrom crossover and Mutation operator on the selected parents
    # % * Perform Selection from the parents and the offsprings
    # % * Replace the unfit individuals with the fit individuals to maintain a
    # %   constant population size.
    f_intermediate = np.ones(shape = (gen,M*pop)) 
    
    # % Select the parents
    # % Parents are selected for reproduction to generate offspring. The
    # % original NSGA-II uses a binary tournament selection based on the
    # % crowded-comparision operator. The arguments are 
    # % pool - size of the mating pool. It is common to have this to be half the
    # %        population size.
    # % tour - Tournament size. Original NSGA-II uses a binary tournament
    # %        selection, but to see the effect of tournament size this is kept
    # %        arbitary, to be choosen by the user.
    pool = round(pop/2)
    tour = 2
    # % Selection process
    # % A binary tournament selection is employed in NSGA-II. In a binary
    # % tournament selection process two individuals are selected at random
    # % and their fitness is compared. The individual with better fitness is
    # % selected as a parent. Tournament selection is carried out until the
    # % pool size is filled. Basically a pool size is the number of parents
    # % to be selected. The input arguments to the function
    # % tournament_selection are chromosome, pool, tour. The function uses
    # % only the information from last two elements in the chromosome vector.
    # % The last element has the crowding distance information while the
    # % penultimate element has the rank information. Selection is based on
    # % rank and if individuals with same rank are encountered, crowding
    # % distance is compared. A lower rank and higher crowding distance is
    # % the selection criteria.
    parent_chromosome = tournament_selection(chromosome, pool, tour)

    # % Perform crossover and Mutation operator
    # % The original NSGA-II algorithm uses Simulated Binary Crossover (SBX) and
    # % Polynomial  mutation. Crossover probability pc = 0.9 and mutation
    # % probability is pm = 1/n, where n is the number of decision variables.
    # % Both real-coded GA and binary-coded GA are implemented in the original
    # % algorithm, while in this program only the real-coded GA is considered.
    # % The distribution indeices for crossover and mutation operators as mu = 20
    # % and mum = 20 respectively.
    mu = 20
    mum = 20
    offspring_chromosome = genetic_operator(parent_chromosome, M, V, mu, mum, min_range, max_range, sys_param)

    # % Intermediate population
    # % Intermediate population is the combined population of parents and
    # % offsprings of the current generation. The population size is two
    # % times the initial population.
    
    main_pop = len(chromosome)
    offspring_pop = len(offspring_chromosome)
    # % intermediate_chromosome is a concatenation of current population and
    # % the offspring population.
    intermediate_chromosome = np.vstack((chromosome , np.c_[offspring_chromosome, np.zeros(shape=(len(offspring_chromosome),2))]))

    # % Non-domination-sort of intermediate population
    # % The intermediate population is sorted again based on non-domination sort
    # % before the replacement operator is performed on the intermediate pop.
    intermediate_chromosome = non_domination_sort_fast(intermediate_chromosome, M, V)
    # % Perform Selection
    # % Once the intermediate population is sorted only the best solution is
    # % selected based on it rank and crowding distance. Each front is filled in
    # % asc ing order until the addition of population size is reached. The
    # % last front is included in the population based on the individuals with
    # % least crowding distance
    chromosome = replace_chromosome(intermediate_chromosome, M, V, pop)
        
    return chromosome_0, chromosome



def initialize_variables(N, M, V, min_range, max_range, sys_param):
    import emodps
    
    K = M + V        
    # % K is the total number of array elements. For ease of computation decision
    # % variables and objective functions are concatenated to form a single
    # % array. For crossover and mutation only the decision variables are used
    # % while for selection, only the objective variable are utilized.
    
    # % Initialize the decision variables based on the minimum and maximum
    # % possible values. V is the number of decision variable. A random
    # % number is picked between the minimum and maximum possible values for
    # % each decision variable.
    f = np.nan*np.ones(shape=(N,K))
    
    f[:,0:V] = np.matlib.repmat( min_range, N, 1 ) + np.matlib.repmat( np.array(max_range) - np.array(min_range), N, 1 )*np.random.rand( N, V ) 
    
    # % For ease of computation and handling data the chromosome also has the
    # % value of the objective function concatenated. The elements
    # % V + 1 to K has the objective function valued. 
    # % The function evaluate_objective takes one chromosome at a time,
    # % infact only the decision variables are passed to the function along
    # % with information about the number of objective functions which are
    # % processed and returns the value for the objective functions. These
    # % values are now stored at the last M columns of the chromosome itself.
    for i in range (N):
        f[i,V:] = emodps.evaluate_objective(f[i,:V],M,V, sys_param)
    return f

# %%
def non_domination_sort_fast(x, M, V):
    values1 = x[:,V] #Objective 1
    values2 = x[:,V+1] #Objective 2
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[] #individuals dominated by p
        n[p]=0 #p got dominated by how many individuals
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append (q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append (p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append (q)
        i = i+1
        front.append (Q)

    del front[len(front)-1]
    
    x = np.c_[x, rank] #Add rank as the next column 
    
    #Sort based on front
    sortby_front = x[np.argsort(x[:,-1])]
    
    y_ = np.nan*np.ones(shape=(1,M+V+2))
    # Extract front by front
    for F in range( len(front) ):        
        y = sortby_front [sortby_front [:,-1] == F]
    
        distance = [0 for i in range(len(y))]
        y = np.c_[y, distance, distance]
        
        # Crowding distance
        for i in range(M):
            index_of_objectives = np.argsort(y[:,V+i])
            sorted_based_on_objective = y[index_of_objectives]
                            
            f_max = sorted_based_on_objective[-1, V + i]
            f_min = sorted_based_on_objective[0, V + i]
            
            y[index_of_objectives[-1] , M + V + 1 + i] = np.Infinity
            y[index_of_objectives[0], M + V + 1 + i] = np.Infinity
            
            for j in range ( 2, len(index_of_objectives) - 1):
                next_obj  = sorted_based_on_objective[j + 1,V + i]
                previous_obj  = sorted_based_on_objective[j - 1,V + i]
                
                if (f_max - f_min == 0):
                    y[index_of_objectives[j],M + V + 1 + i] = np.Infinity
                else:
                    y[index_of_objectives[j],M + V + 1 + i] = (next_obj - previous_obj)/(f_max - f_min)
                    
        distance = []
        distance = y[:,-1] + y[:,-2]
        y = y[:, : M + V + 1]    
        y = np.c_[y, distance]
    
        y_ = np.vstack((y_,y))
        y_ = y_[~np.isnan(y_).any(axis=1)]
    
    return y_


# %%
def non_domination_sort_mod(x, M, V):
    # % This function sort the current popultion based on non-domination. All the
    # % individuals in the first front are given a rank of 1, the second front
    # % individuals are assigned rank 2 and so on. After assigning the rank the
    # % crowding in each front is calculated.
    # %  Redistribution and use in source and binary forms, with or without 
    # %  modification, are permitted provided that the following conditions are 
    # %  met:
    # %
    # %     * Redistributions of source code must retain the above copyright 
    # %       notice, this list of conditions and the following disclaimer.
    # %     * Redistributions in binary form must reproduce the above copyright 
    # %       notice, this list of conditions and the following disclaimer in 
    # %       the documentation and/or other materials provided with the distribution
    # %      
    # %  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
    # %  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
    # %  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
    # %  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
    # %  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
    # %  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
    # %  SUBSTITUTE GOODS OR SERVICES  LOSS OF USE, DATA, OR PROFITS  OR BUSINESS 
    # %  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
    # %  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
    # %  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
    # %  POSSIBILITY OF SUCH DAMAGE.
    
    N, m = x.shape
    del m
    
    # % Initialize the front number to 1.
    front = 1 
    
    # % There is nothing to this assignment, used only to manipulate easily in
    # % MATLAB.
    front = [[]] #F[front].f = []
    individual = []
    
    # %% Non-Dominated sort. 
    # % The initialized population is sorted based on non-domination. The fast
    # % sort algorithm [1] is described as below for each
    
    # % • for each individual p in main population P do the following
    # %   – Initialize Sp = []. This set would contain all the individuals that is
    # %     being dominated by p.
    # %   – Initialize np = 0. This would be the number of individuals that domi-
    # %     nate p.
    # %   – for each individual q in P
    # %       * if p dominated q then
    # %           · add q to the set Sp i.e. Sp = Sp ? {q}
    # %       * else if q dominates p then
    # %           · increment the domination counter for p i.e. np = np + 1
    # %   – if np = 0 i.e. no individuals dominate p then p belongs to the first
    # %     front  Set rank of individual p to one i.e prank = 1. Update the first
    # %     front set by adding p to front one i.e F1 = F1 ? {p}
    # % • This is carried out for all the individuals in main population P.
    # % • Initialize the front counter to one. i = 1
    # % • following is carried out while the ith front is nonempty i.e. Fi != []
    # %   – Q = []. The set for storing the individuals for (i + 1)th front.
    # %   – for each individual p in front Fi
    # %       * for each individual q in Sp (Sp is the set of individuals
    # %         dominated by p)
    # %           · nq = nq?1, decrement the domination count for individual q.
    # %           · if nq = 0 then none of the individuals in the subsequent
    # %             fronts would dominate q. Hence set qrank = i + 1. Update
    # %             the set Q with individual q i.e. Q = Q ? q.
    # %   – Increment the front counter by one.
    # %   – Now the set Q is the next front and hence Fi = Q.
    # %
    # % This algorithm is better than the original NSGA ([2]) since it utilize
    # % the informatoion about the set that an individual dominate (Sp) and
    # % number of individuals that dominate the individual (np).
    
    for i in range(N):
        # % Number of individuals that dominate this individual
        individual[i].n = 0
        # % Individuals which this individual dominate
        individual[i].p = []
        for j in range(N):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for k in range(M) :
                if (x[i,V + k] < x[j,V + k]) :
                    dom_less = dom_less + 1
                elif (x[i,V + k] == x[j,V + k]) :
                    dom_equal = dom_equal + 1
                else :
                    dom_more = dom_more + 1
                
            
            if dom_less == 0 and dom_equal != M :
                individual(i).n = individual(i).n + 1
            elif dom_more == 0 and dom_equal != M :
                individual(i).p = [individual(i).p, j]
            
           
        if individual(i).n == 0 :
            x[i,M + V + 1] = 1
            F(front).f = [F[front].f, i]
            
    # % Find the subsequent fronts
    while len(F[front].f) != 0 : #!isempty(F(front).f):
       Q = []
       for i in len(F(front).f):
           if len(individual(F(front).f(i)).p) == 0:
            	for j in len(individual(F(front).f(i)).p) :
                	individual(individual(F(front).f(i)).p [j]).n = individual(individual(F(front).f(i)).p [j]).n - 1 
            	   	if individual(individual(F(front).f(i)).p [j]).n == 0 :
                           x[individual(F(front).f(i)).p [j],M + V + 1] = front + 1 
                           Q = [Q, individual(F(front).f(i)).p, j]
                        
       front =  front + 1 
       F(front).f = Q 
    
    
    temp,index_of_fronts = np.sort(x[:,M + V + 1]) 
    for i in range ( len(index_of_fronts) ) :
        sorted_based_on_front[i,:] = x[index_of_fronts[i],:]
    
    current_index = 0
    
    # %% Crowding distance
    # %The crowing distance is calculated as below
    # % • For each front Fi, n is the number of individuals.
    # %   – initialize the distance to be zero for all the individuals i.e. Fi(dj ) = 0,
    # %     where j corresponds to the jth individual in front Fi.
    # %   – for each objective function m
    # %       * Sort the individuals in front Fi based on objective m i.e. I =
    # %         sort(Fi,m).
    # %       * Assign infinite distance to boundary values for each individual
    # %         in Fi i.e. I(d1) = ? and I(dn) = ?
    # %       * for k = 2 to (n ? 1)
    # %           · I(dk) = I(dk) + (I(k + 1).m ? I(k ? 1).m)/fmax(m) - fmin(m)
    # %           · I(k).m is the value of the mth objective function of the kth
    # %             individual in I
    
    # % Find the crowding distance for each individual in each front
    for f in range (len(front)):
    # %    objective = [] 
        distance = 0 
        y = [] 
        previous_index = current_index + 1 
        for i in range( len(F(front).f) ) :
            y[i,:] = sorted_based_on_front[current_index + i,:] 
        
        current_index = current_index + i 
        # % Sort each individual based on the objective
        sorted_based_on_objective = [] 
        for i in range(M):
            sorted_based_on_objective, index_of_objectives = np.sort(y[:,V + i]) 
            sorted_based_on_objective = [] 
            for j in len(index_of_objectives):
                sorted_based_on_objective[j,:] = y[index_of_objectives [j],:]
                
            f_max = sorted_based_on_objective(len(index_of_objectives), V + i) 
            f_min = sorted_based_on_objective(1, V + i) 
            y[index_of_objectives(len(index_of_objectives)),M + V + 1 + i] = np.inf 
            y[index_of_objectives(1),M + V + 1 + i] = np.inf
            for j in range( 2 , len(index_of_objectives) ):
                next_obj  = sorted_based_on_objective(j + 1,V + i) 
                previous_obj  = sorted_based_on_objective(j - 1,V + i) 
                if (f_max - f_min == 0):
                    y[index_of_objectives [j],M + V + 1 + i] = np.inf
                else:
                    y[index_of_objectives [j],M + V + 1 + i] = (next_obj - previous_obj)/(f_max - f_min)
        distance = []
        distance[:,1] = np.zeros(shape = (len(F(front).f),1))
        for i in range(M):
            distance[:,0] = distance[:,1] + y[:,M + V + 1 + i]
        
        y[:,M + V + 2] = distance
        y = y[:,1 : M + V + 2]
        z[previous_index:current_index,:] = y
    f = z()    
    
    return f


# %%
def tournament_selection(chromosome, pool_size, tour_size):
    
    # % Get the size of chromosome. The number of chromosome is not important
    # % while the number of elements in chromosome are important.
    pop, variables = chromosome.shape
    # print('chromosome type', type(chromosome),  ',  shape',chromosome.shape)
    # print(chromosome[0,-1])
    # % The peunltimate element contains the information about rank.
    # rank = variables - 1
    # % The last element contains information about crowding distance.
    # distance = variables
    
    candidate = np.zeros(tour_size)
    c_obj_rank = np.zeros(tour_size)
    c_obj_distance = np.zeros(tour_size)
    f_ = np.nan*np.ones(shape=(1,chromosome.shape[1]))
    
    # % Until the mating pool is filled, perform tournament selection
    for i in range (pool_size):
        # % Select n individuals at random, where n = tour_size
        for j in range (tour_size):
            # % Select an individual at random
            candidate[j] = round(pop*random.random())
            
            # # % Make sure that the array starts from one. 
            if candidate[j] == 40 :
                candidate[j] = 39
                                    
        # % Collect information about the selected candidates.
        # for j in range (tour_size):
            c_obj_rank[j] = chromosome[int(candidate[j]),-2]            
            c_obj_distance[j] = chromosome[int(candidate[j]),-1]
        
        
        # % If more than one candiate have the least rank then find the candidate
        # % within that group having the maximum crowding distance.
        if len(set(c_obj_rank))!=1: #diff rank
            min_candidate = np.argmin(c_obj_rank)
            # print('min_candidate',min_candidate)
            # print(' candidate',candidate[int(min_candidate)])
            f = chromosome[int(candidate[min_candidate]),:]
        
        else: #same rank
            # max_candidate = np.where(c_obj_distance(min_candidate) == max(c_obj_distance(min_candidate)))
            max_candidate = np.argmax(c_obj_distance)
            # % If a few individuals have the least rank and have maximum crowding
            # % distance, select only one individual (not at random). 
            # if len(max_candidate) != 1:
            #     max_candidate = max_candidate[0]
           
            # % Add the selected individual to the mating pool
            f = chromosome[int(candidate[max_candidate]),:]
        # else: #different rank
            # % Add the selected individual to the mating pool
            
        f_ = np.vstack((f_,f))
        f_ = f_[~np.isnan(f_).any(axis=1)]
    return f_


# %%
def replace_chromosome(intermediate_chromosome, M, V,pop):
    
    N = len(intermediate_chromosome)
    # print('N = ', N)

    # % Get the index for the population sort based on the rank
    index = np.argsort(intermediate_chromosome[:,M + V + 1])
    
    
    sorted_chromosome = intermediate_chromosome[index]
    # print('sorted_chromosome',sorted_chromosome)
    # % Find the maximum rank in the current population
    max_rank = max(intermediate_chromosome[:,M + V ]) 
    
    f = np.nan * np.ones(shape=(pop,M+V+2))
    
    # % Start adding each front based on rank and crowing distance until the
    # % whole population is filled.
    previous_index = 0
    for i in range ( int( max_rank ) ):
        # % Get the index for current rank i.e the last the last element in the
        # % sorted_chromosome with rank i. 
        current_index = max(np.argwhere(sorted_chromosome[:,M + V ] == i)) 
        previous_index = int(previous_index)
        current_index = int(current_index)
        
        # print('previous index', previous_index, 'current index', current_index)
        # % Check to see if the population is filled if all the individuals with
        # % rank i is added to the population. 
        if current_index > pop :
            # % If so then find the number of individuals with in with current
            # % rank i.
            remaining = pop - previous_index 
            # % Get information about the individuals in the current rank i.
            temp_pop = sorted_chromosome[previous_index : current_index + 1, :]
            # % Sort the individuals with rank i in the desc ing order based on
            # % the crowding distance.
            
            temp_sort_index = -np.argsort(-temp_pop[:, M + V + 1])
            temp_sort = temp_pop[temp_sort_index]
            
            
            # % Start filling individuals into the population in desc ing order
            # % until the population is filled.
            
            f = temp_sort[:remaining, :]
            
            return f
        elif current_index < pop:
            # % Add all the individuals with rank i into the population.
            f[previous_index : current_index + 1, :] = sorted_chromosome[previous_index : current_index + 1, :]
        else:
            # % Add all the individuals with rank i into the population.
            f = sorted_chromosome[:pop, :]
            return f
        
        # % Get the index for the last added individual.
        previous_index = current_index
        # print('f', f)
    
    return f

#%%
def genetic_operator(parent_chromosome, M, V, mu, mum, l_limit, u_limit, sys_param):
    import emodps
    
    N, m = parent_chromosome.shape
    
    p = 0 
    # % Flags used to set if crossover and mutation were actually performed. 
    was_crossover = 0 
    was_mutation = 0 
    
    u = np.zeros(V)
    bq = np.zeros(V)
    r = np.zeros(V)
    delta = np.zeros(V)
    child = np.nan * np.ones(shape = (1,M+V) )
    
    for i in range ( N ) :
        # % With 90 % probability perform crossover
        if random.random() < 0.9:
            # % Initialize the children to be null vector.
            child_1 = np.nan*np.ones(V+M)
            child_2 = np.nan*np.ones(V+M)
            # % Select the first parent
            parent_1 = round(N*random.random()) 
            if parent_1 == N:
                parent_1 = N-1 
             
            # % Select the second parent
            parent_2 = round(N*random.random()) 
            if parent_2 == N:
                parent_2 = N-1 
             
            # % Make sure both the parents are not the same. 
            while (parent_1 == parent_2):
                parent_2 = round(N*random.random()) 
                if parent_2 == N:
                    parent_2 = N-1 
                 
             
            # % Get the chromosome information for each randomnly selected
            # % parents
            parent_1 = parent_chromosome[parent_1,:] 
            parent_2 = parent_chromosome[parent_2,:] 
            # % Perform corssover for each decision variable in the chromosome.
            for j in range (V):
                # % SBX (Simulated Binary Crossover).
                # % For more information about SBX refer the enclosed pdf file.
                # % Generate a random number
                u [j] = random.random() 
                if u [j] <= 0.5:
                    bq [j] = (2*u [j])**(1/(mu+1)) 
                else:
                    bq [j] = (1/(2*(1 - u [j])))**(1/(mu+1)) 
                
                # % Generate the jth element of first child
                child_1 [j] = 0.5*(((1 + bq [j])*parent_1 [j]) + (1 - bq [j])*parent_2 [j]) 
                # % Generate the jth element of second child
                child_2 [j] =  0.5*(((1 - bq [j])*parent_1 [j]) + (1 + bq [j])*parent_2 [j]) 
                # % Make sure that the generated element is within the specified
                # % decision space else set it to the appropriate extrema.
                                
                if child_1 [j] > u_limit [j]:
                    child_1 [j] = u_limit [j] 
                elif child_1 [j] < l_limit [j]:
                    child_1 [j] = l_limit [j] 
                
                if child_2 [j] > u_limit [j]:
                    child_2 [j] = u_limit [j] 
                elif child_2 [j] < l_limit [j]:
                    child_2 [j] = l_limit [j]
                    
            # % Evaluate the objective function for the offsprings and as before
            # % concatenate the offspring chromosome with objective value.
            
            child_1[V : M + V] = emodps.evaluate_objective(child_1, M, V, sys_param) 
            child_2[V : M + V] = emodps.evaluate_objective(child_2, M, V, sys_param) 
            # % Set the crossover flag. When crossover is performed two children
            # % are generate, while when mutation is performed only child is generated.
            was_crossover = 1 
            was_mutation = 0 
        # % With 10 % probability perform mutation. Mutation is based on
        # % polynomial mutation. 
        else :
            # % Select at random the parent.
            parent_3 = round(N*random.random()) 
            if parent_3 == N:
                parent_3 = N-1
            
            # % Get the chromosome information for the randomnly selected parent.
            child_3 = parent_chromosome[parent_3,:]
            # % Perform mutation on eact element of the selected parent.
            for j in range( V ):
               r [j] = random.random()
               if r [j] < 0.5:
                   delta [j] = (2*r [j])**(1/(mum+1)) - 1 
               else:
                   delta [j] = 1 - (2*(1 - r [j]))**(1/(mum+1)) 
               
               # % Generate the corresponding child element.
               child_3 [j] = child_3 [j] + delta [j] 
               # % Make sure that the generated element is within the decision
               # % space.
               if child_3 [j] > u_limit [j]:
                   child_3 [j] = u_limit [j] 
               elif child_3 [j] < l_limit [j]:
                   child_3 [j] = l_limit [j]
                   
            # % Evaluate the objective function for the offspring and as before
            # % concatenate the offspring chromosome with objective value.    
            child_3[V : M + V] = emodps.evaluate_objective(child_3, M, V, sys_param) 
            # % Set the mutation flag
            was_mutation = 1 
            was_crossover = 0
        # % Keep proper count and appropriately fill the child variable with all
        # % the generated children for the particular generation.
        if was_crossover:
            child = np.vstack((child,  child_1,child_2))
            child = child[~np.isnan(child).any(axis=1)]
            was_crossover = 0 
            p = p + 2 
        elif was_mutation:
            child = np.vstack((child,  child_3[: M + V]))
            was_mutation = 0 
            p = p + 1
            
    f = child
    
    return f