import numpy as np
import math
import random
import csv
# i would want to acknowledge the creators of all the libraries used in this work

# file_name = input("input the test instance: ")
#Task 1
#Read in problem instance file
def read_file(file_name):
    with open(file_name, 'r') as file:
        contents = file.read()
    #splitting the characters into individual characters
    char = contents.split()
    # first two numbers, rows(m) and columns(n)
    m, n = int(char[0]), int(char[1])
    # Creating the
    A = np.zeros((m, n))
    C = np.zeros(n)
    # Iterating to fill up the C vector
    i = 0
    for num in char[2:]: #numbers afer the first two characters
        C[i] = int(num)
        i += 1
        if i == n:
            break
    # Iterating over the remaining numbers and fill up the elements in A.
    i = n + 2
    rows = [] # empty list to store row indexes
    cols = [] # empty list to store corresponding columns

    while i < len(char):
        #getting the row index
        row_ind = int(char[i])
        rows.append(row_ind)
        i += 1
        # getting the columns for the row indexes
        col_ind = char[i:i + row_ind]
        cols.append(col_ind)
        i += row_ind
        # Get the new row index
        if i == len(char):
            break
    # convert from strings to integers
    cols = [[int(col) for col in row] for row in cols]
    #updating the 1s in the A matrix
    for row_A, cols_r in zip(A, cols):
        for c in cols_r:
            row_A[c-1] = 1
    #print(rows)
    # print((sum([1 for val in A[3] if val == 1])))

    # print(A[16])
    return A, C

# A, C = read_file(file_name)







# task 2
# Calculate the solution using a greedy algorithm
def Constructive_heuristics(C, A):
    # Initialize variables
    n = len(C)  # Number of variables
    x = [0]*n  # Solution vector
    satisfied_const = [0] * len(A)  # Constraint satisfaction vector

    # Iterating until all constraints are satisfied
    while not all(satisfied_const):
        # Calculate the ratio of cost to unsatisfied constraints for each variable
        ratios = []
        for i in range(n):
            #considering only yet uncovered variables with at least one unsatisfied constraint
            sum_var = np.sum(A[:, i] * (1 - np.array(satisfied_const)))
            if x[i] == 0 and sum_var > 0:
                ratio = C[i] / sum_var
            else:
                ratio = float('inf')  #Set the ratio to infinity if the variable cannot be selected
            ratios.append(ratio)
        # Selecting variable with minimum ratio
        min_idx = ratios.index(min(ratios))
        # End the loop if no improvement can be made
        if ratios[min_idx] == float('inf'):
            break
        # Update the solution and constraint satisfaction vector
        x[min_idx] = 1
        for j in range(len(A)):
            satisfied_const[j] = max(satisfied_const[j],A[j][min_idx])

    value = sum([C[i] * x[i] for i in range(n)])  # final cost (C^Tx)
    # Return the solution and its value
    return x, value
# # Call the greedy algorithm and print the results
# solution, value = Constructive_heuristics(C, A)
# print("Solution:", solution)
# print("Value:", value)
# print(sum([1 for val in solution if val == 1]))
#
# # initial_sln = solution



# Task 3
# Get the number of searches to perform from the user
def local_search(costs, A, start_solution, num_searches, neigh_size=5):
    curr_best = np.copy(start_solution) #current best solution
    curr_best_cost = np.dot(costs, start_solution) # current best cost
     # Initialising the neighbourhood success rates and history
    neigh_success_rates = [1 / neigh_size] * neigh_size
    improve_history = [] # improvement history
    # loop the "num_searches' times to find a better solution
    for _ in range(num_searches):
        # calling the get_neighbour function to get a new solution and neighbourhood
        new_solution, neighbourhood = get_neighbour(curr_best, A, neigh_success_rates)
        # calculate the cost of the new solution
        new_solution_cost = np.dot(costs,new_solution)
        # checking if the new solution has a better cost than the current solution
        if new_solution_cost < curr_best_cost:
            # updating the current best solution with the new solution if the condition is met
            curr_best = new_solution
            curr_best_cost = new_solution_cost
            # Updating improvement history and success rates
            improve_history.append(neighbourhood)
            # Remove the oldest history if history becomes too long
            if len(improve_history) > neigh_size:
                improve_history.pop(0)
            #update success rate for each neighbourhood
            for i in range(neigh_size):
                neigh_success_rates[i] = improve_history.count(i) / len(improve_history)

    return curr_best, curr_best_cost


def get_neighbour(sol, constraint_matrix, _success_rates):
    new_sol = np.copy(sol)
    # Finding the positions of 0's and 1's in the solution
    zeros = [] # zero positions
    ones = [] # one positions
    for i in range(len(sol)):
        if sol[i] == 0:
            zeros.append(i)
        elif sol[i] == 1:
            ones.append(i)
    # If there are no 0's or 1's, return the unchanged solution
    if len(zeros) == 0 or len(ones) == 0:
        return new_sol
    # Selecting neighborhood based on success rates
    neighbourhood = np.random.choice(len(_success_rates), p =_success_rates)
    # interchanging values in selected neighborhood(randomly picking positions from the zeros and ones list)
    random0 = np.random.choice(zeros)
    random1 = np.random.choice(ones)
    #interchanging the values at the chosen positions in the new solution
    new_sol[random0], new_sol[random1] = new_sol[random1], new_sol[random0]
    # checking if the solution satisfy the constraints(Ax>=1)
    if np.all(np.dot(constraint_matrix, new_sol) >= 1):
        # return the new solution and neighbourhood if condition is satified
        return new_sol, neighbourhood
    else:
        # return original solution and None if the condition is not satisfied
        return sol, None


# # Get the number of searches to perform from the user
# num_searches = int(input("How many times do you want to search? "))
#
# # using the greedy algorithm solution as initial solution
# # n = len(C)  # Number of variables
# start_solution = np.copy(solution)
# # Perform the Local Search
# LS_solution, LS_cost = local_search(C, A, start_solution, num_searches)
#
# # Print the final solution and its cost
# print("Final Solution:", LS_solution)
# print("Final Cost:", LS_cost)
# print(sum([1 for val in final_solution if val == 1]))

# Saving the best solution to a CSV file
# with open(f"{file_name}.csv", "w", newline="") as csvfile:
#         csvwriter = csv.writer(csvfile)
#         for i in range(len(LS_solution)):
#             csvwriter.writerow([i + 1, LS_solution[i]])
#         csvwriter.writerow(["Cost", LS_cost])


# Task 4
# Modified get_neighbour function without neighborhood_success_rates parameter
def make_small_change(sol, constraint_matrix):
    new_sol = np.copy(sol)
    # Finding the positions of 0's and 1's in the solution
    zeros = []  # zero positions
    ones = []  # one positions
    for i in range(len(sol)):
        if sol[i] == 0:
            zeros.append(i)
        elif sol[i] == 1:
            ones.append(i)
    # return unmodified soultion if there are no 1's and 0's present
    if len(zeros) == 0 or len(ones) == 0:
        return new_sol
    #Select random zero and one positions to swap
    random_zero = np.random.choice(zeros)
    random_one = np.random.choice(ones)
    # interchanging the values at the chosen positions
    new_sol[random_zero], new_sol[random_one] = new_sol[random_one], new_sol[random_zero]
    # Check if the new solution satisfies the constraints
    if np.all(np.dot(constraint_matrix, new_sol) >= 1):
        return new_sol
    #return original solution if condition is not satisfied
    else:
        return sol

# Simulated annealing function
def simulated_annealing(costs, constraints, initial_solution, num_iter, initial_temp, alpha):
    # getting the initial solution and calculates its cost
    curr_solution = initial_solution.copy()
    curr_cost = np.dot(costs, initial_solution)

    # Keep track of the best solution and the best cost storing the initial solution and the current cost as best so far.
    best_solution = initial_solution.copy()
    best_cost = curr_cost
    # Adding threshold to avoid overflow when calculating acceptance probability
    threshold = -1e308 # to prevent an overflow if the exponent in the acceptance probabilty is too large, that is
    # if "(C(s) - C(s')) / T" is to large from the Boltzmann distribution: P = exp((C(s) - C(s')) / T).

    # Iterating for the given number of iterations
    for iteration in range(num_iter):
        # Calculate the current temperature using alpha and the iteration it is on
        temp = initial_temp * (alpha ** iteration)

        # Creating a new solution by making a small change to the current solution
        new_solution = make_small_change(curr_solution, constraints)
        new_cost = np.dot(costs, new_solution)

        # Calculating the probability of accepting the new solution
        exponent = min((curr_cost - new_cost) / temp, threshold) #relative improvement compared to previous soln
        accept_prob = math.exp(exponent) #probability acceptance to escape local minima (boltzmann distribution)

        # If new soltuin is better or random number is less than the acceptance probability
        if new_cost < curr_cost or random.random() < accept_prob:
            curr_solution = new_solution
            curr_cost = new_cost
            # Update the best solution and cost if the new solution is better
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

    return best_solution, best_cost

#
# # Parameters for simulated annealing
# num_iter = 10000 #maximum number of iteration
# initial_temp = 300
# alpha = 0.95
#
# # Call simulated annealing and print the results
# sa_solution, sa_cost = simulated_annealing(C, A, LS_solution, num_iter, initial_temp, alpha)
# print("Simulated Annealing Solution:", sa_solution)
# print("Simulated Annealing Cost:", sa_cost)
# print(sum([1 for val in sa_solution if val == 1]))


#Task 5

# List of test instances
#get instances from user
get_input = input("Enter test file(separate different files with commas): ")
test_instances = get_input.split(",")

# defining parameters for local search and simulated annealing
# local search
num_iter = [10000,5000]
neighbour_size = [3,5]
# for simulated annealiing
max_iter = [5000,10000]
start_temp = [200, 100]
alpha = [0.95, 0.99]


#load the files and run them
for file in test_instances:
    # loading the txt file
    A, C = read_file(file)
    # Run the constructive heuristics
    greedy_soln, greedy_value = Constructive_heuristics(C,A)
    print(f" test {file} results for constructive heuristics: {greedy_value} ")

    # run local search
    # to find the best results and corresponding parameters
    # initialising
    best_ls_results = float("inf")
    best_ls_param = None
    best_ls_soln = None

    for size in neighbour_size:
        for iter in num_iter:
            ls_solution, ls_cost_Value = local_search(C, A, greedy_soln,num_searches= iter, neigh_size=size)
            # finding the best solution out of the combinations
            if ls_cost_Value < best_ls_results:
                best_ls_results = ls_cost_Value
                best_ls_param = (size,iter)
                best_ls_soln = ls_solution

    print(f"test {file} results for local search: {best_ls_results} with parameters {best_ls_param}")
    # Saving the best solution to a CSV file
    with open(f"{file}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(len(best_ls_soln)):
            csvwriter.writerow([i + 1, best_ls_soln[i]])
        csvwriter.writerow(["Cost", best_ls_results])
    #for simulateed annealing
    #finding best results using best results from the local search as initial solution
    best_SA_results = float("inf")
    best_SA_paramters = None
    best_SA_soln = None

    for search in max_iter:
        for temp in start_temp:
            for a in alpha:
                SA_solution, SA_best_value = simulated_annealing(C, A, best_ls_soln, num_iter=search, initial_temp=temp,
                                                                 alpha=a)
                # finding the best solution out of the combinations
                if SA_best_value < best_SA_results:
                    best_SA_results = SA_best_value
                    best_SA_paramters = (search, temp, a)
                    #best_SA_soln = SA_solution
    print(
        f"Best simulated annealing result for {file}: {best_ls_results} with parameters {best_SA_paramters}")




