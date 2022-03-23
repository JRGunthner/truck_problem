import numpy as np
import pandas as pd
import webbrowser

# GENETIC ALGORITHM TO SOLVE THE TRUCK PROBLEM
# by Juliano Günthner
# May 31th, 2019

genes  = 35
weight = np.random.uniform(low=1.0, high=250.0, size=(genes,1))[:,0]
value  = np.random.uniform(low=1,   high=1500.0, size=(genes,1))[:,0]
chromosomes 	= 50
num_generations = 100
num_parents_mating = int(0.2*chromosomes)
index = []

item_number = 0
for i in range(len(weight)):
	index.append('item ' + str(item_number))	
	item_number+=1

total_value  = 0
total_weight = 0
error = 1;

def cal_pop_fitness(pop):
	global capacity, total_value, total_weight, error;
	
	total_value  = 0
	total_weight = 0
	idx = 0
	
	for i in pop:
		if idx >= genes:
			break
		if (i == 1):
			total_weight += weight[idx]
			total_value  += value[idx]
		idx += 1
	
	if total_weight > capacity:
		return 0
	else:
		error = 0;
		return total_value

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :]   = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
	offspring = np.empty(offspring_size)
	# The point at which crossover takes place between two parents. Usually, it is at the center.
	crossover_point = np.uint8(offspring_size[1]/2)
	for k in range(offspring_size[0]):
		# Index of the first parent to mate.
		parent1_idx = k%parents.shape[0]
		# Index of the second parent to mate.
		parent2_idx = (k+1)%parents.shape[0]
		# The new offspring will have its first half of its genes taken from the first parent.
		offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
		# The new offspring will have its second half of its genes taken from the second parent.
		offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
	return offspring

def mutation(offspring_crossover):
	# Mutation changes a single gene in each offspring randomly.
	k = 0
	for idx in range(offspring_crossover.shape[0]):
		k = np.random.uniform(0,1)
		if k > 0.08:
			g = np.random.randint(0,genes)
			if offspring_crossover[idx,g] == 0:
				offspring_crossover[idx,g] = 1
			else:
				offspring_crossover[idx,g] = 0
	return offspring_crossover

if __name__ == "__main__":

	capacity = input("Insira o limite de peso do caminhão (kg): ")
	capacity = float(capacity)

	# Defining the population size.
	population_size = (chromosomes, genes)

	# Creating the initial population.
	new_population = np.random.randint(1, size=(chromosomes,genes))

	best_outputs = []

	for generation in range(num_generations):
		print("Geração: ", generation)

		# Measuring the fitness of each chromosome in the population.
		fitness = []
		for c in new_population:
			fitness.append(cal_pop_fitness(c))

		# Selecting the best parents in the population for mating.
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		# Generating next generation using crossover.
		offspring_crossover = crossover(parents, offspring_size=(population_size[0]-parents.shape[0], genes))
		
		# Adding some variations to the offspring using mutation.
		offspring_mutation = mutation(offspring_crossover)

		# Creating the new population based on the parents and offspring.
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation

	# Getting the best solution after iterating finishing all generations.
	fitness = []
	for c in new_population:
		fitness.append(cal_pop_fitness(c))

	# Then return the index of that solution corresponding to the best fitness.
	best_match_idx = np.where(fitness == np.max(fitness))

	best_solution = new_population[best_match_idx, :][0][0]

	print("Finalizado")

	cnt = 0;
	best_weight = []
	best_value  = []
	best_index  = []
	cols = ["Peso (kg)", "Valor (R$)"]
	for i in best_solution:
		if i==1:
			best_weight.append(weight[cnt])
			best_value.append(value[cnt])
			best_index.append(index[cnt])
		cnt += 1

	items_input = np.column_stack((weight,value))
	best_combination = np.column_stack((best_weight,best_value))

	df = pd.DataFrame(items_input, index = index, columns = cols)

	html_input = df.to_html()

	df = pd.DataFrame(best_combination, index = best_index, columns = cols)

	html_result = df.to_html()

	f = open("output.html","w+")

	text_html = """<html>
	<!DOCTYPE html>
	<html lang="pt-br">
	<head>
	<title>Algoritmo Genetico, Problema do Caminhao</title>
	<link rel="stylesheet" type="text/css" href="style.css">
	</head>
	<body>
	<h1>Solução para o carregamento do caminhão</h1>
	"""
	f.write(text_html)
	f.write("<h2>Lista dos itens</h2>")
	f.write(html_input)
	f.write("<h2>Lista dos itens após otimização</h2>")
	f.write(html_result)
	f.write("<h3>Limite de peso do caminhão: %5.2fkg</h3>" % capacity)
	f.write("<h3>Peso total: %5.2fkg</h3>" % np.sum(weight))
	f.write("<h3>Valor total: R$%5.2f</h3>" % np.sum(value))
	f.write("<h4>Peso total após otimização: %5.2fkg</h4>" % np.sum(best_weight))
	f.write("<h4>Valor total após otimização: R$%5.2f</h4>" % np.sum(best_value))
	if np.sum(best_weight) > capacity:
		f.write("<h4>ERRO: SOLUÇÃO NÃO ENCONTRADA!</h4>")

	text_html = """
    </body>
    </html>"""

	f.write(text_html)

	webbrowser.open_new_tab('output.html')
