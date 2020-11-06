import numpy as np


class GA:
    def __init__(
        self,
        fitness_function,
        population_size: int,
        chromosome_size: int,
        bounds: list,
        survival_rate: float,
        mutation_ratio: float,
        generation_count: int,
    ):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.bounds = bounds
        self.survival_rate = survival_rate
        self.mutation_ratio = mutation_ratio
        self.generation_count = generation_count
        self.fitness_function = fitness_function
        self.best_results = []
        self.best_fitness = []

        # self.population = np.random.choice(
        #     2, size=(self.population_size, self.chromosome_size), p=[0.7, 0.3])
        self.population = np.array([bounds[0]] * chromosome_size) + np.array(
            [abs(bounds[1] - bounds[0])] * chromosome_size
        ) * np.random.rand(self.population_size, self.chromosome_size)

    def calc_fitness(self):
        self.fitness = np.array(
            [self.fitness_function(chromosome) for chromosome in self.population]
        )

    def selection(self):
        fitness = np.copy(self.fitness)
        parents = np.hstack(
            (np.copy(self.population), fitness.reshape(self.population.shape[0], -1))
        )
        parents = parents[parents[:, -1].argsort()]
        cutoff = int(np.floor(self.survival_rate * self.population_size))
        self.parents = np.copy(parents[:cutoff, :-1])
        self.best_results.append(list(np.copy(self.parents[0, :])))

    def crossover(self):
        # Crossing is done in a single-point manner, so as to breed the best of the current generation
        offspring_size = (
            int(np.ceil((1 - self.survival_rate) * self.population_size)),
            self.chromosome_size,
        )
        parents_size = self.parents.shape[0]
        children = np.empty(offspring_size)

        for i in range(offspring_size[0]):
            parent1 = i % parents_size
            parent2 = (i + 1) % parents_size
            crossover_point = int(np.floor(np.random.rand() * self.chromosome_size))

            children[i, :crossover_point] = self.parents[parent1, :crossover_point]
            children[i, crossover_point:] = self.parents[parent2, crossover_point:]
        self.children = np.copy(children)

    def mutation(self):
        mutation_size = int(np.floor(self.mutation_ratio * self.chromosome_size))

        for child in range(self.children.shape[0]):
            genes = np.random.choice(self.chromosome_size, mutation_size)
            for gen_pos in genes:
                self.children[child, gen_pos] += (
                    np.random.choice([-1.0, 1.0])
                    * self.fitness[child % self.parents.shape[0]]
                )
            self.children[child] = np.minimum(
                self.children[child], np.array([1.0] * self.chromosome_size)
            )
            self.children[child] = np.maximum(
                self.children[child], np.array([0.0] * self.chromosome_size)
            )
            # print(factor, self.children[child])

    def run(self, verbose: bool = False):
        if verbose:
            all_featues = np.ones(self.chromosome_size)
            print(f"Initial Accuracy: %.3f." % (1 - self.fitness_function(all_featues)))

        for gen in range(self.generation_count):
            if verbose:
                print("----------------------------------")
                print(
                    "*  Epoch: {epoch:{width}}".format(
                        epoch=gen + 1, width=len(str(self.generation_count))
                    ),
                    end=" | ",
                )
            # Calculation of fitness of each generation
            self.calc_fitness()

            # Adding the best result of each generation to a list
            self.best_fitness.append(self.fitness.min())
            if verbose:
                print("Accuracy: %.3f." % (1 - self.fitness.min()))

            # Selection of Parents to crossover
            self.selection()

            # Crossing-over to get children
            self.crossover()

            # Adding variations/mutations to the current children
            self.mutation()

            # Updating the population with the parent and children
            population = np.empty(self.population.shape)
            population[: self.parents.shape[0], :] = self.parents
            population[self.parents.shape[0] :, :] = self.children
            self.population = np.copy(population)

        if verbose:
            print("----------------------------------")

        return (self.best_fitness, self.best_results)
