import numpy as np


def DE(
    fitness_function,
    bounds: list,
    dimension: int,
    target_size: int,
    epochs: int,
    scaling_factor: float,
    crossover_prob: float,
    threshold_epochs: int = None,
    verbose: bool = False,
):

    target_vector = np.array([bounds[0]] * dimension) + np.array(
        [abs(bounds[1] - bounds[0])] * dimension
    ) * np.random.rand(
        target_size, dimension
    )  # target vector

    fitness = np.array([fitness_function(pop) for pop in target_vector])

    iter_fitness = []
    iter_best = []

    if verbose:
        all_featues = np.ones(dimension)
        print(f"Initial Accuracy: %.3f." % (1 - fitness_function(all_featues)))

    for epoch in range(epochs):
        if verbose:
            print("----------------------------------")
            print(
                "*  Epoch: {epoch:{width}}".format(
                    epoch=epoch + 1, width=len(str(epochs))
                ),
                end=" | ",
            )
        trial_vector = []

        for pop in range(target_size):
            choices = list(range(pop)) + list(range(pop + 1, target_size))
            idx = np.random.choice(choices, 3)

            X1 = np.copy(target_vector[idx[0]])
            X2 = np.copy(target_vector[idx[1]])
            X3 = np.copy(target_vector[idx[2]])

            doner_vector = X1 + scaling_factor * (X2 - X3)

            delta = np.random.choice(range(dimension))

            tv_row = [
                doner_vector[i]
                if np.random.rand() <= crossover_prob or delta == i
                else target_vector[pop, i]
                for i in range(dimension)
            ]
            trial_vector.append(tv_row)

        trial_vector = np.clip(np.array(trial_vector), bounds[0], bounds[1])
        trial_fitness = np.array([fitness_function(pop) for pop in trial_vector])

        for idx in range(target_size):
            if trial_fitness[idx] < fitness[idx]:
                target_vector[idx] = np.copy(trial_vector[idx])
                fitness[idx] = trial_fitness[idx]

        iter_fitness.append(np.min(fitness))
        iter_best.append(list(target_vector[np.argmin(fitness)]))
        if verbose:
            print("Accuracy: %.3f." % (1 - iter_fitness[-1]))

        if (
            threshold_epochs
            and len(iter_fitness) >= threshold_epochs
            and np.all(iter_fitness[-1] == iter_fitness[-threshold_epochs:])
        ):
            break

    if verbose:
        print("----------------------------------")

    return (iter_fitness, iter_best)
