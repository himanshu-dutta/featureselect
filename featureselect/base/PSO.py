import numpy as np


def PSO(
    fitness_function,
    particle_size: int,
    particle_dim: int,
    bounds: list,
    intertial_weight: int,
    particle_coeff: int,
    swarm_coeff: int,
    epochs: int,
    threshold_epochs: int = None,
    verbose: float = False,
):

    pop = np.array([bounds[0]] * particle_dim) + np.array(
        [abs(bounds[1] - bounds[0])] * particle_dim
    ) * np.random.rand(
        particle_size, particle_dim
    )  # initial population

    vel = np.array([bounds[0]] * particle_dim) + np.array(
        [abs(bounds[1] - bounds[0]) / 2] * particle_dim
    ) * np.random.rand(particle_size, particle_dim)

    fit = np.array([fitness_function(p) for p in pop])

    if verbose:
        all_featues = np.ones(particle_dim)
        print(f"Initial Accuracy: %.3f." % (1 - fitness_function(all_featues)))

    pbest = np.copy(pop)
    f_pbest = np.copy(fit)

    f_gbest, gbest = np.min(f_pbest), pbest[np.argmin(f_pbest)]
    best_fit_epochs = []
    best_particle_epochs = []

    # iterating over the entire population

    for epoch in range(epochs):
        if verbose:
            print("----------------------------------")
            print(
                "*  Epoch: {epoch:{width}}".format(
                    epoch=epoch + 1, width=len(str(epochs))
                ),
                end=" | ",
            )

        # updating individuals in the population

        for p in range(particle_size):

            vel[p, :] = (
                (intertial_weight * vel[p, :])
                + (
                    particle_coeff
                    * np.random.rand(1, particle_dim)
                    * (pbest[p, :] - pop[p, :])
                )
                + (swarm_coeff * np.random.rand(1, particle_dim) * (gbest - pop[p, :]))
            )

            pop[p, :] += vel[p, :]
            pop[p, :] = np.where(pop[p] > 1, 1.0, np.where(pop[p] < 0, 0.0, pop[p]))

            fit[p] = fitness_function(pop[p, :])

            if fit[p] < f_pbest[p]:
                f_pbest[p] = fit[p]
                pbest[p] = pop[p]

                if f_pbest[p] < f_gbest:
                    f_gbest = f_pbest[p]
                    gbest = pbest[p]

        if verbose:
            print("Accuracy: %.3f." % (1 - f_gbest))

        best_fit_epochs.append(f_gbest)
        best_particle_epochs.append(gbest)
        if (
            threshold_epochs
            and len(best_fit_epochs) >= threshold_epochs
            and np.all(best_fit_epochs[-1] == best_fit_epochs[-threshold_epochs:])
        ):
            break

    if verbose:
        print("----------------------------------")

    return (best_fit_epochs, best_particle_epochs)
