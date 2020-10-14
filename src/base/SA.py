import numpy as np


def SA(
    fitness_function,
    problem_dim: int,
    bounds: list,
    epochs: int,
    trials: int,
    initial_probability: float,
    final_probability: float,
    verbose: bool = False,
):

    x = np.array([bounds[0]] * problem_dim) + np.array(
        [abs(bounds[1] - bounds[0])] * problem_dim
    ) * np.random.rand(
        problem_dim
    )  # initial population

    na = 1.0  # number of accepted solution
    temp_init = -1.0 / np.log(initial_probability)  # initial temprature
    temp_final = -1.0 / np.log(final_probability)  # final temprature
    red = (temp_final / temp_init) ** (1.0 / (epochs - 1))  # reduction factor
    DeltaE_avg = 0.0

    temp = temp_init
    fit = fitness_function(x)

    x_best = np.zeros((epochs + 1, problem_dim))
    fit_best = np.zeros(epochs + 1)
    x_best[0] = np.copy(x)
    fit_best[0] = np.copy(fit)

    fit_epochs = []
    x_epochs = []

    if verbose:
        all_featues = np.ones(problem_dim)
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

        for trial in range(trials):
            x_new = np.copy(
                x + (np.random.uniform(low=-0.25, high=0.25, size=problem_dim))
            )

            x_new = np.minimum(x_new, np.array([1.0] * problem_dim))
            x_new = np.maximum(x_new, np.array([0.0] * problem_dim))
            fit_new = fitness_function(x_new)
            DeltaE = np.abs(fit_new - fit)

            if fit_new > fit:
                if epoch == 0 and trial == 0:
                    DeltaE_avg = DeltaE

                p = np.exp(-DeltaE / (DeltaE_avg * temp))

                if np.random.rand() < p:
                    accept = True
                else:
                    accept = False
            else:
                accept = True

            if accept == True:
                x = np.copy(x_new)
                fit = np.copy(fit_new)
                na += 1
                DeltaE_avg = (DeltaE_avg * (na - 1.0) + DeltaE) / na

        x_best[epoch + 1] = np.copy(x)
        fit_best[epoch + 1] = np.copy(fit)

        temp = red * temp
        fit_epochs.append(fit)
        x_epochs.append(x)
        if verbose:
            print("Accuracy: %.3f." % (1 - fit))

    if verbose:
        print("----------------------------------")

    return (fit_epochs, x_epochs)
