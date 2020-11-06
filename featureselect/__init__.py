__all__ = ["featureselect"]
__package__ = "featureselect"
__name__ = "featureselect"

# preprocessing & utility
import numpy as np
from .base.utils import set_parameters

# model algorithm
from .base.SA import SA
from .base.GA import GA
from .base.DE import DE
from .base.PSO import PSO


def DEOptimizer(
    dataset: tuple,
    model,
    fit: str = "fit",
    predict: str = "predict",
    epochs: int = 10,
    threshold: float = 0.5,
    test_size: float = 0.2,
    early_stopping: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    Differential Evolution Optimizer
    --------------------------------

    Parameters
    ----------

    - dataset : tuple, default=None
        The dataset to find optimized features for.

    - model : function, default=None
        The model which is to be trained to make predictions on the dataset.

    - fit : str, default='fit'
        THe method used by the model to fit to training data.

    - predict : str, default='predict'
        THe method used by the model to predict given the input.

    Output
    ------

    - accuracy : Best accuracy score found by the optimizer.

    - features : features which get the best accuracy attanined.
    """

    X, y = dataset

    DIM = X.shape[1]
    BOUNDS = [0.0, 1.0]

    SCALING_FACTOR = 0.85
    CROSS_PROB = 0.8
    TARGET_SIZE = 20
    EPOCHS = epochs

    fitness = set_parameters(model, X, y, fit, predict, threshold, test_size, **kwargs)

    losses, features = DE(
        fitness,
        BOUNDS,
        DIM,
        TARGET_SIZE,
        EPOCHS,
        SCALING_FACTOR,
        CROSS_PROB,
        int(EPOCHS / 5) if early_stopping else None,
        verbose,
    )

    loss, feature = np.min(losses), features[np.argmin(losses)]
    accuracy = 1 - loss

    feature = [1 if attr > threshold else 0 for attr in feature]
    feature = np.argwhere(feature == np.amax(feature)).reshape(
        -1,
    )

    return (accuracy, feature)


def GAOptimizer(
    dataset: tuple,
    model,
    fit: str = "fit",
    predict: str = "predict",
    epochs: int = 10,
    threshold: float = 0.5,
    test_size: float = 0.2,
    verbose: bool = False,
    **kwargs
):
    """
    Genetic Algorithm Optimizer
    ---------------------------

    Parameters
    ----------

    - dataset : tuple, default=None
        The dataset to find optimized features for.

    - model : function, default=None
        The model which is to be trained to make predictions on the dataset.

    - fit : str, default='fit'
        THe method used by the model to fit to training data.

    - predict : str, default='predict'
        THe method used by the model to predict given the input.

    Output
    ------

    - accuracy : Best accuracy score found by the optimizer.

    - features : features which get the best accuracy attanined.
    """

    opt = "GA"

    X, y = dataset

    # setting hyperparameters
    DIM = X.shape[1]
    BOUNDS = [0.0, 1.0]

    POP_SIZE = 20
    CHROM_SIZE = DIM
    SURVIVAL_RATE = 0.6
    MUTATION_RATIO = 0.30
    GEN_COUNT = epochs

    fitness = set_parameters(model, X, y, fit, predict, threshold, test_size, **kwargs)

    algo = GA(
        fitness, POP_SIZE, CHROM_SIZE, BOUNDS, SURVIVAL_RATE, MUTATION_RATIO, GEN_COUNT
    )

    losses, features = algo.run(verbose)
    loss, feature = np.min(losses), features[np.argmin(losses)]
    accuracy = 1 - loss

    feature = [1 if attr > threshold else 0 for attr in feature]
    feature = np.argwhere(feature == np.amax(feature)).reshape(
        -1,
    )

    return (accuracy, feature)


def PSOptimizer(
    dataset: tuple,
    model,
    fit: str = "fit",
    predict: str = "predict",
    epochs: int = 10,
    threshold: float = 0.5,
    test_size: float = 0.2,
    early_stopping: bool = False,
    verbose: bool = False,
    **kwargs
):
    """
    Particle Swarm Optimizer
    ------------------------

    Parameters
    ----------

    - dataset : tuple, default=None
        The dataset to find optimized features for.

    - model : function, default=None
        The model which is to be trained to make predictions on the dataset.

    - fit : str, default='fit'
        THe method used by the model to fit to training data.

    - predict : str, default='predict'
        THe method used by the model to predict given the input.

    Output
    ------

    - accuracy : Best accuracy score found by the optimizer.

    - features : features which get the best accuracy attanined.
    """

    opt = "PSO"

    X, y = dataset

    # setting hyperparameters

    DIM = X.shape[1]
    BOUNDS = [0.0, 1.0]

    EPOCHS = epochs  # the number of iterations for which the swarm will train
    w = 0.50  # inertial value for effect of current velocity
    c1 = 0.45  # coeff for effect of particle best position
    c2 = 0.55  # coeff for effect of swarm best position
    SIZE = 50  # number of particles

    fitness = set_parameters(model, X, y, fit, predict, threshold, test_size, **kwargs)

    losses, features = PSO(
        fitness,
        SIZE,
        DIM,
        BOUNDS,
        w,
        c1,
        c2,
        EPOCHS,
        int(EPOCHS / 5) if early_stopping else None,
        verbose,
    )
    loss, feature = losses[-1], features[-1]
    accuracy = 1 - loss

    feature = [1 if attr > threshold else 0 for attr in feature]
    feature = np.argwhere(feature == np.max(feature)).reshape(
        -1,
    )

    return (accuracy, feature)


def SAOptimizer(
    dataset: tuple,
    model,
    fit: str = "fit",
    predict: str = "predict",
    epochs: int = 10,
    threshold: float = 0.5,
    test_size: float = 0.2,
    verbose: bool = False,
    **kwargs
):
    """
    Simulated Annealing Optimizer
    -----------------------------

    Parameters
    ----------

    - dataset : tuple, default=None
        The dataset to find optimized features for.

    - model : function, default=None
        The model which is to be trained to make predictions on the dataset.

    - fit : str, default='fit'
        THe method used by the model to fit to training data.

    - predict : str, default='predict'
        THe method used by the model to predict given the input.

    Output
    ------

    - accuracy : Best accuracy score found by the optimizer.

    - features : features which get the best accuracy attanined.
    """

    opt = "SA"

    X, y = dataset

    # setting hyperparameters
    DIM = X.shape[1]
    BOUNDS = [0.0, 1.0]

    EPOCHS = epochs
    TRIALS = 50
    P1 = 0.8
    P2 = 0.001

    fitness = set_parameters(model, X, y, fit, predict, threshold, test_size, **kwargs)

    losses, features = SA(fitness, DIM, BOUNDS, EPOCHS, TRIALS, P1, P2, verbose)
    loss, feature = losses[-1], features[-1]
    accuracy = 1 - loss

    feature = [1 if attr > threshold else 0 for attr in feature]
    feature = np.argwhere(feature == np.max(feature)).reshape(
        -1,
    )

    return (accuracy, feature)
