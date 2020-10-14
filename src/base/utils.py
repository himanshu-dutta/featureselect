import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def set_parameters(
    model,
    X,
    y,
    fit: str = "fit",
    predict: str = "predict",
    threshold: float = 0.5,
    test_size: float = 0.2,
    **kwrgs
):
    def fitness(x):
        mdl = model(**kwrgs)
        return fitness_function(
            x, mdl, X, y, threshold, test_size, fit=fit, predict=predict
        )

    return fitness


def fitness_function(
    x,
    model,
    X,
    y,
    threshold: float = 0.5,
    test_size: float = 0.2,
    fit: str = "fit",
    predict: str = "predict",
):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=test_size
    )

    par_ = [1 if col > threshold else 0 for col in x]
    cols = np.argwhere(par_ == np.amax(par_)).reshape(
        -1,
    )
    x_train = X_train[:, cols]
    x_test = X_test[:, cols]

    getattr(model, fit)(x_train, y_train)

    y_p = getattr(model, predict)(x_test)
    acc = accuracy_score(y_test, y_p)

    return 1 - acc
