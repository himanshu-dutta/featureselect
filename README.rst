=========================
Feature Select PyPackage
=========================

`Feature Select`_ is a simple yet effective solution to select features from a numeric dataset, which yields the best results, given a Machine Learning algorithm.

- GitHub repo: https://github.com/himanshu-dutta/featureselect/
- Free software: MIT license

Features
--------

- Multiple optimization algorithms to work with.
- Works with most class based Machine Learning models over a range of libraries.
- Compatible with all platforms.

.. _`Feature Select` : https://github.com/himanshu-dutta/featureselect/

Quickstart
----------

Install the latest Feature Select with ::

    pip install featureselect


Usage
-----

.. code:: python

   from featureselect import DEOptimizer, SAOptimizer, GAOptimizer, PSOptimizer
   from sklearn.tree import DecisionTreeClassifier
   import pandas as pd
   
   # loading a dataset
   dataset = pd.read_csv("dataset.csv", header=None)
   dataset[34] = dataset[34].apply(lambda x: 1 if x == "g" else 0)
   dataset = dataset.dropna()
   X, y = dataset.iloc[:, :-1].to_numpy(), dataset.iloc[:, -1].to_numpy()

   # best_accuracy, index_of_best_features = GAOptimizer((X, y), DecisionTreeClassifier, epochs = 10, threshold=0.6, verbose=1, max_depth=3)
   # best_accuracy, index_of_best_features = SAOptimizer((X, y), DecisionTreeClassifier, epochs = 10, threshold=0.6, verbose=True, max_depth=3)
   # best_accuracy, index_of_best_features = PSOptimizer((X, y), DecisionTreeClassifier, epochs = 10, verbose=1, max_depth=3)


   best_accuracy, index_of_best_features = DEOptimizer((X, y), DecisionTreeClassifier, epochs = 10, threshold=0.6, verbose=1, max_depth=3)

   #############
   #   Output
   #############
   Initial Accuracy: 0.887.
   ----------------------------------
   *  Epoch:  1 | Accuracy: 0.958.
   ----------------------------------
   *  Epoch:  2 | Accuracy: 0.958.
   ----------------------------------
   *  Epoch:  3 | Accuracy: 0.958.
   ----------------------------------
   *  Epoch:  4 | Accuracy: 0.958.
   ----------------------------------
   *  Epoch:  5 | Accuracy: 0.972.
   ----------------------------------
   *  Epoch:  6 | Accuracy: 0.972.
   ----------------------------------
   *  Epoch:  7 | Accuracy: 0.972.
   ----------------------------------
   *  Epoch:  8 | Accuracy: 0.972.
   ----------------------------------
   *  Epoch:  9 | Accuracy: 0.986.
   ----------------------------------
   *  Epoch: 10 | Accuracy: 0.986.
   ----------------------------------
   (0.9859154929577465, array([ 2,  4,  5,  6,  9, 11, 12, 13, 14, 17, 19, 20, 21, 24, 26, 29, 32]))




Note
----

The project is still in developement phase and will be expanded and made better over time. Any contribution to it is welcomed. Stable release would be made available soon.
