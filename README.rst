====================================
Joel Friedly's AI Algorithms Library
====================================

This library is slow and buggy; you should not use it.

It was created as part of the requirements for three classes that I took in college:  Intro to Neural Networks, Intro to Artificial Intelligence, and Intro to Machine Learning.
Each of these classes required students to implement a neural network, and so this library grew over time.
As it grew, K-Means clustering and a K-Nearest Neighbor classifier were added to the library, now in ``k_means.py`` and ``k_nearest_neighbor.py``, respectively.

----

Structure
---------

The library contains five classes divided among three files (modules).

* neural_network.py contains Neuron, NeuralNetwork and NeuralNetworkManager

* k_means.py contains Cluster and KMeansClusterManager

* k_nearest_neighbor contains KNNClassifier

* utils.py contains useful functions (activation functions, Euclidean distance, etc.)

``neural_network.py``
'''''''''''''''''''''
This module is the core of the library, implementing all the NN related functionality.
The Neuron and even the NeuralNetwork classes aren't really intended to be used by programmers, but are available if necessary.
Instead, the NeuralNetworkManager class is meant as the primary interface for acting on NNs, and it contains the highest-level functions.

``k_means.py``
''''''''''''''
This module was originally added to support K-Means clustering in use with a neural network, but is sufficiently abstract to be used on its own.
The Cluster class, like the Neuron or NeuralNetwork classes, is not really intended to be used by programmers, but is available if necessary.
Instead, the KMeansClusterManager class is meant as the primary interface for acting on classifiers, and it follow a format very similar to the NeuralNetworkManager class.

``k_nearest_neighbor.py``
'''''''''''''''''''''''''
This module was added to support KNN classifying for a lab that required both KNN and neural network-based classification.
The KNNClassifier class follows a format somewhat similar to the Manager classes above.

``utils.py``
''''''''''''
This module contains mostly activation functions for different neural networks, but also has some general use functions, such as an efficient implementation of Euclidean distance.

See the individual modules for further documentation.
