# ML Toys

ML Toys is a collection of machine learning toy problems.
These toy problems are intended to facilitate the testing of new machine learning code and algorithms by providing small instances with a variety of challenges.
Given an appropriate algorithm, these toy problems can be quickly solved.
However, not all toy problems will be solvable with all algorithms.

ML Toys follows the [factory method](https://en.wikipedia.org/wiki/Factory_method_pattern) design pattern, and provides MLToyFactory classes for each kind of toy problem.
An MLToyFactory object will generate random MLToyInstance objects unless a seed is provided.
This allows an algorithm to be tested repeatedly with minimal worry about overfitting, since there will be a fresh MLToyInstance wraps its own set of training and test data.

A [demo script estimating the training data average](demos/demo-average.py) is provided.

```
demo model using training average for all predictions:
ConstantFunctionFactory : min/mean/max = 0.0000/0.0000/0.0000
LinearFunctionFactory : min/mean/max = 0.0016/0.1427/0.3219
```
