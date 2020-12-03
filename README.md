# Rpnn (under construction)
This is an idiomatic C++ implementation of _Resilient backprop Neural Network_ with an easy and convenient user interface.
No deppendancies, fully contained implementation.

Resilient backprop is [known to be the fastest](https://en.wikipedia.org/wiki/Rprop) learning NN in the family of backprops, featuring
a number of advantages of the standard backprop mechanisms:
- the learning rule is no longer proportional to the size of the gradient, only the sign of the computed gradient matters
  - programmatically it means no need for plugging a partial derrevative of the logistic function (plugging only the logistic is enough)
  - it's not prone to [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) that the standart backprop suffers from
- the configuration of the _rprop_ is simple and not as complex and sensitive as the standard backprop's
- this implemenation provides auto normalization of outputs and optionally of the inputs (why inputs might need normalization [link TBU])
- the framework is fully and easy SERSES'able [link TBU]
- the framwork also support multi-class classification (support of _Softmax_ logistic at the output perceptron)
