# `Rpnn` (under construction)
This is an idiomatic C++ implementation of _Resilient backprop Neural Network_ with an easy and convenient user interface.  
No dependencies, fully contained implementation.

##### Enhancement requests and/or questions are more than welcome: *ldn.softdev@gmail.com*

Resilient backprop is [known to be the fastest](https://en.wikipedia.org/wiki/Rprop) learning NN in the family of backprops, featuring
a number of advantages of the standard backprop mechanism:
- the learning rule is not proportional to the size of the gradient, only the sign of the computed gradient matters
  - programmatically it means no need for plugging a derivative of the logistic function (plugging only the logistic will do)
  - it's not prone to [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) that the standard backprop suffers from
- the configuration of the _rprop_ is simple and not as complex and sensitive as the standard backprop's
- this implementation provides auto normalization of outputs and optionally of the inputs (why inputs might need normalization [link TBU])
- the framework is fully and easily SERSES'able [link TBU]
- the framework also support multi-class classification (support of _Softmax_ logistic at the output perceptron)
- the framework features a detection mechanism of local minimum traps and bouncing its weights out of the trap - ensures a high probability of convergence 

### Content:
1. [cli tool](https://github.com/ldn-softdev/jtc#cli-tool)
    * [Manual installation](...)
    * [`rpn` operations](...)
2. [C++ user interface](...)


## cli tool
This package provides a simple unix cli tool which allows running `Rpnn` from shell and toying with your data easily:
```help
bash $ rpn -h
usage: rpn [-dhu] [-G N,M] [-c cost_func] [-e target_err] [-f file_name] [-g N,M] [-l transfer]
           [-m factor] [-n min,max] [-o transfer] [-p N,M] [-r file_name] [-s seed] [-t perceptrons]
           [epochs]

Resilient Propagation Neural network
Version 0.01 (built on Dec  3 2020), developed by Dmitry Lyssenko (ldn.softdev@gmail.com)

optional arguments:
 -d             turn on debugs (multiple calls increase verbosity)
 -h             help screen
 -u             round up outputs to integer values
 -G N,M         recursively interconnect neurons N to M
 -c cost_func   cost function [default: Sse]
 -e target_err  convergence target error [default: 0.001]
 -f file_name   file to dump Rpnn brain to [default: rpn.bin]
 -g N,M         grow synapse from neuron N to neuron M
 -l transfer    effectors logistic function [default: Sigmoid]
 -m factor      local minimum trap detection (0: disable) [default: 2]
 -n min,max     input normalization (min=max to disable) [default: -1,+1]
 -o transfer    output neurons logistic function [default: Sigmoid]
 -p N,M         prune synapse at neuron N to neuron M
 -r file_name   file to reinstate Rpnn brain from [default: rpn.bin]
 -s seed        seed for randomizer (0: auto) [default: 0]
 -t perceptrons full mesh topology (enumerated perceptrons) [default: 1,1]

standalone arguments:
  epochs        epochs to run convergence [default: 100000]

available cost functions:
	o Sse
	o Xntropy

available logistic functions:
	o Sigmoid
	o Tanh
	o Tanhfast
	o Relu
	o Softplus
	o Softmax

- parameters N,M are zero based, the index 0 refers to a reserved neuron "the one"
- factor for option -m is multiple of the total count of synapses (weights)

bash $ 
```

### Manual installation
1. Download, or clone the repo
2. compile using _C++14_ or later:
    ```bash
    bash $ c++ -o rpn -Wall -Wextra -std=c++14 -Ofast rpn.cpp
    bash $
    ```

### `rpn` operations
`rpn` operates in two modes:
1. learning mode
2. trained more

#### Learning mode
In learning mode the `rpn` learns from the provided input/targes samples and once the solution is found (`rpn` successfully converges) it dumps its
trained brains into the file (default filename is `rpn.bin`)

Inputs are read line-by-line, each line containing _input and target figures_, so that the number of input figures corresponds to the number of
receptros in the configured topology and the number of target figures corresponds to the number or output neurons. The figures on the input line
should be separated with blank space, optionally with `,` or `=` symbols
(note: there's no semantical significance for separators, so they could be interchanged freely)

For example, say your topology has 2 receptors and 1 output neuron, then any of following input lines are fine:
```
0 0 0
0, 1, 1
1, 0 = 1
1==1,,0
```
The last line though might be confusing, as it still facilitates two inputs (`1`, `1`) and a single output (`0`), so apply your discretion when
using separators.

If `rpn` does not find a solution (fails to converge), then it does not dump its brains into the file (then you should adjust parameters,
e.g.: increase epochs, alter target error, change topology, etc). 

#### Trained mode
To start `rpn` in a trained mode, you need to give it a parameter `-r` followed by the file name where `rpn` brains are (default is `rpn.bin`)
in the trained mode `rpn` accepts the input lines the same way like in the _Learning mode_, only the figures on each line here are inputs only
(no target patterns this time)

#### Hello World!
_Hello World!_ task in the NN is the training of _XOR_ function (it's the simplest task that requires a multi-perceptron to converge).

Topology for the `rpn` can be given using `-t` option followed by the perceptron sizes over the comma. E.g., to train `rpn` for the _XOR_ function,
following topology is required:

        input1 -> R-----H
                   \   / \
                    \ /   \
                     X     O -> output
                    / \   /
                   /   \ /
        input2 -> R-----H
That topology is made of 3 layers:  
  - 1st layer is made of 2 receptors (`R`)  
  - 2nd layer is made of 2 hidden neurons (`H`)  
  - and finally the 3rd layer is made of a single output neuron (`O`).
Thus, it could be expressed to `rpn` as `-t 2,2,1` (note: no spaces between numbers). `rpn` provides a full-mesh synapse connectivity between
layers.

And here we're good to run our first data sample:
```
bash $ <<<"
0, 0 = 1
1, 0 = 0
0, 1 = 0
1, 1 = 1
" rpn -t2,2,1
Rpnn has converged at epoch 17 with error: 0.000299919
bash $ 
```
Now file `rpn.bin` contains the brain dump of the trained pattern and could be reused on the input data:
```
bash $ <<<"
0 0
1 1
0 1
" rpn -u -r rpn.bin
1
1
0
bash $
```
> As you might have noticed, `rpn` was trained for _NOT XOR_ function instead

That shows that the network has learnt the training material properly.


#### Multi-class
The above example illustrates a _binary_ classification, though it's not the only possible type of classification, sometimes tasks reqiure multiple classes.
E.g., the same solution could be explressed as 3 classes:

a) set _class1_ when the inputs are all zero (`0`,`0`)  
b) set _class2_ when the inputs vary (`1`,`0`, or `1`,`0`)  
c) set _class3_ when the inputs are all ones (`1`,`1`)  

This type of classification require setting the logistic of all 3 output neurons to _Softmax_ activation function (default is `Sigmoid` for all neurons)
and the cost function to be _Cross-Entropy_ (default is _Sum Squared Error_ - `Sse`):
```
bash $ <<<"
0,0 = 1 0 0
1,0 = 0 1 0
0,1 = 0 1 0
1,1 = 0 0 1
" rpn -t2,2,3 -o Softmax -c Xntropy
Rpnn has converged at epoch 22 with error: 0.000758367
bash $ 
```
Now, the trained network will display all 3 classes (output neurons):
```
bash $ <<<"
0 0        
1 1        
0 1        
" rpn -ur rpn.bin
1 0 0
0 0 1
0 1 0
bash $ 
```



