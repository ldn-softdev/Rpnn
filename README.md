# `Rpnn` (under construction)
This is an idiomatic C++ implementation of _Resilient backprop Neural Network_ with an easy and convenient user interface.  
No dependencies, fully contained implementation.

##### Enhancement requests and/or questions are more than welcome: *ldn.softdev@gmail.com*

#

Resilient backprop is [known to be the fastest](https://en.wikipedia.org/wiki/Rprop) learning NN in the family of backprops, featuring
a number of advantages over the standard backprop mechanism:
- the learning rule is no longer proportional to the size of the gradient, only the sign of the computed gradient matters
  - programmatically it means no need for plugging a derivative of the logistic function (plugging only the logistic will do)
  - it's not prone to [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) that the standard backprop suffers from
- the configuration of the _rprop_ is simple and not as complex and sensitive as the standard backprop's
- this implementation provides auto normalization of outputs and optionally of the inputs (why inputs might need normalization [link TBU])
- the framework is fully and easily SERSES'able [link TBU]
- the framework also support multi-class classification (support of _Softmax_ logistic at the output perceptron)
- the framework features a detection mechanism of local minimum traps and bouncing its weights out of the trap - ensures a high probability of convergence 

### Content:
1. [cli toy](https://github.com/ldn-softdev/jtc#cli-toy)
    * [Manual installation](...)
    * [`rpn` operations](...)
      * [learning mode](...)
      * [trained mode](...)
      * [Hello World!](...)
      * [Multi-class](...)
    * [`rpn` options and parameters](...)
      * [Defaut parameters](...)
      * [Configuring NN Topology]
2. [C++ user interface](...)


## cli toy
This package provides a simple unix cli tool which allows running `Rpnn` from shell and toying with your data easily:
```
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
2. trained mode

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
_"Hello World!"_ task in the NN is the training of _XOR_ function (it's the simplest task that requires a multi-perceptron to converge).

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
```bash
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
```bash
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
The above example illustrates a _binary_ classification, though it's not the only possible type of classification, sometimes tasks require multiple classes.
E.g., the same solution could be expressed as 3 classes:

a) set _class1_ when the inputs are all zero (`0`,`0`)  
b) set _class2_ when the inputs vary (`1`,`0`, or `1`,`0`)  
c) set _class3_ when the inputs are all ones (`1`,`1`)  

This type of classification require setting the logistic of all 3 output neurons to _Softmax_ activation function (default is `Sigmoid` for all neurons)
and the cost function to be _Cross-Entropy_ (default is _Sum Squared Error_ - `Sse`):
```bash
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
```bash
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


### `rpn` options and parameters
`rpn` is extensively debuggable, though using debug depth higher than 3 (`-ddd`) is not advisable as it will cause huge dumps on the console

`rpn` has following default parameters when none given:
```
bash $ rpn -d
.configure_rpn(), receptors: 1
.configure_rpn(), effectors: 1
.configure_rpn(), output neurons: 1
.configure_rpn(), target error: 0.001
.configure_rpn(), normalize inputs: true
.configure_rpn(), LM trail size: 4
.configure_rpn(), cost function: cf_Sse
.configure_rpn(), randomizer seed: timer (1607022081931188)
.configure_rpn(), epochs to run: 100000
.run_convergence(), start reading training patterns...

^Caborted due to user interrupt received: SIGINT (2)
bash $ 
```

#### Defaut parameters

- Number of receptors 1
- Number of effectors 1 (effector is a non-receptor neuron)
- Number of output neurons 1 (output neuron is also effector)
thus such default topology is expressed as an option `-t 1,1` (there are only 2 neurons in such topology)

```
.configure_rpn(), target error: 0.001
```
\- option `-e` allows setting the target error for convergence. Some tasks might not even have global minimum solutions (typically it'll be
function approximations/regressions) thus adjusting target error (to the higher end) might be required.
> next version of the framework will have an option of finding the deepest local minimum in absence of a global one (i.e. the manual weight adjustments
won't be required)

```
.configure_rpn(), normalize inputs: true
```
\- Inputs normalization is on by default and could be turned off with option `-n 0,0`, or `-n 1,1` (any combination where `min` and `max` parameters
are the same). Given that often the logistic function are bounded type (`sigmoid`, `tanh`, etc) the faster convergence occurs when input's max and min
values are mapped around logistic's zero point. Default input normalization values are `-n -1,+1`.

Also, Rpnn limits _delta weight_ to the minimal and maximal values `1.e-6` and `1.e+4` respectively:
```
#define RPNN_MIN_STEP   1.e-6
#define RPNN_MAX_STEP   1.e+4
```
Thus, very small or very large input values simply won't converge, the input normalization ensures respective resolution precision.
> next version of the framework will provide an option to alter such parameters

```
.configure_rpn(), LM trail size: 4
```
\- the framework provides a way to detect if during the convergence it ends up in the local minimum and re-initialize all the weights bouncing itself
out of the local minimum trap. That mechanism is facilitated with the recording the error trail of each epoch's global error. The size of such trail
typically is proportional to the total number of weights in the given topology with the default factor of `-m 2`. Though it does not always work and
sometimes a longer trail needs to be tracked.
> The mechanism poses a dilemma though: LM trap detection drastically improves chances for a successful converge, but the trail size slows down the
convergence itself (the bigger trail size, the slower training runs) - finding a right balance is the subject of some research for a given task.

```
.configure_rpn(), cost function: cf_Sse
```
Default cost function to evaluate convergence (across all the output neurons) is _Squared Sum Erros_ (`Sse`).
Another cost function is _Cross Entropy_ (`Xntropy`)

```
.configure_rpn(), randomizer seed: timer (1607022081931188)
```
A seed for randomization (weights initializing) is taken from the timer, though for some debugging (or research) purposes it might require running
the convergence with the same seed, which could be done using option `-s 1607022081931188`


#### Configuring NN Topology

NN topology could be verified with `-dd` debugging depth:
```
bash $ rpn -dd
.configure_rpn(), receptors: 1
.configure_rpn(), effectors: 1
.configure_rpn(), output neurons: 1
.configure_rpn(), target error: 0.001
.configure_rpn(), normalize inputs: true
.configure_rpn(), LM trail size: 4
.configure_rpn(), cost function: cf_Sse
.configure_rpn(), randomizer seed: timer (1607027167962425)
.configure_rpn(), epochs to run: 100000
..configure_rpn(), class 'Rpnn'...
   Rpnn::addr(): 0x7ffee2b85bb0
   Rpnn::min_step(): 1e-06
   Rpnn::max_step(): 10000
   Rpnn::dw_factor(): 1.1618
   Rpnn::target_error_: 0.001
   Rpnn::cost_func(): "Sse"
   Rpnn::wbp_: 0x7ffee2b85c88
   Rpnn::epoch_: 0
   Rpnn::terminate_: false
   Rpnn::effectors_start_idx(): 2
   Rpnn::output_neurons_start_idx(): 2
   Rpnn::neurons()[0]: class 'rpnnNeuron'...
      neurons()[0]::addr(): 0x7f91fbc07170
      neurons()[0]::host_nn_ptr(): 0x7ffee2b85bb0
      neurons()[0]::is_receptor(): true
      neurons()[0]::transfer_func(): "Sigmoid"
      neurons()[0]::out(): 1
      neurons()[0]::delta(): 0
      neurons()[0]::bp_err(): 0
      neurons()[0]::synapses(): []
      neurons()[0]::inputs_ptr(): nullptr
      neurons()[0]::sum_: 0
   Rpnn::neurons()[1]: class 'rpnnNeuron'...
      neurons()[1]::addr(): 0x7f91fbc06cf0
      neurons()[1]::host_nn_ptr(): 0x7ffee2b85bb0
      neurons()[1]::is_receptor(): true
      neurons()[1]::transfer_func(): "Sigmoid"
      neurons()[1]::out(): 1
      neurons()[1]::delta(): 0
      neurons()[1]::bp_err(): 0
      neurons()[1]::synapses(): []
      neurons()[1]::inputs_ptr(): nullptr
      neurons()[1]::sum_: 0
   Rpnn::neurons()[2]: class 'rpnnNeuron'...
      neurons()[2]::addr(): 0x7f91fbc06d70
      neurons()[2]::host_nn_ptr(): 0x7ffee2b85bb0
      neurons()[2]::is_receptor(): false
      neurons()[2]::transfer_func(): "Sigmoid"
      neurons()[2]::out(): 1
      neurons()[2]::delta(): 0
      neurons()[2]::bp_err(): 0
      neurons()[2]::synapses()[0]: rpnnSynapse.. host_nn_ptr():0x7ffee2b85bb0, linked_neuron_ptr():0x7f91fbc07170, weight():2.23001e-314, delta_weight():2.23003e-314, gradient():2.23003e-314, prior_gradient():0
      neurons()[2]::synapses()[1]: rpnnSynapse.. host_nn_ptr():0x7ffee2b85bb0, linked_neuron_ptr():0x7f91fbc06cf0, weight():2.23003e-314, delta_weight():2.23003e-314, gradient():2, prior_gradient():0
      neurons()[2]::inputs_ptr(): nullptr
      neurons()[2]::sum_: 0
   Rpnn::input_sets_: []
   Rpnn::nis_[0]: Norm.. found_min():6.95327e-310, found_max():1.38833e-309, base():-1, range():2
   Rpnn::target_sets_: []
   Rpnn::nts_: []
   Rpnn::output_errors_[0]: 0
   Rpnn::lm_detector(): fifoDeque.. capacity():4, fifo():[]
.run_convergence(), start reading training patterns...

^Caborted due to user interrupt received: SIGINT (2)
bash $ 
```

Neurons synapses prove linkage to other neurons via `linked_neuron_ptr()`, so the topology could be traces down. In every topology there's one hidden
neuron (a.k.a. "the one"), that neuron is required for a NN convergence and every effector is linked to that neuron - it's listed as the very first
neuron in the above output.  
All the other neurons are from user's configuration, i.e.: Neuron with address `0x7f91fbc06cf0` is a receptor (`is_receptor(): true`),
the logistic for receptor is irrelevant, as receptors only facilitate input values (patterns) access.
   
`Sigmoid` is a default transfer function for all the neurons, though all effectors (and output neurons separately) could be setup using other logistics:
 - `Tanh` - could be used in hidden and output neurons
 - `Tanhfast` - could be used in hidden and output neurons
 - `Relu` - could be used only in hidden neurons
 - `Softplus` - could be used only in hidden neurons 
 - `Softmax` - could be used in hidden and output neurons, though for hidden neurons it's slower then `Relu`.

Setting hidden effectors to a non-bound logistic (e.g.: `Relu`) requires understanding of the implications. On one hand it may result in a very fast convergence
(if weights initialization is favorable, or multi-dimensional plane of f(x) = error(weights) is favorable for given task):
```
bash $ <<<"
0 0 0
1 0 1
0 1 1
1 1 0
" rpn -t2,2,1 -l Relu -o Sigmoid 
Rpnn has converged at epoch 12 with error: 0.000974536
```
Bur convergence of hidden neurons on `Relu` may (and most likely will) kick the weight way off the global region, resulting in wandering often around local minimals: 
```
bash $ <<<"
0 0 0
1 0 1
0 1 1
1 1 0
" rpn -t2,2,1 -l Relu -o Sigmoid 
Rpnn has converged at epoch 97363 with error: 0.000343371
bash $ 
```
