
# `Rpnn` (under construction)
This is an idiomatic C++ implementation of _Resilient backprop Neural Network_ with an easy and convenient user interface, 
no dependencies - fully contained implementation.

##### Enhancement requests and/or questions are more than welcome: *ldn.softdev@gmail.com*

#

Resilient backprop is [known to be the fastest](https://en.wikipedia.org/wiki/Rprop) learning NN in the family of backprops, featuring
a number of advantages over the standard backprop mechanism:
- the learning rule is no longer proportional to the size of the gradient, only the sign of the computed gradient matters
  - programmatically it means no need for plugging a derivative of the logistic function (plugging only the logistic will do)
  - it's not prone to [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) that the standard backprop suffers from
- the configuration of the _rprop_ is simple and not as complex and sensitive as the standard backprop's
- this implementation provides auto normalization of outputs and optionally of the inputs 
([why inputs may require normalization](https://github.com/ldn-softdev/Rpnn#default-parameters))
- the framework is fully and easily SERSES'able [link TBU]
- the framework also support multi-class classification (support of _Softmax_ logistic at the output perceptron)
- the framework features a detection mechanism of local minimum traps and bouncing its weights out of the trap - ensures a high probability of convergence
- the framework provides a weight bouncer class capable of finding a better (deeper) minimum in absence of the global one
(by running concurent instances and searching for the deepest local minimum)  

#

<p align="center">
Given right configuration (topology, parameters) and enough resources (cpu cores, time)<br>`Rpnn` guaratees finding the best or near the best solutions
</p>

#

### Content:
1. [cli toy](https://github.com/ldn-softdev/Rpnn#cli-toy)
    * [Manual installation](https://github.com/ldn-softdev/Rpnn#manual-installation)
    * [`rpn` operations](https://github.com/ldn-softdev/Rpnn#rpn-operations)
      * [learning mode](https://github.com/ldn-softdev/Rpnn#learning-mode)
      * [trained mode](https://github.com/ldn-softdev/Rpnn#trained-mode)
    * [`rpn` options and parameters](https://github.com/ldn-softdev/Rpnn#rpn-options-and-parameters)
      * [Default parameters](https://github.com/ldn-softdev/Rpnn#default-parameters)
        * [Topology](https://github.com/ldn-softdev/Rpnn#topology)
        * [Target error](https://github.com/ldn-softdev/Rpnn#target-error)
        * [Inputs normalization](https://github.com/ldn-softdev/Rpnn#inputs-normalization)
        * [Local Minimum traps detection](https://github.com/ldn-softdev/Rpnn#local-minimum-traps-detection)
        * [Cost (error) function](https://github.com/ldn-softdev/Rpnn#Cost-error-function)	
      * [Configuring NN Topology](https://github.com/ldn-softdev/Rpnn#configuring-nn-topology)
        * [Growing and pruning synapses](https://github.com/ldn-softdev/Rpnn#Growing-and-pruning-synapses)
2. Study Examples 
    * [Hello World!](https://github.com/ldn-softdev/Rpnn#hello-world)
    * [Multi-class](https://github.com/ldn-softdev/Rpnn#multi-class)
3. [C++ user interface](...)


## cli toy
This package provides a simple unix cli tool which allows running `Rpnn` from shell and toying with your data easily:
```
bash $ rpn -h
usage: rpn [-adhu] [-G N,M] [-P param] [-S separators] [-b threads] [-c cost_func] [-e target_err]
           [-f file_name] [-g N,M] [-l transfer] [-m factor] [-n min,max] [-o transfer] [-p N,M]
           [-r file_name] [-s seed] [-t perceptrons] [epochs]

Resilient Propagation Neural network (https://github.com/ldn-softdev/Rpnn)
Version 1.02 (built on Jan  5 2021), developed by Dmitry Lyssenko (ldn.softdev@gmail.com)

optional arguments:
 -a             plug in a uniform bouncer (alternative to randomizer)
 -d             turn on debugs (multiple calls increase verbosity)
 -h             help screen
 -u             round up outputs to integer values
 -G N,M         recursively interconnect neurons N to M
 -P param       modify generic parameters (PARAM=x,y,..)
 -S separators  value separators (REGEX) [default: \s,;=]
 -b threads     best local minimum search (0: #threads equals #cores)
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

 - parameters N,M are zero based, the index 0 refers to a reserved neuron "the one"
 - factor for option -m is multiple of the total count of synapses (weights)

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

 generic Rpnn parameters (alterable with -P):
 	o BLM_RDCE [5]
	o DW_FACTOR [1.1618]
	o LMD_PTRN [0.001]
	o MAX_STEP [1000]
	o MIN_STEP [1e-06]
	o NRM_MAX [1]
	o NRM_MIN [-1]

 for further details refer to https://github.com/ldn-softdev/Rpnn
bash $ 
```

### Manual installation
1. Download, or clone the repo
2. compile using _C++14_ or later:
    * _MacOS:_
    ```bash
    bash $ c++ -o rpn -Wall -Wextra -std=c++14 -Ofast rpn.cpp
    bash $
    ```
    * _Linux (relocatable image):_
    ```bash
    bash $ c++ -o rpn -Wall -std=gnu++14 -Ofast -static -Wl,--whole-archive -lrt -pthread -lpthread -Wl,--no-whole-archive rpn.cpp
    bash $
    ```
    
### `rpn` operations
`rpn` operates in two modes:
1. learning mode
2. trained mode

#### Learning mode
In the learning mode `rpn` learns from the provided input-target samples and once the solution is found (`rpn` successfully converges) it dumps its
trained brains into the file (default file is `rpn.bin`, option `-f` steers the output filename)

Training patterns are read line-by-line, each line containing _input and target values_, so that the number of input values corresponds to the
number of receptors in the configured topology, and the number of target values corresponds to the number or output neurons. The values on the input
line should be separated either with blank space, or optionally with `,` or `;`, or `=` symbols (separators could be specified using `-S` option
[REGEX notation])
> note: there's no semantical significance for separators, so they could be interchanged/duplicated freely

For example, say your topology has _2 receptors_ and _1 output neuron_, then any of following input lines are fine:
```
0 0 0
0, 1, 1
1,0 =1
1==1,,0
```
The last line though might be confusing, as it still facilitates two inputs (`1`, `1`) and one single output (`0`), so apply your discretion when
using value separators.

If `rpn` does not find a solution (fails to converge), then it does not dump its brains into the file (then you should adjust parameters,
e.g.: increase epochs, alter target error, change topology, etc) - that applies though only when searching for a global minimum.

#### Trained mode
To start `rpn` in a trained mode, you need to give it a parameter `-r` followed by the file name where `rpn` brains are (default is `rpn.bin`)
in the trained mode `rpn` accepts the input lines the same way like in the _Learning mode_, only the values on each line here are input patterns
(if any other values are present on the same line, those will be ignored)  
> if option `-r` is given the others will be ignored


### `rpn` options and parameters
`rpn` is extensively debuggable, though using debug depth higher than 3 (`-ddd`) is not advisable as it will cause huge dumps on the console

`rpn` has following default parameters when none given:
```
bash $ rpn -d
.configure(), receptors: 1
.configure(), effectors: 1
.configure(), output neurons: 1
.configure(), target error: 0.001
.configure(), normalize inputs: true [-1 to +1]
.configure(), LM trail size: 4
.configure(), cost function: cf_Sse
.configure(), generic parameter BLM_RDCE: 5
.configure(), generic parameter DW_FACTOR: 1.1618
.configure(), generic parameter LMD_PTRN: 0.001
.configure(), generic parameter MAX_STEP: 1000
.configure(), generic parameter MIN_STEP: 1e-06
.configure(), generic parameter NRM_MAX: 1
.configure(), generic parameter NRM_MIN: -1
.configure(), blm (threads) engaged: no
.configure(), bouncer: native
.configure(), randomizer seed: timer (1609875073812804)
.configure(), epochs to run: 100000
.read_patterns_(), start reading training patterns (1 inputs + 1 outputs)...

^Caborted due to user interrupt received: SIGINT (2)
bash $ 
```

#### Default parameters

##### Topology
```
.configure(), receptors: 1
.configure(), effectors: 1
.configure(), output neurons: 1
```
- Number of receptors: 1
- Number of effectors: 1 (effector is a non-receptor neuron)
- Number of output neurons: 1 (output neuron is also effector)  
thus such default topology is expressed as an option `-t 1,1` (there are only 2 neurons in such topology)  
> well, there's one more hidden neuron ("the one") which is always implicitely present and is interconnected to all others

option `-t` lets setting up topology and interconnects adjacent layers (perceptrons) in a full mesh 

#
##### Target error
```
.configure(), target error: 0.001
```
\- option `-e` allows setting the target error for convergence  
> network considers convergence done when the network's global error (i.e. the error across all the output neurons) drops below the target error

Some tasks might not even have a global minimum solution (e.g.: approximations/regressions, or even classifications with a weak correlation),
thus manual adjusting target error (to the higher end) might be required.

Though adjusting target error manually could be tedious and non-efficient, `rpn` provides an automatic way for searching the deepest local minimum
in absense of a global one (see option `-b`)

#
##### Inputs normalization

```
.configure(), normalize inputs: true [-1 to +1]
```
\- _Inputs normalization_ is on by default and could be turned off with option `-n 0,0`, or `-n 1,1` (any combination where `min` and `max` parameters
are the same). Given that the logistic function often is a bounded type (e.g.: `sigmoid`, `tanh`, etc) the faster convergence occurs when input's
_max_ and _min_ values are mapped around the logistic's zero point. Default input normalization values are `-n -1,+1`.

Also, Rpnn limits _delta weight_ step's _min_ and _max_ values to `1.e-6` and `1.e+3` respectively (though such default parameters could be
altered with -P option):
```
#define RPNN_MIN_STEP   1.e-6
#define RPNN_MAX_STEP   1.e+3
```
Thus, very small or very large input values w/o normalization simply won't converge - the input normalization ensures respective resolution precision.

For example, this converges fine with the normalization on (default):
```
bash $ <<<"
1e-5, 1e-5 = 10
1e-5, 2e-5 = 20
2e-5, 1e-5 = 20
2e-5, 2e-5 = 10
" rpn -t2,2,1
Rpnn has converged at epoch 40 with error: 0.000512515
bash $ 
bash $ <<<"2e-5, 2e-5 " rpn -ur rpn.bin
10
bash $ 
```

But with the normalization turned off, it'll fail to find a solution:
```
bash $ <<<"                            
1e-5, 1e-5 = 10
1e-5, 2e-5 = 20
2e-5, 1e-5 = 20
2e-5, 2e-5 = 10
" rpn -t2,2,1 -n0,0
Rpnn could not converge for 100000 epochs (err: 1.00001) - not saving
bash $ 
```

> btw, output normalization is always on and there no way to turn it off: the output logistic functions requirement is always to be a bounded type
(in this tool's implementation, not in `Rpnn` class), thus output normalization helps to have any scale of output parameters

#
##### Local Minimum traps detection
```
.configure(), LM trail size: 4
```
\- the framework provides a way to detect if during the convergence it ends up in the local minimum valley so that it could re-initialize all its
weights and bounce itself out of the local minimum trap.  
That mechanism is facilitated with the tracking the error trail of each epoch's global error. The size of such trail typically is proportional to
the total number of weights in the given topology with the default factor of `-m 4` (i.e. times 4). Though it does not always work optimally and
sometimes requires adjustments (to a shorter facter, e.g. 2 or 3 - to speedup convergence) or a longer one (to ensure a reliable LM detection).
> The mechanism poses a dilemma though: LM trap detection drastically improves chances for a successful converge, but the trail size slows down the
convergence itself (the bigger trail size, the slower training runs) - finding a right balance is the subject of some research for a given task.

#
##### Cost (error) function
```
.configure(), cost function: cf_Sse
```
Default cost function to evaluate convergence (across all the output neurons) is _Sum of Squared Errors_ (`Sse`).
Another cost function is _Cross Entropy_ (`Xntropy`)

Typically _Cross Entropy_ is used togerther with `Softmax` logistic functions of the output neurons.  
-\ to alter the cost function, use `-c Xntropy`

#

```
.configure_rpn(), randomizer seed: timer (1607090033445218)
```
A seed for randomization (weights initializing) is taken from the timer, though for some debugging (or research) purposes it might be required 
running multiple convergences with the same seed, which could be done using option `-s <seed>`

#
option `-u` rounds up output result (in trained mode) to an integer, w/o it `rpn` will display the resulting value (with the achieved accuracy):
```
bash $ <<<"
1e-5, 1e-5
1e-5, 2e-5
2e-5, 1e-5
2e-5, 2e-5
" rpn -r rpn.bin
10.1844
19.9954
19.9816
10.1786
bash $ 
```

#
option `-f <file>` lets dumping trained `rpn` brains (upon a successful convergence) into the file of your choice (default output file is `rpn.bin`)
option `-r <file>` reads and reinstate brains state entirely from the file ready to run the input data 



#### Configuring NN Topology

NN topology could be verified with `-dd` debug depth:
```
bash $ rpn -dd
.configure_rpn(), receptors: 1
.configure_rpn(), effectors: 1
.configure_rpn(), output neurons: 1
.configure_rpn(), target error: 0.001
.configure_rpn(), normalize inputs: true [-1 to 1]
.configure_rpn(), LM trail size: 4
.configure_rpn(), cost function: cf_Sse
.configure_rpn(), randomizer seed: timer (1607090056539971)
.configure_rpn(), epochs to run: 100000
..configure_rpn(), class 'Rpnn'...
   Rpnn::addr(): 0x7ffee229d710
   Rpnn::min_step(): 1e-06
   Rpnn::max_step(): 1000
   Rpnn::dw_factor(): 1.1618
   Rpnn::target_error_: 0.001
   Rpnn::cost_func(): "Sse"
   Rpnn::wbp_: 0x7ffee229d7e8
   Rpnn::epoch_: 0
   Rpnn::terminate_: false
   Rpnn::effectors_start_idx(): 2
   Rpnn::output_neurons_start_idx(): 2
   Rpnn::neurons()[0]: class 'rpnnNeuron'...
      neurons()[0]::addr(): 0x7ffa28407170
      neurons()[0]::host_nn_ptr(): 0x7ffee229d710
      neurons()[0]::is_receptor(): true
      neurons()[0]::transfer_func(): "Sigmoid"
      neurons()[0]::out(): 1
      neurons()[0]::delta(): 0
      neurons()[0]::bp_err(): 0
      neurons()[0]::synapses(): []
      neurons()[0]::inputs_ptr(): nullptr
      neurons()[0]::sum_: 0
   Rpnn::neurons()[1]: class 'rpnnNeuron'...
      neurons()[1]::addr(): 0x7ffa28406cf0
      neurons()[1]::host_nn_ptr(): 0x7ffee229d710
      neurons()[1]::is_receptor(): true
      neurons()[1]::transfer_func(): "Sigmoid"
      neurons()[1]::out(): 1
      neurons()[1]::delta(): 0
      neurons()[1]::bp_err(): 0
      neurons()[1]::synapses(): []
      neurons()[1]::inputs_ptr(): nullptr
      neurons()[1]::sum_: 0
   Rpnn::neurons()[2]: class 'rpnnNeuron'...
      neurons()[2]::addr(): 0x7ffa28406d70
      neurons()[2]::host_nn_ptr(): 0x7ffee229d710
      neurons()[2]::is_receptor(): false
      neurons()[2]::transfer_func(): "Sigmoid"
      neurons()[2]::out(): 1
      neurons()[2]::delta(): 0
      neurons()[2]::bp_err(): 0
      neurons()[2]::synapses()[0]: rpnnSynapse.. host_nn_ptr():0x7ffee229d710, linked_neuron_ptr():0x7ffa28407170, weight():2.23461e-314, delta_weight():2.23464e-314, gradient():2.23464e-314, prior_gradient():0
      neurons()[2]::synapses()[1]: rpnnSynapse.. host_nn_ptr():0x7ffee229d710, linked_neuron_ptr():0x7ffa28406cf0, weight():2.23464e-314, delta_weight():2.23464e-314, gradient():1.72723e-77, prior_gradient():0
      neurons()[2]::inputs_ptr(): nullptr
      neurons()[2]::sum_: 0
   Rpnn::input_sets_: []
   Rpnn::input_normalization()[0]: Norm.. found_min():2.23462e-314, found_max():1, base():-1, range():2
   Rpnn::target_sets_: []
   Rpnn::target_normalization(): []
   Rpnn::output_errors_[0]: 0
   Rpnn::lm_detector(): fifoDeque.. capacity():4, fifo():[]
.run_convergence(), start reading training patterns...

^Caborted due to user interrupt received: SIGINT (2)
bash $ 
```

Neuron synapses provide linkage to other neurons via `linked_neuron_ptr()`, so the topology could be traces down. In every topology there's one hidden
neuron (a.k.a. _"the one"_), that neuron is required for a NN convergence and every effector is linked to that neuron - _"the one"_ is always shown
first neuron in the above output.  
All the other neurons are from user's configuration, i.e.: Neuron with address `0x7f91fbc06cf0` is a receptor (`is_receptor(): true`),
the logistic for a receptor is irrelevant, as receptors only facilitate input (patterns) values access.
   
`Sigmoid` is a default transfer function for all the neurons, though all effectors (and output neurons separately) could be setup using other logistics:
 - `Tanh` - could be used in hidden and output neurons
 - `Tanhfast` - could be used in hidden and output neurons
 - `Relu` - could be used only in hidden neurons
 - `Softplus` - could be used only in hidden neurons 
 - `Softmax` - сould be used in output neurons only.

Setting hidden effectors to a non-bound logistic (`Relu`, `Softmax`) requires understanding of the implications. On one hand it may result in a very fast
convergence (if weights initialization is favorable, or multi-dimensional plane of _f(x) = error(weights)_ is favorable for given task):
```
bash $ <<<"
0 0 = 0
1 0 = 1
0 1 = 1
1 1 = 0
" rpn -t2,2,1 -l Relu -o Sigmoid 
Rpnn has converged at epoch 12 with error: 0.000974536
```
However, convergence of hidden neurons on `Relu` may (and most likely will) kick the weights way afar the global region, resulting in wandering often 
around local minimums: 
```
bash $ <<<"
0 0 = 0
1 0 = 1
0 1 = 1
1 1 = 0
" rpn -t2,2,1 -l Relu -o Sigmoid 
Rpnn has converged at epoch 97363 with error: 0.000343371
bash $ 
```

##### Growing and pruning synapses
If a full-mesh connectivity between neuron layers is not enough and you want to add more connections (typically recursive links), then it's possible 
to do it via options `-g`, `-G`:

- `-g N,M` allows adding a single synapse for neurons N to M's output (i.e., a connection from M to N considering forward signal flow)
- `-G N,M` this will ensure that between neurons N and M all recursive synapses added
> base of values M and N is the same as in debug output - it's all zero based, but the first neuron is always reserved) 


option `-p N,M` allows pruning a single synapse at the neuron N for the (address of) neuron M

### 
#### Hello World!
_"Hello World!"_ task in the NN is the training of _XOR_ function (it's the simplest task that requires a multi-perceptron to converge).

Topology for the `rpn` can be given using `-t` option followed by the perceptron sizes over the comma. E.g., to train `rpn` for the _XOR_ function,
following topology is required:

    
        input1 -> R---H
                   \ / \
                    X   O -> output
                   / \ /
        input2 -> R---H
    
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

This type of classification require setting the logistic of all 3 output neurons (one output neuron per class) to _Softmax_ activation function (default is
`Sigmoid` for all neurons) and the cost function to be _Cross-Entropy_ (default is _Sum of Squared Errors_ - `Sse`):
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




