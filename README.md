
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
        * [Generic parameters](https://github.com/ldn-softdev/Rpnn#generic-parameters)
        * [Searching best local minimum](https://github.com/ldn-softdev/Rpnn#finding-best-local-minimum)
        * [Alternative weight bouncer](https://github.com/ldn-softdev/Rpnn#alternative-weight-bouncer)
        * [Seed for randomizer](https://github.com/ldn-softdev/Rpnn#seed-for-randomizer)
        * [Epochs to run](https://github.com/ldn-softdev/Rpnn#epochs-to-run)
      * [Configuring NN Topology](https://github.com/ldn-softdev/Rpnn#configuring-nn-topology)
      * [Other supported options](https://github.com/ldn-softdev/Rpnn#other-supported-options)
        * [Growing and pruning synapses](https://github.com/ldn-softdev/Rpnn#Growing-and-pruning-synapses)
2. [Study Examples](https://github.com/ldn-softdev/Rpnn#study-examples) 
    * [Hello World!](https://github.com/ldn-softdev/Rpnn#hello-world)
    * [Multi-class](https://github.com/ldn-softdev/Rpnn#multi-class)
    * [Classification as probability](https://github.com/ldn-softdev/Rpnn#classification-as-probability)
3. [C++ user interface](...)


## cli toy
This package provides a simple unix cli tool which allows running `Rpnn` from shell and toying with your data.
This cli tool is probably the easiest way to introduce yourself to classical NN (no programming skill required),
however it's also could be used in all the areas where backprop is applicable (classification, regressions, aproximations,
prediction, etx):
```
bash $ rpn -h
usage: rpn [-adhu] [-G N,M] [-P param] [-S separators] [-b threads] [-c cost_func] [-e target_err]
           [-f file_name] [-g N,M] [-l transfer] [-m factor] [-n min,max] [-o transfer] [-p N,M]
           [-r file_name] [-s seed] [-t perceptrons] [epochs]

Resilient Propagation Neural network (https://github.com/ldn-softdev/Rpnn)
Version 1.03 (built on Jan  6 2021), developed by Dmitry Lyssenko (ldn.softdev@gmail.com)

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
> if option `-r` is given the others will be ignored (save `-u`)

Option `-u` instructs `rpn` to round up all the outputs (if real numbers used) to an interger part (in the trainded mode, of course); in case if
outputs are symbolic enumerations, then show real convertion numbers instead 



### `rpn` options and parameters
`rpn` is extensively debuggable, though using debug depth higher than `3` (`-ddd`) is not advisable as it will cause huge dumps on the console

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
- Number of receptors: `1`
- Number of effectors: `1` (effector is a non-receptor neuron)
- Number of output neurons: `1` (output neuron is also effector)  
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
the total number of weights in the given topology with the default factor of `-m 4` (i.e. times `4`). Though it does not always work optimally and
sometimes requires adjustments (to a shorter factor, e.g. `2` or `3` - to speedup convergence) or a longer one (to ensure a reliable LM detection).
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
\- to alter the cost function, use `-c Xntropy`

##### Generic parameters
```
.configure(), generic parameter BLM_RDCE: 5
.configure(), generic parameter DW_FACTOR: 1.1618
.configure(), generic parameter LMD_PTRN: 0.001
.configure(), generic parameter MAX_STEP: 1000
.configure(), generic parameter MIN_STEP: 1e-06
.configure(), generic parameter NRM_MAX: 1
.configure(), generic parameter NRM_MIN: -1
```

`Rpnn` framework has a few default parameters, which normally do not require much of fiddling. However, any of those also could be modified usign `-P` option.  
Say, you want to try another input normalization range, e.g.: `-50, +50`. Then either way will do it:  
 - `-P NRM_MAX=50 -P NRM_MIN:50` - separator between parameter name and value could be either `:` or `=`,
 - `-P NRM_MAX:50,-50` - if multiple values given, then those applied on the respective parameters in the displayed order starting from the given one (the order
 is the same on the help screen `-h`)

Anoter example, it's possible to alter all the values in one go, like this:  
`rpn -P BLM_RDCE:'15, 1.5, 0.01, 1e+5, 1e-10, 20, -20' ...`
> note: quotes are used because of the spaces separating parameters (i.e., to dodge shell interpolation)

Description:
 * `BLM_RDCE` - reduce factor for _best local minimum_ search mode - varies from `1` (exclusively) to to any higher number - the higher number,
 the harder `Rpnn` will try finding the best (deepest) LM (i.e. more attempt will be made). The factor is exponential though, so numbers
 above `10` might already severily impact resolutino time (deppends on many other factros too - size of the topology, size of the `LMD_PTRN`, etc)
 * `DW_FACTOR` - a momentum factor to increase synapse's _delta weights_ if gradient sign did not change. This factor also has an exponential effect
 and setting it to too big values may result that `Rpnn` will be overshooting minimums too frequently. Setting it to values lower or too close to `1`
 does not make sense either - slow momentum will result in slower convergence as well make `Rpnn` suffer from
 [_vanishing gradient_](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) problem
 * `LMD_PTRN` - a percentage factor for _Local Minimum trap detection_ mechanism - how accurately the mechanism will try recognizing error
 looping behavior. The smaller value the more accurate detection is, the higher value provides more coarse looping detection (which might
 result in an earlier detection but also a false-positive detection too)
 * `MAX_STEP`
 * `MIN_STEP` - these two provide upper and lower capping for _delta weight_ in synapses
 * `NRM_MAX`
 * `NRM_MIN` - these two provide _max_, _min_ normalization boundary for input pattern values normalization. Setting them to the same (any) value
 results in disabling input normalization (which is not advisable)

#
##### Searching best local minimum
```
.configure(), blm (threads) engaged: no
```
By default `rpn` will try finding a global minimum (convergence point, where the global error is close to zero - below target's error) and if it fails
to find one - it won't save it's brains at the end. However, most of the tasks where backprops apply _do not have global minimum_. There the solution
sounds like this: find the deepest possible local minimum.

The approach `Rpnn` takes in such case is running and tracking multiple convergences until local minimum is detected and then picking the convergence
result with the smallest error (deepest found LM). It's implemented by running multiple instances of configured NN concurently (each instance will be
run multiple times too).

To enable such mode (a.k.a. _BLM search_) is to give option `-b` followed by the number of threads. if number `0` given (`-b0`), then the number of threads
will correspondds to the maximum number of supported hardware threads (#cores times #threads per core).

The target error (`-e`) in such case serves as a twofold exit criteria:
- if NN able to converge below the target error (i.e. a global minimum is found)
- if adjusted internal goal error's delta (with found so far best minimum) drops below target error
> adjustment of the _goal error_ occurs every time when any `Rpnn` instance detects LM and LM's error is worse than found so far best minimum - then
the _goal error_ is adjusted by `BLM_RDCE` factor).

Because _BLM search_ is suitable for finding even the global minimum, it does not hurt to run `rpn` always with `-b` option.

#
##### Alternative weight bouncer
```
.configure(), bouncer: native
```
By default `rpn` will use a simple randomizer to update its weights before starting a new convergence (and when bouncing itself out of LM trap).
For a reference there's another bouncer provided: it builds first a limited set of uniform weights distributions (acorss all the weights) and then
uses them up (in a random order) untill all exhausted - that method is more deterministic than random weight bouncing in terms that it tries only a certain
prescribed sets of weights distributions.

To plug-in the alternative weight update function give `-a` option.

#
##### Seed for randomizer
```
.configure(), randomizer seed: timer (1609875073812804)
```
A seed for randomization (weights initializing) is taken from the timer, though for some debugging (or research) purposes it might be required
running multiple convergences with the same seed, which could be done using option `-s <seed>`
> note: though setting the same seed won't provide deterministic behavior in the _BLM_ (option `-b`), due to inability to control concurent convergence

#
##### Epochs to run
```
.configure(), epochs to run: 100000
```
When NN tries once all the given input patterns and learns from them (by back-propagating the resulting error adjusting its weights towards the closer match)
it's called an _epoch_. Because the _error plane_ is always smoothly differentiable, it inevitably leads towards the minimum either local or global
(thanks to the _learning rule_), however, it certainly requires an _unknown_ number of such itterations (epochs). `Rpnn` typically reach the minimums quite
quickly and then (if LMD is enabled) will try bouncing itself out of the found LM and will descend into another one.  
To cap the number of such itterations the number of ephoch sets the limit. The maximum number of epochs is given as the only standalone attribute to Rpnn (if
omitted, then default number `100000` is used).

The above mostly applies when _BLM_ search is not engaged, otherwise, there the number of attempts is rather limited by the number of LM found (which is a
combinations of 2 factors `BLM_RDCE` and _target error_), thogh setting ephoch number to a very shallow value (even in _BLM_) is not advisable, as it may
result in a premature end of convergence even before reaching the LM.


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
first in the above output  
All the other neurons are from user's configuration, i.e.: Neuron with address `0x7f91fbc06cf0` is a receptor (`is_receptor(): true`),
the logistic for a receptor is irrelevant, as receptors only facilitate input (patterns) values access.

`Sigmoid` is a default transfer function for all the neurons, though all effectors (and output neurons separately) could be setup using other logistics:
 - `Tanh` - could be used in hidden and output neurons
 - `Tanhfast` - could be used in hidden and output neurons
 - `Relu` - could be used only in hidden neurons
 - `Softplus` - could be used only in hidden neurons
 - `Softmax` - сould be used in output neurons only.

All the neuron's logistic function could be setup using `-l` followed the name of the function  
Output neuron's only logistics could be setup using `-o` option followed the name of the function

Setting hidden effectors to a non-bound logistic (`Relu`, `Softplus`) requires understanding of the implications. On one hand it may result in a very fast
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

# 
#### Other supported options

- `-f <file>` - lets dumping trained `rpn` brains (upon a successful convergence) into the file of your choice (default output file is `rpn.bin`)
- `-r <file>` - starts `rpn` in the trainde mode - reads and reinstate brains state entirely from the file, ready to read & run the input data
- `-S <separators>` - allows specifying a list of separators used for the input lines. Default are `\s,;=` (note the REGEX notation)


### Study Examples
Let's review a few of academical and real wordls examples

#### Hello World!
_"Hello World!"_ task in the NN is the training of _XOR_ function - it's the simplest task that requires a multi-perceptron to converge (why is that - is
out of scope of this tutorial, but you can easily google up
[`The XOR Problem in Neural Networks`](https://www.google.com/search?hl=en&q=The+XOR+Problem+in+Neural+Networks&btnG=Google+Search)).

Topology for the `rpn` can be given using `-t` option followed by the perceptron sizes over the comma. E.g.: to train `rpn` for the _XOR_ function,
following topology is required:

        input1 -> R---H
                   \ / \
                    X   O -> output
                   / \ /
        input2 -> R---H

That topology is made of 3 layers:
  - 1st layer is made of 2 receptors (`R`) - first layer is always a layer of receptors
  - 2nd layer is made of 2 hidden neurons (`H`)
  - and finally the 3rd layer is made of a single output neuron (`O`)

Thus, it could be expressed as `rpn -t 2,2,1` (note: no spaces between numbers). `rpn` provides a full-mesh synapse connectivity between
all adjacent layers.

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
E.g.: the same solution could be expressed as 3 classes:

a) set _class1_ when the inputs are all zero (`0`,`0`)  
b) set _class2_ when the inputs vary (`1`,`0` or `1`,`0`)  
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

#### Classification as probability
However, like it was mentioned before, it's quite rarre when the problem solution has a global minimum. Even for classification types of tasks the
real world input data may contain duplicate and even confilicting data, Let's consider this input set (_NOT XOR_ with noisy inputs):
```
bash $ <<<"
0, 0 = 1
1, 0 = 0
0, 1 = 0
1, 1 = 1
1, 1 = 0
1, 1 = 1
" rpn -t2,2,1
Rpnn could not converge for 100000 epochs (err: 0.636509) - not saving
bash $ 
```
The last tree lines in the training data set represent the issue: with 66% probability it indicates that pattern `1, 1` produces the `1` signal, while
with 33% probability it's `0` signal.

So, it's time to engage _BLM_ search:
```
bash $ <<<"
0, 0 = 1
1, 0 = 0
0, 1 = 0
1, 1 = 1
1, 1 = 0
1, 1 = 1
" rpn -t2,2,1 -b0
Rpnn found best local minimum, combined total epochs 2659 with error: 0.333333
bash $ 
```
Let's see how it learned the problem:
```
bash $ <<<"
0, 0
1, 0
1, 1
" rpn -ur rpn.bin
1
0
1
bash $ 
```
\- as you see it resolved the problem chosing most probable outcome for the confilicint pattern, but what's the actual number for the problematic pattern?
```
bash $ <<<'1, 1' rpn -r rpn.bin
0.666659
bash $ 
```
- it found a LM where the error corresponds to the probability of signal occurrence in the input. The pattern `1, 1` results in 0.66,6659% probabiliy
(of the output signal occurrence) in this case!

















