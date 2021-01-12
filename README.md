
# `Rpnn` (under construction)
This is an idiomatic C++ implementation of _Resilient backprop Neural Network_ with an easy and convenient user interface; 
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
- the framework is fully and easily SERDES'able [link TBU]
- the framework also support multi-class classification (support of _Softmax_ logistic at the output perceptron)
- the framework features a detection mechanism of local minimum traps and bouncing its weights out of the trap - ensures a high probability of convergence
- the framework provides a weight bouncer class capable of finding a better (deeper) minimum in absence of the global one
(by running concurrent instances and searching for the deepest local minimum)  

#

<p align="center">
Given right configuration (topology, parameters) and enough resources (cpu cores, time)<br>`Rpnn` guarantees finding the best or near the best solutions
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
        * [Searching best local minimum](https://github.com/ldn-softdev/Rpnn#searching-best-local-minimum)
        * [Alternative weight bouncer](https://github.com/ldn-softdev/Rpnn#alternative-weight-bouncer)
        * [Seed for randomizer](https://github.com/ldn-softdev/Rpnn#seed-for-randomizer)
        * [Epochs to run](https://github.com/ldn-softdev/Rpnn#epochs-to-run)
      * [Configuring NN Topology](https://github.com/ldn-softdev/Rpnn#configuring-nn-topology)
        * [Growing and pruning synapses](https://github.com/ldn-softdev/Rpnn#Growing-and-pruning-synapses)
      * [Other supported options](https://github.com/ldn-softdev/Rpnn#other-supported-options)
2. [Study Examples](https://github.com/ldn-softdev/Rpnn#study-examples) 
    * [Hello World!](https://github.com/ldn-softdev/Rpnn#hello-world)
    * [Enumerated patterns](https://github.com/ldn-softdev/Rpnn#enumerated-patterns)
    * [Multi-class](https://github.com/ldn-softdev/Rpnn#multi-class)
    * [Classification as probability](https://github.com/ldn-softdev/Rpnn#classification-as-probability)
    * [Couple classification examples from internet](https://github.com/ldn-softdev/Rpnn#couple-classification-examples-from-internet)
      * [Iris classification](https://github.com/ldn-softdev/Rpnn#iris-classification)
      * [Car Evaluation](https://github.com/ldn-softdev/Rpnn#car-evaluation)
3. [C++ class user interface](https://github.com/ldn-softdev/Rpnn#c-class-user-interface)
    * [Essential SYNOPSIS](https://github.com/ldn-softdev/Rpnn#essential-synopsis)
      * [Topology methods](https://github.com/ldn-softdev/Rpnn#topology-methods)


## cli toy
This package provides a simple unix cli tool which allows running `Rpnn` from shell and toying with your data.
This cli tool is probably the easiest way to introduce yourself to classical NN (no programming skills required),
however it's also could be used in all the areas where backprop is applicable (classification, regressions, approximations,
prediction, etc):
```
bash $ rpn -h
usage: rpn [-adhu] [-G N,M] [-P param] [-S separators] [-b threads] [-c cost_func] [-e target_err]
           [-f file_name] [-g N,M] [-l transfer] [-m factor] [-n min,max] [-o transfer] [-p N,M]
           [-r file_name] [-s seed] [-t perceptrons] [epochs]

Resilient Propagation Neural network (https://github.com/ldn-softdev/Rpnn)
Version 1.04 (built on Jan  7 2021), developed by Dmitry Lyssenko (ldn.softdev@gmail.com)

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
0, 1,  1
1,0 =1
1==1,,0
```
The last line though might be confusing, as it still facilitates two inputs (`1`, `1`) and one single output (`0`), so apply your discretion when
using value separators.

If `rpn` does not find a solution (fails to converge), then it does not dump its brains into the file (then you should adjust parameters,
e.g.: increase epochs, alter target error, change topology, etc) - that applies though only when searching for a global minimum - when engaging
[_BLM_ search](https://github.com/ldn-softdev/Rpnn#searching-best-local-minimum) NN always converges with some degree of success.

#### Trained mode
To start `rpn` in a trained mode, you need to give it a parameter `-r` followed by the file name where `rpn` brains are (default is `rpn.bin`)
in the trained mode `rpn` accepts the input lines the same way like in the _Learning mode_, only the values on each line here are input patterns
(if any other values are present on the same line, those will be ignored)  
> when option `-r` is given the others will be ignored (save `-u`)

Option `-u` instructs `rpn` to round up all the outputs (if real numbers used) to an integer part (in the trained mode, of course); in case if
outputs are symbolic enumerations, then show real conversion numbers instead 



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
- Number of receptors: `1` (receptor is a neuron w/o synapses, facing user inputs)
- Number of effectors: `1` (effector is a non-receptor neuron)
- Number of output neurons: `1` (output neuron is also an effector)  
thus such default topology is expressed as an option `-t 1,1` (there are only 2 neurons in such topology)  
> well, there's one more hidden neuron ("the one") which is always implicitly present and is interconnected to all others

option `-t` lets setting up topology and interconnects adjacent layers (perceptrons) in a full mesh

#
##### Target error
```
.configure(), target error: 0.001
```
\- option `-e` allows setting the target error for convergence  
> network considers convergence done when the network's global error (i.e., the error across all the output neurons) drops below the target error

Some tasks might not even have a global minimum solution (e.g.: approximations/regressions, or even classifications with a weak correlation),
thus manual adjusting target error (to the higher end) might be required.

Though adjusting target error manually could be tedious and non-efficient, `rpn` provides an automatic way for searching the deepest local minimum
in absence of a global one (see option [`-b`](https://github.com/ldn-softdev/Rpnn#searching-best-local-minimum))

#
##### Inputs normalization

```
.configure(), normalize inputs: true [-1 to +1]
```
\- _Inputs normalization_ is on by default and could be turned off with option `-n 0,0`, or `-n 1,1` (any combination where `min` and `max` parameters
are the same). Given that the logistic function often is a bounded type (e.g.: `sigmoid`, `tanh`, etc) the faster convergence occurs when input's
_max_ and _min_ values are mapped around the logistic's zero point. Default input normalization values are `-n -1,+1`.

Also, Rpnn limits _delta weight_ step's _min_ and _max_ values to `1.e-6` and `1.e+3` respectively (though such default parameters could be
altered with [`-P`](https://github.com/ldn-softdev/Rpnn#generic-parameters) option):
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
That mechanism is facilitated with the tracking the error trail of each epoch's global error. The size of such trail is typically proportional to
the total number of weights in a given topology with the default factor of `-m 4` (i.e., times `4`). Though it does not always work optimally and
sometimes requires adjustments (to a shorter factor, e.g.: `2` or `3` - to speedup convergence) or a longer one (to ensure a more reliable LM detection).
> The mechanism poses a dilemma though: LM trap detection drastically improves chances for a successful converge, but the trail size slows down the
convergence itself (the bigger trail size, the slower training runs) - finding a right balance is the subject of some research for a given problem solution.

Setting trail size to zero (`-m0`) disables LM detection (and also will render
[_BLM_ search](https://github.com/ldn-softdev/Rpnn#searching-best-local-minimum) ineffective)

#
##### Cost (error) function
```
.configure(), cost function: cf_Sse
```
Default cost function to evaluate convergence (across all the output neurons) is _Sum of Squared Errors_ (`Sse`).
Another cost function is _Cross Entropy_ (`Xntropy`)

Typically, _Cross Entropy_ is used together with `Softmax` logistic functions of the output neurons.  
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

`Rpnn` framework has a few default parameters, which normally do not require much of fiddling. However, any of those also could be modified using `-P` option.  
Say, you want to try another input normalization range, e.g.: `-50, +50`. Then either way will do it:  
 - `-P NRM_MAX=50 -P NRM_MIN:50` - separator between parameter name and value could be either `:` or `=`,
 - `-P NRM_MAX:50,-50` - if multiple values given, then those applied on the respective parameters in the displayed order starting from the given one (the order
 is the same on the help screen [`-h`](https://github.com/ldn-softdev/Rpnn#cli-toy))

For example, it's possible to alter all the values in one go, like this:  
`rpn -P BLM_RDCE:'15, 1.5, 0.01, 1e+5, 1e-10, 20, -20' ...`
> note: quotes are used because of the spaces separating parameters (i.e., to dodge shell interpolation)

Description:
 * `BLM_RDCE` - reduce factor for _best local minimum_ search mode - varies from `1` (exclusively) to to any higher number - the higher number,
 the harder `Rpnn` will try finding the best (deepest) LM (i.e., more attempt will be made). The factor is exponential though, so numbers
 above `10` might already severely impact resolution time (depends on many other factors too - size of the topology, size of the `LMD_PTRN`, etc)
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
to find one - it won't save its brains at the end. However, most of the tasks where backprops may apply _do not have global minimum_ at all.
There the solution sounds like this: find the deepest possible _local minimum_.

The approach `Rpnn` takes in such case is running and tracking multiple convergences until local minimum is detected and then picking the convergence
result with the smallest error (deepest found LM). It's implemented by running multiple instances of configured NN concurrently (each instance will be
run multiple times too).

To enable such mode (a.k.a. _BLM search_) is to give option `-b` followed by the number of threads. if number `0` given (`-b0`), then the number of threads
will correspond to the maximum number of supported hardware threads (#cores times #threads per core).

The target error (`-e`) in such case serves as a twofold exit criteria:
- if NN able to converge below the target error (i.e., a global minimum is found)
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
For a reference there's another bouncer provided: it builds first a limited set of uniform weights distributions (across all the weights) and then
uses them up (in a random order) until all exhausted - that method is more deterministic than random weight bouncing in terms that it tries only a certain
prescribed sets of weights distributions.

To plug the alternative weight update function give `-a` option.

#
##### Seed for randomizer
```
.configure(), randomizer seed: timer (1609875073812804)
```
A seed for randomization (weights initializing) is taken from the timer, though for some debugging (or research) purposes it might be required
running multiple convergences with the same seed, which could be done using option `-s <seed>`
> note: though setting the same seed won't provide deterministic behavior in the _BLM_ (option `-b`), due to inability to control concurrent convergence

#
##### Epochs to run
```
.configure(), epochs to run: 100000
```
When NN tries once all the given input patterns and learns from them (by back-propagating the resulting error and adjusting its weights towards the closer
match) it's called an _epoch_. Because the _error plane_ is always smoothly differentiable, it inevitably leads towards the minimum either local or global
(thanks to the _learning rule_), however, it certainly requires an _unknown_ number of such iterations (epochs) to reach one.  
`Rpnn`  reaches the minimums quite quickly and then (if _LMD_ is enabled) will try bouncing itself out of the found _LM_ and will descend into another one.  
To cap the number of such iterations the number of epoch sets the limit. The maximum number of epochs is given as the only standalone attribute to `Rpnn` (if
omitted, then default number `100000` is used).

The above mostly applies when _BLM_ search is not engaged, otherwise, there the number of attempts is rather limited by the number of LM found (which is a
combinations of 2 factors `BLM_RDCE` and _target error_), though setting epoch number to a very shallow value is not advisable, as it may result in a
premature end of convergence even before reaching a local or global minimum.


#### Configuring NN Topology

NN topology could be verified with `-dd` debug depth:
```
bash $ <<<"0 0 " rpn -dd
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
.configure(), randomizer seed: timer (1610037935476627)
.configure(), epochs to run: 100000
.read_patterns_(), start reading training patterns (1 inputs + 1 outputs)...
.read_patterns_(), read 1 pattern(s)
.resolve(), training patterns read and loaded, starting convergence...
..resolve(), class 'Rpnn'...
   Rpnn::addr(): 0x7ffee61b1040
   Rpnn::min_step(): 1e-06
   Rpnn::max_step(): 1000
   Rpnn::dw_factor(): 1.1618
   Rpnn::target_error_: 0.001
   Rpnn::cost_func(): "Sse"
   Rpnn::wbp_: 0x7ffee61b1120
   Rpnn::epoch_: 0
   Rpnn::terminate_: false
   Rpnn::effectors_start_idx(): 2
   Rpnn::output_neurons_start_idx(): 2
   Rpnn::neurons()[0]: class 'rpnnNeuron'...
      neurons()[0]::addr(): 0x7fefc2c07900
      neurons()[0]::host_nn_ptr(): 0x7ffee61b1040
      neurons()[0]::is_receptor(): true
      neurons()[0]::transfer_func(): "Sigmoid"
      neurons()[0]::out(): 1
      neurons()[0]::delta(): 0
      neurons()[0]::bp_err(): 0
      neurons()[0]::synapses(): []
      neurons()[0]::inputs_ptr(): nullptr
      neurons()[0]::sum_: 0
   Rpnn::neurons()[1]: class 'rpnnNeuron'...
      neurons()[1]::addr(): 0x7fefc2c07640
      neurons()[1]::host_nn_ptr(): 0x7ffee61b1040
      neurons()[1]::is_receptor(): true
      neurons()[1]::transfer_func(): "Sigmoid"
      neurons()[1]::out(): 1
      neurons()[1]::delta(): 0
      neurons()[1]::bp_err(): 0
      neurons()[1]::synapses(): []
      neurons()[1]::inputs_ptr(): 0x7fefc2c079d0
      neurons()[1]::sum_: 0
   Rpnn::neurons()[2]: class 'rpnnNeuron'...
      neurons()[2]::addr(): 0x7fefc2c076c0
      neurons()[2]::host_nn_ptr(): 0x7ffee61b1040
      neurons()[2]::is_receptor(): false
      neurons()[2]::transfer_func(): "Sigmoid"
      neurons()[2]::out(): 1
      neurons()[2]::delta(): 0
      neurons()[2]::bp_err(): 0
      neurons()[2]::synapses()[0]: rpnnSynapse.. host_nn_ptr():0x7ffee61b1040, linked_neuron_ptr():0x7fefc2c07900, weight():2.20194e-314, delta_weight():2.20197e-314, gradient():2.20197e-314, prior_gradient():0
      neurons()[2]::synapses()[1]: rpnnSynapse.. host_nn_ptr():0x7ffee61b1040, linked_neuron_ptr():0x7fefc2c07640, weight():2.20197e-314, delta_weight():2.20197e-314, gradient():2.20197e-314, prior_gradient():0
      neurons()[2]::inputs_ptr(): nullptr
      neurons()[2]::sum_: 0
   Rpnn::input_sets_[0][0]: -1
   Rpnn::input_normalization()[0]: Norm.. found_min():0, found_max():0, base():-1, range():2
   Rpnn::target_sets_[0][0]: 0
   Rpnn::target_normalization()[0]: Norm.. found_min():0, found_max():0, base():0, range():1
   Rpnn::output_errors_[0]: 0
   Rpnn::lm_detector(): fifoDeque.. capacity():4, fifo():[]
Rpnn has converged at epoch 2 with error: 0.000525686
.resolve(), dumped rpn brains into file: rpn.bin
bash $ 
```

Neuron synapses provide linkage to other neurons via `linked_neuron_ptr()`, so that a topology could be traces down.  
In every topology there's one hidden neuron (a.k.a. _"the one"_) - that neuron is required for a NN convergence and every effector is linked to that
neuron - _"the one"_ is always shown first in the above output  
All the other neurons are from user's configuration, e.g.: the neuron with address `0x7f91fbc06cf0` is a receptor (`is_receptor(): true`),
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
> base of values M and N is the same as in debug output - it's all zero based, but the first neuron (with index of `0`) is always reserved)


option `-p N,M` allows pruning a single synapse at the neuron N for the (address of) neuron M  
a variant of option `-p N` prunes all the synapses from neuron N

# 
#### Other supported options

- `-f <file>` - lets dumping trained `rpn` brains (upon a successful convergence) into the file of your choice (default output file is `rpn.bin`)
- `-r <file>` - starts `rpn` in the trained mode - reads and reinstate brains state entirely from the file, ready to read & run the input data
- `-S <separators>` - allows specifying a list of separators used for the input lines. Default are `\s,;=` (_note the REGEX notation_)


#
### Study Examples
Let's review a few of academical and real world examples

#### Hello World!
_"Hello World!"_ problem in the NN is the training of _XOR_ function - it's the simplest problem that requires a multi-perceptron to converge
(why is that - is out of scope of this tutorial, but you can easily google up
[`The XOR Problem in Neural Networks`](https://www.google.com/search?hl=en&q=The+XOR+Problem+in+Neural+Networks&btnG=Google+Search)).

Topology for the `rpn` can be given using `-t` option followed by the perceptron sizes over the comma. E.g.: to train `rpn` for the _XOR_ function,
following topology is required:

        input1 -> (R)---(H)
                     \ / \
                      X   (O) -> output
                     / \ /
        input2 -> (R)---(H)

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
> As you might have noticed, `rpn` was trained for _**NOT XOR**_ function instead

That shows that the network has learnt the training material properly.


#### Enumerated patterns
`rpn` can accept not only numerical input/output patterns but also symbolical (per each input/output channel individually).
When symbolical input is detected, all symbolic tokens are tracked and enumerated, i.e., internally translated into respective
ordinal numerical values.

Because enumeration always occurs from the first seen token in each channel (receptor/ output neuron) separately (individually), such inputs can
be used only when the channel values (input/output) are independent from each other.

Say, we want to train the NN for XOR problem encoding signals `0` as `low` and `1` as `high`.
If we do it like this, it still works:
```
bash $ <<<"
high, low  = high
low,  high = high
low,  low  = low
high, high = low
" rpn -t2,2,1
Rpnn has converged at epoch 21 with error: 0.000462312
bash $ 
bash $ <<<"
high, low  = high
low,  high = high
low,  low  = low
high, high = low
" rpn -r rpn.bin
high
high
low
low
bash $ 
```
However, the signal mapping semantic here won't be like as one would expect. In fact, the above example corresponds to training this pattern:
```
0, 0 = 0
1, 1 = 0
1, 0 = 1
0, 1 = 1
```
I.e., in the 1st input channel `high`,`low` tokens correspond respectively to `0`,`1` signals; th 2nd input channel do `low`,`high`to `0`,`1` respectively.
the output channel's `high`,`low` tokens correspond respectively to `0`,`1`. 
It can be observed when showing real converged numbers instead of tokens:
```
bash $ <<<"
high, low  = high
low,  high = high
low,  low  = low
high, high = low
" rpn -r rpn.bin -u
0.00837045
0.0114564
0.970546
0.999975
bash $ 
```


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
However, like it was mentioned before, it's quite rare when the problem solution has a global minimum. Even for classification types of problems the
real world input data may contain duplicate and even conflicting data.  
Let's consider this input set - _NOT XOR_ with noisy inputs:
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
\- as you see it resolved the problem choosing most probable outcome for the conflicting pattern, but what's the actual number for the problematic pattern?
```
bash $ <<<'1, 1' rpn -r rpn.bin
0.666659
bash $ 
```
\- it found a LM where the error corresponds to the probability of signal occurrence in the input. The pattern `1, 1` results in 0.66,6659% probability
(of the output signal occurrence) in this case!


#### Couple classification examples from internet
 Quick search on internet leads to [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php) for the real world data sets.
 Let's take couple samples from there:  
 
##### Iris classification
Here's a page for [Iris](https://archive.ics.uci.edu/ml/datasets/Iris) flower classification,
the entire data set [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) contains 150 input patterns,
each made of 4 input parameters and 1 output class, the data (mixed) look like this:
```
bash $ cat iris.data | sort -R
7.1,3.0,5.9,2.1,Iris-virginica
5.8,2.7,3.9,1.2,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
4.9,3.1,1.5,0.1,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
...
```

The output class has only 3 values:
 - `Iris Setosa`
 - `Iris Versicolour`
 - `Iris Virginica`

The input values are some botanical property measurements of the flowers.
But do we really need to know that info about the input/output data? No, really not. What we really need to know is the total size of the input set (`150`),
number of inputs channels (`4`) and number of output classes (`1`).

Normally, NN are trained to be able to _generalize_ rather then _memorize_ the data. Thus, selecting the training set requires a careful selection of
the _representative_ data out of the input sets. Then train the NN on the training set and verify that it works on the rest of the data (i.e., the data
that it has never seen before).

Let's skip that careful selection and just randomly pick a half of the set (hoping it would be representative enough):
```
bash $ <iris.data sort -R | head -n75 >iris_train.data
bash $ 
```
Now, let's train the NN for iris problem (let's start with just 2 neurons in the hidden layer):
```
bash $ <iris_train.data rpn -f iris.bin -t4,2,1 -b0 
Rpnn found best local minimum, combined total epochs 15331 with error: 0.122279
bash $ 
```
With given error, it makes only 3 mistakes (out of 150), which is only 2% error rate - not bad at all! By far not all real-world problems have
such a good correlations between inputs and outputs

```
bash $ <iris.data rpn -riris.bin -d
.run(), reinstated rpn brains from file: iris.bin
.run(), cnv_.size(): 5
.run(), receptors_count: 4
...
.read_patterns_(), .read_patterns_(), read input values: 5.9 3.2 4.8 1.8 Iris-versicolor
Iris-virginica
...
.read_patterns_(), .read_patterns_(), read input values: 6.7 3.0 5.0 1.7 Iris-versicolor
Iris-virginica
...
.read_patterns_(), .read_patterns_(), read input values: 6.0 2.7 5.1 1.6 Iris-versicolor
Iris-virginica
```

> There's also an easier visualization than looking at debugging - by transforming outputs into JSON values and then using
[**`jtc`**](https://github.com/ldn-softdev/jtc) to apply a series of transformations and display the discrepancies between input data
and the produced results, like this:
>```
>bash $ <iris.data rpn -r iris.bin -d 2>&1 | tail -n+4 | sed -E 's/^.* ([^ ]+)$/\1/; s/.*/"&"/' |\  
>jtc -J / -jw'[::2]<V>v' -T0 -w'[1::2]' -T'[{{V}},{{}}]' / -jw'><i:' / -rw'[:]<I>k[0]<V>f[-1][1]<V>s<>F[-1]' -T'{"{I}":{{}}}'
>{ "70": [ "Iris-versicolor", "Iris-virginica" ] }
>{ "77": [ "Iris-versicolor", "Iris-virginica" ] }
>{ "83": [ "Iris-versicolor", "Iris-virginica" ] }
>bash $ 
>```

There might be a temptation to achieve even a more perfect result by throwing more neurons into the topology, but then there's a risk of
_overtraining_ the network, where it starts memorizing the data instead of _generalizing_.  
Typically the behavior of the _overtrained_ NN is attributed to following:
- once trained (on the training set), it converges with a suspiciously low error - like a global minimum is found
- when verified on the training data - it's indeed produces zero mistakes
- but once verified on the entire data sets (i.e., on the data NN hasn't seen before) it starts making unexpectedly many mistakes

\- then it's indeed could be that NN is overtrained, or the selected training set is _*unrepresentative*_ (e.g.: it could be made of data
showing a strong input-output correlations while weak correlation data were left out)

> Important: the computational power of the NN is driven by the total count of synapses and not so much by the count of neurons!

##### Car Evaluation

page: [Car evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) requires classification training  
data set file: [car.data](https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data)  
description file: [car.names](https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names)  

data look like this:
```
bash $ <car.data sort -R
med,low,2,4,big,high,vgood
low,med,3,4,small,high,good
low,low,2,more,small,low,unacc
vhigh,low,4,more,big,med,acc
high,high,3,more,big,low,unacc
med,med,3,more,med,high,vgood
...
```
Three are no continuous input values (only discrete, enumerated inputs).  The last column represent an output class (made of 4 values:
`unacc`, `acc`, `good`, `vgood`). 
There is a total of 1728 values:
```
bash $ <car.data wc -l
    1728
bash $ 
```

Thus, training data preparation (_**which is generally the most crusial part of a successful NN training**_) here also could be trivial. Description provide
following class distribution:
```
   class      N          N[%]
   -----------------------------
   unacc     1210     (70.023 %) 
   acc        384     (22.222 %) 
   good        69     ( 3.993 %) 
   vgood       65     ( 3.762 %) 
```
Thus, for the training set, we can select all entries for `good`, `vgood` patterns and around 120 (random) patterns form each `unacc`, `acc` classes:
```
bash $ for i in unacc acc good vgood; do <car.data grep ",$i" | sort -R | head -n120; done | sort -R >car_test.data
bash $ <car_test.data wc -l
     374
bash $ 
```
After playing a bit with variations of topology and other parameters, this pans out like a quite promising solution:
```
bash $ time <car_test.data rpn -t6,6,4,2,1 -b0 -e1e-30 -PBLM_RDCE:30 -f car.bin -lSoftplus -oSigmoid
Rpnn found best local minimum, combined total epochs 683236 with error: 3.19595

real	1m22.389s
user	10m9.294s
sys	0m1.718s
bash $
bash $ <car.data sort -R | rpn -r car.bin -d
.run(), reinstated rpn brains from file: car.bin
.run(), cnv_.size(): 7
.run(), receptors_count: 6
.read_patterns_(), .read_patterns_(), read input values: high high 4 more big high acc
acc
.read_patterns_(), .read_patterns_(), read input values: high med 4 2 small high unacc
unacc
.read_patterns_(), .read_patterns_(), read input values: med low 4 4 small high good
good
.read_patterns_(), .read_patterns_(), read input values: high vhigh 2 more med med unacc
unacc
...
bash $
bash $ <car.data rpn -r car.bin -d 2>&1 | tail -n+4 | sed -E 's/^.* ([^ ]+)$/\1/; s/.*/"&"/' |\
jtc -J / -jw'[::2]<V>v' -T0 -w'[1::2]' -T'[{{V}},{{}}]' / -jw'><i:' / -rw'[:]<I>k[0]<V>f[-1][1]<V>s<>F[-1]' -T'{"{I}":{{}}}' | wc -l
     149
bash $ 
```
\- The resulting error rates:
- on the training set: 28 / 374 = 7.487%
- on the entire data set: 149 / 1728 = 8.623%

So, this another example of a quite correlateable data with a good solution

## C++ class user interface
C++ class interface is rather simple. Begin with including a header file:
```
#include "rpnn.hpp"
```
That hearder file contans following classes:
 - _`class rpnnSynapse`_: facilitates synapse (neurons linkage) as well as (connection) weight
 - _`class rpnnNeuron`_: facilitates neurons - holds connecting synapses, logistic function and link to the input patterns (for receptors only)  
 - _`class rpnnBouncer`_: default class facilitating NN weight assignments (via randomization) - allows plugging weight update methods
 - _`class Rpnn`_: user facing class facilitating Resilient backPropagation Neural Network
 - _`class uniformBouncer`_: an example of pluggable functor class (for a reference) providing a uniform weight distribution 
 - _`class blmFinder`_ - a child class of `rpnnBouncer` facilitating a search of a better LM via spawning concurrent (multithreaded) copies of Rpnn host 

> `Rpnn` (and other classes user-facing methods) support _fluent_ notation


### Essential SYNOPSIS:
here's an example how to to create and train Rpnn for XOR, OR, AND problem (i.e., 3 output classes) with statically hardcoded data:
```
    // ...
    // input patterns:
    std::vector<std::vector<double>>
    	input_ptrn = {{0,1,0,1},		// 1st channel (to the 1st receptor)
                      {0,0,1,1}};		// 2nd channel (to the 2nd receptor)
    // target patterns:
    std::vector<std::vector<double>>
    	target_ptrn = {{0,1,1,1},		// OR output channel
                       {0,1,1,0},		// XOR output channel
                       {0,0,0,1}};		// AND output channel

    // configuring and training Rpnn
    Rpnn nn;
    nn.full_mesh(2, 2, 3)			// begin with defining topology
      .normalize()				// optional, but if used, must be called before load_patterns(..) 
      .load_patterns(input_ptrn, target_ptrn)	// load up in/out patterns, required
      .lm_detection(nn.synapse_count() * 3)	// engage LM trap detection (optional, but desirable)
      .target_error(0.00001)			// in this example it's optional
      .converge(10000);				// find solution

    // Offload NN brains into the file
    std::ofstream file("oxa.bin", std::ios::binary);	// open file
    file << std::noskipws << Blob(nn);		// dump (serialize) NN to file
```
Now the counterpart - reading a (trained) Rpnn brains from the file and activating with user data:
```
    // ...
    Blob b(std::istream_iterator<char>(std::ifstream{"oxa.bin", std::ios::binary}>>std::noskipws),
           std::istream_iterator<char>{});	// read serialized NN from file into blob

    Rpnn nn(b);					// Create Rpnn obj & de-serialize blob into NN
    // the two above declarations coudl be combined into one: Rpnn nn(Blob(...));

    // activate varous channels
    std::cout << "1 AND 0 = " << nn.activate({1, 0}).out(2) << std::endl; 
    std::cout << "1 XOR 1 = " << nn.activate({1, 1}).out(1) << std::endl;
    std::cout << "0 OR 1 = "  << nn.activate({0, 1}).out()  << std::endl;
```

#### Topology methods:

```
    Rpnn nn;		// The copy constructor exist, but rather performs a cloning operation;
			// the move constructor is deleted (but that might be easily lifted in the future)
			// there's one another forms for constructor:
			// 	Rpnn(Blob & blob);  - it restores Rpnn state from the blob

    nn.full_mesh(..);	// it's best to begin with creating a skeleton of topology:
			// full_mesh method exists in two variants:
			// 1. variadic form - accepts topology as enumerated perceptrons, e.g.:
			//   	full_mesh(5,3,2,3) - 5 receptors, 3 neurons in 1st hidden layer, 2 neurons in 2nd
			//                           hidden layer, 3 output neurons
			// 2. accepts a templated STL trivial container (std::vector, std::list, std::deque, etc)
			//	std::vector<int> my_topology{5,3,2,3};
			//	full_mesh(my_topology);
```

If by chance a _full-mesh_ topology is not good enough, then it's possible to modify it by _growing_ and _pruning_ synapses:  
\- class `pnnNeuron` provides an access to the methods allowing linking, growing and pruning synapses.
However, neurons themselves have to be accessed first.

There are 3 types of neurons which typically a user would need an access to:
1. _receptros_: these neurons don't have synapses and facilitate input patterns access
2. _effectors_: these are non-receptors - i.e., neurons with synapses 
3. _output_neurons_: these are the effectors in the last (output) layer

Structurally, all the neurons are being held in the sequentual container (`std::list`) and could be accessed using following iterators:

```
                         neurons().begin()      effectors_itr()
                               |                       |
                               v                       v
 std::list<rpnnNeuron>:      ("1"),  (R1)  ...  (Rn)  (E1) ... (En-m), (O1) .. (Om)
                                      ^                                 ^
                                      |                                 |
                                receptors_itr()                output_neurons_itr()
```
Thus:  
\- All neurons are accessible via `neruons()` container and its iterators `neurons().begin()` -> `neurons().end()`  
\- all receptrors are accessible via `receptors_itr()` -> `effectors_itr()` iterators  
\- all effectors (output neurons are the effectors too) are accessible via `effectors_itr()` -> `neurons().end()`  
\- all the output neurons are accessible via `output_neurons_itr()` -> `neurons().end()`  

> The first neuron is always a reserved neuron (a.k.a "the one") - it's a specially reserved empty receptor, all the effectors should
> have synapses towards this neuron and it's better not to mess up with it (as it will damage the NN ability to function properly);
> the effectors linkage to "the one" is maintained by the class itself and does not require any of overhead from the user

So, now that any of neurons can be accessed (via itterators), then following neurons methods exist to manage their synapses:
- `linkup_synapse(rpnnNeuron &)`: link synapse to a neuron by its address
-  `grow_synapses(idx1, ..)`: link synapse(s) by their index in the std::list container (variadic arguments)
-  `prune_synapses(idx1, ..)`: prune synapse(s) to neurons by its (neuron's) index (variadic arguments)

E.g., to link a first effector to the first receptor, it can be done either way:
```
    nn.effectors_itr()->linkup_synapse(*nn.receptors_itr());
    // or
    nn.effectors_itr()->grow_synapses(1);
```


```
...
```


