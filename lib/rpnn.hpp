/*
 *  rpnn.hpp
 *
 *  Created by Dmitry Lyssenko
 *  Rpnn - Resilient Propagation with Local Minima detection.
 *
 *
 */



#pragma once

#include <vector>
#include <list>
#include <math.h>
#include <random>
#include <chrono>
#include <limits>
#include "normalization.hpp"
#include "extensions.hpp"
#include "Blob.hpp"
#include "Outable.hpp"
#include "FifoDeque.hpp"
#include "dbg.hpp"


#define SIZE_T(N) static_cast<size_t>(N)

#define RPNN_MIN_STEP   1.e-6
#define RPNN_MAX_STEP   1.e+4
#define RPNN_DW_FACTOR  1.1618034
#define RPNN_BASE -1.0
#define RPNN_RANGE 2.0
#define RPNN_LMD_PCNT 0.2





//                          Class Synapse
//
class Rpnn;
class rpnnNeuron;

class rpnnSynapse {
 public:
                        rpnnSynapse(void) = default;            // SERDES requirement
                        rpnnSynapse(Rpnn &nn, rpnnNeuron & n):
                         nnp_(&nn), lpn_(&n) {}

    // access methods
    double              weight(void) const
                         { return w_; }
    void                weight(double w)
                         { w_ = w; }
    double              delta_weight(void) const
                         { return dw_; }
    void                delta_weight(double dw)
                         { dw_ = dw; }
    double              gradient(void) const
                         { return g_; }
    void                gradient(double g)
                         { g_ = g; }
    rpnnNeuron &        prior_neuron(void)
                         { return *lpn_; }
    const rpnnNeuron &  prior_neuron(void) const
                         { return *lpn_; }
    double              prior_gradient(void) const
                         { return pg_; }

    // synapse methods
    void                linkup_neuron(rpnnNeuron & n)
                         { lpn_ = &n; }
    rpnnNeuron &        linked_neuron(void)
                         { return *lpn_; }
    double              scan_input(size_t p = 0) const;         // linked Neuron output P x weight
    void                commit_weight(void);
    void                backpropagate_error(double nd);
    void                compute_gradient(double nd, size_t p);  // computed by Rpnn during training
    void                nn(Rpnn & nn)
                         { nnp_ = &nn; }
    Rpnn &              nn(void)                                // required in commit_weight()
                         { return *nnp_; }
    const Rpnn &        nn(void) const
                         { return *nnp_; }

    SERDES(rpnnSynapse, w_, dw_, g_, pg_, nnp_, lpn_)
    COUTABLE(rpnnSynapse, host_nn_ptr(), linked_neuron_ptr(),
                          weight(), delta_weight(), gradient(), prior_gradient())
    // Below definitions are for OUTABLE interface only
    const Rpnn *        host_nn_ptr(void) const
                         { return nnp_; }
    const rpnnNeuron *  linked_neuron_ptr(void) const
                         { return lpn_; }

 protected:
    void                preserve_gradients_(void)
                         { pg_ = g_; }

    // synapse's data
    Rpnn *              nnp_{nullptr};                          // required in commit_weight()
    // Rpnn access is required for each synapse in order to access global NN parameters
    // like dw_factor and min/max_step
    rpnnNeuron *        lpn_{nullptr};                          // linked prior neuron
    // linking neuron by address imposes a requirement for neurons to be in
    // a container w/o relocation policy, e.g. std::list

    // data required in training
    double              w_;                                     // weight
    double              dw_;                                    // delta weight
    double              g_;                                     // gradient value
    double              pg_{0};                                 // saved prior gradient

};





class rpnnNeuron {
 public:

    using vDouble = std::vector<double>;
    using vSynapse = std::vector<rpnnSynapse>;
    using lNeuron = std::vector<rpnnNeuron> ;
    using t_func = double (double, rpnnNeuron *);

    // list all internally predefined transfer functions
    #define RPNN_TFUNC \
                Sigmoid, \
                Tanh, \
                Tanhfast, \
                Relu, \
                Softplus, \
                Softmax
    ENUMSTR(tFunc, RPNN_TFUNC)

    // declare all static transfer functions: tf_Sigmoid, Tanh, etc
    #define XMACRO(X) static double tf_ ## X(double, rpnnNeuron*);
    XMACRO_FOR_EACH(RPNN_TFUNC)
    #undef XMACRO


                        rpnnNeuron(void) = default;
                        rpnnNeuron(Rpnn &nn):
                         nnp_(&nn) {}

    double              out(size_t p) const                     // if receptor's input connected
                         { return (is_ == nullptr) ? out_: (*is_)[p]; } // then return from it
    double              out(void) const                         // effector's output
                         { return out_; }
    rpnnNeuron &        load(double x);                         // load data directly into receptor
    bool                is_receptor(void) const
                         { return synapses_.empty(); }
    bool                is_effector(void) const
                         { return not is_receptor(); }
    double              bp_err(void) const
                         { return bpe_; }
    void                bp_err(double x)
                         { bpe_ = x; }
    double              delta(void) const
                         { return nd_; }
    void                delta(double x)
                         { nd_ = x; }
    const vDouble *     input_pattern(void)
                         { return is_; }
    void                input_pattern(const vDouble &x)
                         { is_ = &x; }
    void                cutoff_input_pattern(void)
                         { is_ = nullptr; }
    // synapses access
    vSynapse &          synapses(void)
                         { return synapses_; }
    const vSynapse &    synapses(void) const
                         { return synapses_; }
    rpnnSynapse &       synapse(size_t i)
                         { return synapses_[i]; }
    void                linkup_synapse(rpnnNeuron & n)          // link synapse to neuron by addr
                         { synapses_.emplace_back(nn(), n); }
    void                grow_synapses(size_t i);                // link synapse by idx of neuron
    template<typename... Args>
    void                grow_synapses(size_t first, Args... rest)
                         { grow_synapses(first); grow_synapses(rest...); }
    void                decay_synapses(size_t n);                // n - linked neuron n in topology
    template<typename... Args>
    void                decay_synapses(size_t first, Args... rest)
                         { decay_synapses(first); decay_synapses(rest...); }
    rpnnNeuron &        resize(size_t x) {                      // resize synapses count
                         synapses().resize(x);
                         for(auto &s: synapses()) s.nn(nn());
                         return *this;
                        }

    // education methods
    void                update_delta(void)
                         { nd_ = bpe_ * transfer_delta_( out() ); }
    void                backpropagate_error(void)
                         { for(auto &s: synapses()) s.backpropagate_error( delta() ); }
    void                compute_gradient(size_t p)             // p: is_ idx (in case of receptor)
                         { for(auto &s: synapses()) s.compute_gradient(delta(), p); }
    void                commit_weights(void)
                         { for(auto &s: synapses()) s.commit_weight(); }
    void                activate(size_t p) {
                         sum_ = 0;
                         for(auto & s: synapses()) sum_ += s.scan_input(p);
                         out_ =  transfer(sum_, this);
                        }
    double              transfer(double x, rpnnNeuron* nptr)
                         { return tf_(x, nptr); }
    void                transfer_function(t_func *fptr)
                         { tf_ = fptr; }
    t_func *            transfer_function(void)
                         { return tf_; }


    void                nn(Rpnn & nn)
                         { nnp_ = &nn; }
    void                nn(Rpnn * nn)
                         { nnp_ = nn; }
    Rpnn &              nn(void)                                // required in commit_weight()
                         { return *nnp_; }
    const Rpnn &        nn(void) const
                         { return *nnp_; }


    #define XMACRO(X) rpnnNeuron::tf_ ## X,
    SERDES(rpnnNeuron, out_, synapses_, nd_, bpe_,
                       XMACRO_FOR_EACH(RPNN_TFUNC)              // list all the pointers to tf
                       tf_, nnp_, &rpnnNeuron::serdes_is_)
    #undef XMACRO
    OUTABLE(rpnnNeuron, addr(), host_nn_ptr(), is_receptor(), transfer_func(),
                        out(), delta(), bp_err(), synapses(), inputs_ptr(), sum_)
    // Below definitions are for OUTABLE interface only
    const rpnnNeuron *  addr(void) const
                         { return this; }
    const Rpnn *        host_nn_ptr(void) const
                         { return nnp_; }
    const char *        transfer_func(void) const {
                         size_t ti;
                         for(ti = 0; ti < tf_vec_.size(); ++ti)
                          if(tf_vec_[ti] == tf_) break;
                         if(ti >= tf_vec_.size()) return "user's function";
                         return ENUMS(tFunc, ti);
                        }
    const vDouble *     inputs_ptr(void) const
                         { return is_; }

 protected:

    double              transfer_delta_(double x);

    // data
    t_func *            tf_{rpnnNeuron::tf_Sigmoid};            // pointer to a default transf.func.
    Rpnn *              nnp_{nullptr};
    const vDouble *     is_{nullptr};                           // input source of patterns

    double              out_{1.};                               // output result
    vSynapse            synapses_;

    // data required in training
    double              nd_;                                    // neuron's delta
    double              bpe_;                                   // back propagation error's delta

 private:
    // SERDES user interfaces
    void                serdes_is_(Blob &b) const;                 // serializer
    void                serdes_is_(Blob &b);                       // de-serializer

    double              sum_;                                   // sum of synapses scan (tmp value)
    // provide mapping for all predefined transfer functions (used in debugs only)
    #define XMACRO(X)   tf_ ## X,
   std::vector<t_func*> tf_vec_{ XMACRO_FOR_EACH(RPNN_TFUNC) };
    #undef XMACRO

};

STRINGIFY(rpnnNeuron::tFunc, RPNN_TFUNC)
#undef RPNN_TFUNC





class rpnnBouncer {
 // neuron weight randomizer class
 public:
                        rpnnBouncer(void) {
                         seed_ = std::chrono::system_clock::now().time_since_epoch().count();
                         rnd_.seed(seed_);
                        }
                        rpnnBouncer(Rpnn *nn): rpnnBouncer()
                         { nnp_ = nn; }
    virtual            ~rpnnBouncer(void) = default;

    size_t              seed(void) const
                         { return seed_; }
    void                seed(size_t x)
                         { rnd_.seed(x); }
    virtual void        bounce(void);
    double              base(void) const
                         { return base_; }
    double              range(void) const
                          { return range_; }
    void                base_range(double b, double r)
                         { base_ = b; range_ = r; }
    Rpnn &              nn(void) const
                         { return *nnp_; }

    bool                finish_upon_convergence(void) const
                         { return fuc_; }
    void                finish_upon_convergence(bool x)
                         { fuc_ = x; }

    SERDES(rpnnBouncer, nnp_, range_, base_, fuc_, seed_)
    OUTABLE(rpnnBouncer, &nn(), base(), range())

 protected:

    Rpnn *              nnp_{nullptr};
    double              range_{RPNN_RANGE};
    double              base_{RPNN_BASE};
    bool                fuc_{ true };                           // finish upon convergence
    std::mt19937_64     rnd_;

 private:
    size_t              seed_;                                  // storing only for tracking
};





class Rpnn {
    friend rpnnNeuron;                                          // because of normalize_(..)

 public:

    using lNeuron = std::list<rpnnNeuron>;
    using vvDouble = std::vector<std::vector<double>>;
    using c_func = double (double, double);

    #define THROWREASON \
                neuron_idx_out_of_range, \
                illegal_logistic_at_output, \
                loading_in_non_receptor, \
                insufficient_inputs, \
                min_two_perceptrons_requied, \
                perceptron_size_illegal, \
                the_one_misconfigured, \
                effectors_undefined, \
                receptors_misconfigured, \
                output_neurons_undefined, \
                output_config_inconsistent, \
                stopped_on_nan_error
    ENUMSTR(ThrowReason, THROWREASON)

    #define RPNN_COSTFUNC \
                Sse, \
                Xntropy
    ENUMSTR(costFunc, RPNN_COSTFUNC)

    // declare all static const functions
    #define XMACRO(X) static double cf_ ## X(double, double);
    XMACRO_FOR_EACH(RPNN_COSTFUNC)
    #undef XMACRO

                        Rpnn(void) = default;

    // access methods
    size_t              synapses_count(void) const {
                         size_t sum = 0;
                         for(auto ei = effectors(); ei != neurons().end(); ei++)
                          sum += ei->synapses().size();
                         return sum ;
                        }

    lNeuron &           neurons(void)
                         { return neurons_; }
    const lNeuron &     neurons(void) const
                         { return neurons_; }
    rpnnNeuron &        neuron(size_t i) {
                         if(i >= neurons().size())
                          throw EXP(ThrowReason::neuron_idx_out_of_range);
                         auto it = neurons().begin();
                         std::advance(it, i);
                         return *it;
                        }
    lNeuron::iterator   effectors(void) const
                         { return effectors_; }
    lNeuron::iterator   output_neurons(void) const
                         { return output_neurons_; }

    vvDouble &          input_patterns(void)
                         { return input_sets_; }
    const vvDouble &    input_patterns(void) const
                         { return input_sets_; }
    vvDouble &          target_patterns(void)
                         { return target_sets_; }
  std::vector<double> & output_errors(void)
                         { return output_errors_; }
    const std::vector<double> &
                        output_errors(void) const
                         { return output_errors_; }
    double              output_error(size_t i) const            // per output neuron
                         { return output_errors_[i]; }
    double              target_error(void) const
                         { return target_error_; }
    Rpnn &              target_error(double x)
                         { target_error_ = x; return *this; }
    double              global_error(void) const {             // average for all output neurons
                         double sum = 0;
                         for(auto e: output_errors()) sum += e;
                         return sum / output_errors().size();
                        }
    bool                stop_on_nan(void) const                 // stop if error turns Nan
                        { return stop_on_nan_; }
    Rpnn &              stop_on_nan(bool x)
                         { stop_on_nan_ = x; return *this; }
    const fifoDeque<double> &
                        lm_detector(void) const
                         { return error_trail_; }
    size_t              lm_detection(void) const
                         { return error_trail_.capacity(); }
    Rpnn &              lm_detection(size_t x)
                         { error_trail_.capacity(x); return *this; }


    size_t              epoch(void) const
                         { return epoch_; }
    double              min_step(void) const
                         { return min_step_; }
    Rpnn &              min_step(double x)
                         { min_step_ = x; return *this; }
    double              max_step(void) const
                         { return max_step_; }
    Rpnn &              max_step(double x)
                         { max_step_ = x; return *this; }
    double              dw_factor(void) const
                         { return dw_factor_; }
    Rpnn &              dw_factor(double x)
                         { dw_factor_ = x; return *this; }


    // configuration methods
    Rpnn &              resize(size_t x) {                      // number of neurons in topology
                         neurons().resize(x);
                         for(auto &n: neurons()) n.nn(this);
                         output_neurons_ = effectors_ = nend_;
                         return *this;
                        }
    template<template<typename, typename> class Container, typename T, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<T>>::value and
                            std::is_fundamental<T>::value, Rpnn &>::type
                        full_mesh(const Container<T, A> & c);   // container form
    template<typename... Args>
    Rpnn &              full_mesh(Args... args) {               // variadic form
                         std::vector<int> perceptrons;
                         build_vector_(perceptrons, args...);
                         full_mesh(perceptrons);
                         return *this;
                        }
    Rpnn &              load_patterns(const vvDouble & ip, const vvDouble & tp);
    Rpnn &              load_patterns(vvDouble && ip, const vvDouble & tp);

    // training methods
    double              out(size_t n = 0) const  {              // user function to read results
                         auto on = output_neurons_;
                         std::advance(on, n);
                         return nts_.empty()? on->out(): nts_.at(n).denormalize(on->out());
                        }
    void                topology_check(void);
    void                converge(size_t epochs);
    void                reset_errors(void)
                         { for(auto & e: output_errors()) e = 0.; }
    bool                normalizing(void) const {               // is input normalization engaged?
                         return not nis_.empty();
                        }
    Rpnn &              normalize(double min = -1., double max = 1.)
                         { nis_.resize(1, Norm(min, max - min)); return *this; }
    Rpnn &              activate(size_t p) {
                         for(auto ei = effectors(); ei != neurons().end(); ei++)
                          ei->activate(p);
                         return *this;
                        }
    Rpnn &              activate(void) {                        // directly loaded into receptors
                         for(auto ri = ++neurons().begin(); ri != effectors(); ++ri)
                          if(ri->input_pattern() == nullptr) break;
                          else ri->cutoff_input_pattern();
                         return activate(0);
                        }
    Rpnn &              activate(const std::vector<double> &inputs) {
                         auto ii = inputs.begin();              // load inputs into receptors
                         for(auto ri = ++neurons().begin(); ri != effectors(); ++ri, ++ii)
                          if(ii == inputs.end()) throw EXP(Rpnn::insufficient_inputs);
                          else ri->load(*ii);
                         return activate();
                        }
    void                init_neurons_(void);
    Rpnn &              cost_function(c_func *cf)
                         { cf_ = cf; return *this; }
    c_func *            cost_function(void) const
                         { return cf_; }

    void                bounce_weights(void) const
                         { wbp_->bounce(); }
    Rpnn &              bouncer(rpnnBouncer *wb)
                         { wbp_ = wb; return *this; }
    rpnnBouncer &       bouncer(void)
                         { return *wbp_; }


    SERDES(Rpnn, input_sets_, nis_, target_sets_, nts_,neurons_,
                 &Rpnn::serdes_itr_, output_errors_, target_error_,
                 min_step_, max_step_, dw_factor_, wb_, wbp_,
                 Rpnn::cf_Sse, Rpnn::cf_Xntropy, cf_, epoch_, terminate_, error_trail_)

    OUTABLE(Rpnn, addr(), min_step(), max_step(), dw_factor(), target_error_,
                  cost_func(), wbp_, epoch_, terminate_,
                  effectors_start_idx(), output_neurons_start_idx(), neurons(),
                  input_sets_, nis_, target_sets_, nts_, output_errors_, lm_detector())
    // Below definitions are for OUTABLE interface only
    const Rpnn *        addr(void) const
                         { return this; }
    size_t              effectors_start_idx(void) const {
                         if(effectors_ == nend_) return 0;
                         return std::distance(neurons().begin(),
                                              static_cast<lNeuron::const_iterator>(effectors_));
                        }
    size_t              output_neurons_start_idx(void) const {
                         if(output_neurons_ == nend_) return 0;
                         return std::distance(neurons().begin(),
                                        static_cast<lNeuron::const_iterator>(output_neurons_));
                        }
    const char *        cost_func(void) const {
                         size_t ci;
                         for(ci = 0; ci < cf_vec_.size(); ++ci)
                          if(cf_vec_[ci] == cf_) break;
                         if(ci >= cf_vec_.size()) return "user's function";
                         return ENUMS(costFunc, ci);
                        }

    DEBUGGABLE()
    EXCEPTIONS(ThrowReason)

 protected:

    void                compute_error_(size_t p);
    void                educate_(size_t p);
    bool                is_lm_detected_(double err);
    void                normalize_patterns_(vvDouble &ptrn, std::vector<Norm> &np);


    lNeuron             neurons_;                               // vector of all neurons
    lNeuron::iterator   nend_{neurons_.end()};                  // initializer for effectros/on
    lNeuron::iterator   effectors_{nend_};                      // beginning of effectors
    lNeuron::iterator   output_neurons_{nend_};                 // beginning of out.neurons

    std::vector<double> output_errors_;                         // one per each output neuron
    vvDouble            input_sets_;                            // normalized input data patterns
    std::vector<Norm>   nis_;                                   // norm vector per input pattern
    vvDouble            target_sets_;                           // normalized targets data patterns
    std::vector<Norm>   nts_;                                   // norm vector per target pattern

    double              target_error_{0.01};                    // global target error, set by user
    double              min_step_{RPNN_MIN_STEP};
    double              max_step_{RPNN_MAX_STEP};
    double              dw_factor_{RPNN_DW_FACTOR};
    c_func *            cf_{Rpnn::cf_Sse};                      // default cost func.
    rpnnBouncer         wb_{this};
    rpnnBouncer *       wbp_{&wb_};
    fifoDeque<double>   error_trail_;                           // for detecting LM traps

 private:

    Rpnn &              reset_lm_(void)
                         { error_trail_.clear(); return *this; }
    double              normalize_(double x, rpnnNeuron *n) {   // normalize receptr in its pattern
                         auto ni = ++neurons().begin();         // begin from 1st receptor
                         for(auto nsi = nis_.begin(); nsi != nis_.end(); ++nsi, ++ni)
                          if(&*ni == n) return nsi->normalize(x);
                         throw EXP(receptors_misconfigured);
                        }

    // SERDES user interfaces
    void                serdes_itr_(Blob &b) const;                 // serializer
    void                serdes_itr_(Blob &b);                       // de-serializer

    template<typename T, typename... Args>
    typename std::enable_if<std::is_fundamental<T>::value, void>::type
                        build_vector_(std::vector<int> &p, T first, Args... args) {
                         p.push_back(first);
                         build_vector_(p, args...);
                        }
    void                build_vector_(std::vector<int> &) {}

    size_t              epoch_{0};
    bool                stop_on_nan_{true};
    bool                terminate_{false};
    // provide mapping for all predefined const functions (used in debugs only)
    #define XMACRO(X)   cf_ ## X,
   std::vector<c_func*> cf_vec_{ XMACRO_FOR_EACH(RPNN_COSTFUNC) };
    #undef XMACRO

};

STRINGIFY(Rpnn::ThrowReason, THROWREASON)
STRINGIFY(Rpnn::costFunc, RPNN_COSTFUNC)
#undef THROWREASON
#undef RPNN_COSTFUNC



// SERDES user interfaces in rpnnNeuron
//
void rpnnNeuron::serdes_is_(Blob &b) const {
 // serialize is_ only if internal input_patter linked to the neuron
 if(nnp_ == nullptr or nn().input_patterns().empty())
  { b.append_cntr(false); return; }

 for(auto pi = nn().input_patterns().begin(); pi != nn().input_patterns().end(); ++pi)
  if(&*pi == is_)
   { b.append_cntr(true); b.append(is_); return; }

 b.append_cntr(false);
}

void rpnnNeuron::serdes_is_(Blob &b) {
 // de-serialize is_ only if normalization *was* on
 if(b.restore_cntr())                                           // restore state of normalization
  b.restore(is_);
}

// SERDES user interfaces in Rpnn
//
void Rpnn::serdes_itr_(Blob &b) const {
 // serialize effectors and output_neurons iterators
 if(b.append_cntr(effectors_ != nend_))                         // preserve effectors_ itr
  b.append_cntr(std::distance(neurons().begin(),
                              static_cast<lNeuron::const_iterator>(effectors_)));

 if(b.append_cntr(output_neurons_ != nend_))                    // preserve output_neurons_ itr
  b.append_cntr(std::distance(neurons().begin(),
                              static_cast<lNeuron::const_iterator>(output_neurons_)));
}

void Rpnn::serdes_itr_(Blob &b) {
 // de-serialize effectors and output_neurons iterators
 if(b.restore_cntr()) {                                         // restore effectors_ itr
  effectors_ = neurons().begin();
  size_t dist = b.restore_cntr();
  std::advance(effectors_, dist);
 }
 if(b.restore_cntr()) {                                         // restore output_neurons_ itr
  output_neurons_ = neurons().begin();
  size_t dist = b.restore_cntr();
  std::advance(output_neurons_, dist);
 }
}




//                          Synapse methods
//
double rpnnSynapse::scan_input(size_t p) const {
 // argument p is used only by input neurons (receptors) to select sample `p` from the input source
 // for all the effectors there's only one output, thus parameter `p` for them is irrelevant
 return prior_neuron().out(p) * weight();
}



void rpnnSynapse::commit_weight(void) {
 // calculate new delta weight, new weight, and preserve gradient for next pass to check the sign
 dw_ = gradient() * prior_gradient() > 0.?                      // did gradient change sign?
        fmin(delta_weight() * nn().dw_factor(), nn().max_step()):   //  no: increase step
        fmax(delta_weight() / nn().dw_factor(), nn().min_step());   // yes: decrease step
 w_ += gradient() > 0? delta_weight() : -delta_weight();        // make next step

 preserve_gradients_();
 gradient(0.);                                                  // computed by Rpnn during training
}



void rpnnSynapse::backpropagate_error(double nd) {
 // back-prop neruon's computed error to prior linked neuron
 if(prior_neuron().is_effector())                               // back-probp to effectors only
  prior_neuron().bp_err(prior_neuron().bp_err() + weight() * nd);
}



void rpnnSynapse::compute_gradient(double nd, size_t p) {
 // gradient computed and updated during Rpnn training
 double g = gradient() + nd * prior_neuron().out(p);
 if(isinf(g) or isnan(g)) return;                               // gradient non-calculable
 g_ = g;
}





//                          Neuron methods
//
// predefined transfer functions:
double rpnnNeuron::tf_Sigmoid(double x, rpnnNeuron*)            // for any perceptrons
 { return 1. / (1. + exp(-x)); }

double rpnnNeuron::tf_Tanh(double x, rpnnNeuron*)               // for any perceptrons
 { return 2. / (1. + exp(-x)) - 1.; }

double rpnnNeuron::tf_Tanhfast(double x, rpnnNeuron*)           // for any perceptrons
 { return x<=0 ? -1./(x-1.) -1.: 2.-1./(x+1.) -1.; }

double rpnnNeuron::tf_Softplus(double x, rpnnNeuron* n) {       // not for output perceptron
 if(n == nullptr)                                               // testing limits
  throw Rpnn::illegal_logistic_at_output;
 return log(1.0 + exp(x));
}

double rpnnNeuron::tf_Relu(double x, rpnnNeuron* n) {           // not for output perceptron
 if(n == nullptr)                                               // testing limits
  throw Rpnn::illegal_logistic_at_output;
 return x>0 ? x : 0.;
}

// now, the softmax, unlike other logistic functions is a special case: it requires
// the NN to run/activate all the output neurons in a given training set first and
// only then softmax can be calculated (i.e., a sum by all neuron's outputs is required);
// while other output logistics could be mixed and matched, Softmax neurons is best not to mix
double rpnnNeuron::tf_Softmax(double x, rpnnNeuron* n) {  // only for output perceptron
 // maxv - max of outputs for normalization, lse = log of sum's exponent
 // parameters 'maxv' and 'lse' will be stored in neruon[0]'s bpe_ and nd_ variables respectively
 // - those are unused in "the one" - this way we avoid storing them as static values and hence
 // softmax becomes re-enterable
 if(n == nullptr)                               // testing limits
  return x > 0? 1.: 0.;
 auto & one = n->nn().neurons().front();        // 1st neuron - "the one"
 auto & maxv = one.bpe_;                        // utilizing one's bpe_ for storing max value
 auto & lse = one.nd_;                          // utilizing one's nd_ for storing ln(sum(err))

 if(n->transfer_function() != rpnnNeuron::tf_Softmax)// normally, n points to output neuron, else
  return exp(x - maxv - lse);                   // it's a call from derivative: maxv/lse computed
 if(n != &n->nn().neurons().back())             // not all output neurons have been activated
  return x;                                     // so delay until it's a last output neuron

 // here, the last neuron's activation: finish calculating maxv, lse and return activation value
 maxv = x;                                                  // find max output value
 for(auto it = ++n->nn().neurons().rbegin(); it->transfer_function() == tf_Softmax; ++it)
  maxv = std::max(maxv, it->out());

 lse = exp(x - maxv);
 for(auto it = ++n->nn().neurons().rbegin(); it->transfer_function() == tf_Softmax; ++it)
  lse += exp(it->out() - maxv);
 lse = log(lse);

 for(auto it = ++n->nn().neurons().rbegin(); it->transfer_function() == tf_Softmax; ++it)
  it->out_ = exp(it->out() - maxv - lse);                       // correct values for prior outputs
 // explanation how a trivial softmax became so tricky is here:
 // http://stackoverflow.com/questions/9906136/\
 // implementation-of-a-softmax-activation-function-for-neural-networks

 return exp(x - maxv - lse);                             // return the last one
}



rpnnNeuron & rpnnNeuron::load(double x) {                       // load data directly into receptor
 if(not is_receptor() or this == &nn().neurons().front())       // i.e. non-receptor or "the one"
  throw Rpnn::loading_in_non_receptor;
 out_ = nn().normalizing()? nn().normalize_(x, this): x;
 return *this;
}


void rpnnNeuron::grow_synapses(size_t i)                        // extend synapse by idx of neuron
 // slow way (addressing std::list by index)
 { linkup_synapse(nn().neuron(i)); }



void rpnnNeuron::decay_synapses(size_t n) {
 // n - linked neuron n (not a synapse index!) in NN topology
 for(auto si = synapses().begin(); si != synapses().end(); ++si)
  if( &si->linked_neuron() == &nn().neuron(n) )
   { synapses().erase(si); break; }
}



// transfer_delta_ is calculated only for update_delta(), which, in turn, is invoked only
// after all the effectors have been activated; given so, all the effectors retain the "sum_"
// of all synapses' scans, which can be reused.
double rpnnNeuron::transfer_delta_(double x) {
 // dw_ (weights delta) calculation is no longer proportional to the gradient (in Rpnn), instead
 // it relies on the sign of the gradient only. Hence no need knowing the formulae for the
 // transfer's derivative, instead a coarse gradient calculation will suffice ('dy' actually).
 // the benefit - it will work with any transfer function and no need specifying its derivative
 // the step is from RPNN_MIN_STEP (minimum delta step)
 return transfer(sum_ + nn().min_step(), &nn().neurons().front()) - x;
}





//                          Bouncer methods
//
void rpnnBouncer::bounce(void) {
 // bounce all weights for all effectors (synapses)
 for(auto ei = nn().effectors(); ei != nn().neurons().end(); ei++)
  for(auto &s: ei->synapses())
   s.weight( static_cast<double>(rnd_()) * range() / rnd_.max() + base() );
}





//                          Rpnn methods
//

// predefined cost functions:
double Rpnn::cf_Sse(double output, double target) {
 double delta = target - output;
 return delta * delta;
}

double Rpnn::cf_Xntropy(double output, double target) {
 return output == 0.? std::numeric_limits<double>::max(): -log(output) * target;
}



template<template<typename, typename> class Container, typename T, typename A>
typename std::enable_if<std::is_same<A, std::allocator<T>>::value and
                        std::is_fundamental<T>::value, Rpnn &>::type
Rpnn::full_mesh(const Container<T, A> & c) {
 // do some checks and build a full mesh topology
 if(c.size() < 2) throw EXP(Rpnn::min_two_perceptrons_requied);

 size_t sum = 1;                                                // provision "the one"
 for(auto v: c) {
  if(v <= 0) throw EXP(Rpnn::perceptron_size_illegal);
  sum += v;
 }
 resize(sum);

 // build a full mesh topology
 Rpnn::lNeuron::iterator pi{neurons().begin()},                 // perceptron iterator
                         ppi{neurons().begin()};                // prior pi
 for(auto &v: c) {
  if(&v == &c.front()) { advance(pi, v + 1); continue; }        // skip receptors and "the one"
  if(effectors_ == neurons().end()) effectors_ = pi;
  auto bpi = pi;                                                // beginning of perceptron
  for(auto i = v; i > 0; --i) {                                 // go via all neurons in pi
   GUARD(ppi)
   if(&*ppi != &neurons().front())                              // if non-initial effectors-pcptron
    pi->linkup_synapse(neurons().front());                      // linkup "the one"
   while(&*ppi != &*bpi)                                        // go via all neurons in ppi
    { pi->linkup_synapse(*ppi); ++ppi; }
   ++pi;
  }
  ppi = bpi;
 }

 if(output_neurons_ == neurons().end()) output_neurons_ = ppi;
 output_errors_.resize(std::distance(output_neurons_, neurons().end()));
 return *this;
}



Rpnn & Rpnn::load_patterns(const vvDouble & ip, const vvDouble & tp) {
 // load ip/tp into respective containers, connect to input neurons
 // normalize target data, optionally normalize the input data

 // link input patterns to receptors or load and normalize
 if(ip.size() != SIZE_T(std::distance(neurons().begin(), effectors()) - 1))
  throw EXP(Rpnn::receptors_misconfigured);

 auto pi = ip.begin();

 if(normalizing()) {                                            // then copy and normalize
  input_patterns() = ip;
  normalize_patterns_(input_patterns(), nis_);
  pi = input_patterns().begin();
 }
 // linkup each receptor to the respective input pattern
 for(auto ri = ++neurons().begin(); ri != effectors(); ++ri, ++pi)
  ri->input_pattern(*pi);

 // load targets (targets must be always copied and normalized)
 if(tp.size() != SIZE_T(std::distance(output_neurons(), neurons().end())))
  throw EXP(Rpnn::output_config_inconsistent);
 target_patterns() = tp;
 normalize_patterns_(target_patterns(), nts_);

 return *this;
}



Rpnn & Rpnn::load_patterns(vvDouble && ip, const vvDouble & tp) {
 // facilitate the first argument being r-value (second is always copied)
 input_patterns() = std::move(ip);
 return load_patterns(input_patterns(), tp);
}




void Rpnn::topology_check(void) {
 // check:
 // minimal topology must have at least 3 neurons ("the one", 1x receptor, 1x effector)
 if(neurons().size() <= 2)
  throw EXP(Rpnn::min_two_perceptrons_requied);

 // first neuron is "the one": should have no synapses, out_ must be 1, input pattern is null
 auto & one = neurons().front();
 if(not one.synapses().empty() or one.out() != 1. or one.input_pattern() != nullptr)
  throw EXP(Rpnn::the_one_misconfigured);

 // effectors iterator assigned (following all receptors)
 size_t e_dist = std::distance(neurons().begin(), effectors());
 if(effectors() == nend_ or e_dist <= 1)
  throw EXP(Rpnn::effectors_undefined);

 // all receptors should have input patters assigned (same size) and don't have synapses
 size_t ips = 0;                                                // input patterns size
 for(auto ri = ++neurons().begin(); ri != effectors(); ++ri) {
  if(ri->input_pattern() == nullptr or not ri->synapses().empty())
   throw EXP(Rpnn::receptors_misconfigured);
  if(ips == 0 and ri == ++neurons().begin())                    // ips unresolved & it's 1st neuron
   ips = ri->input_pattern()->size();
  else
   if(ips != ri->input_pattern()->size())
    throw EXP(Rpnn::receptors_misconfigured);
 }

 // output_neurons iterator assigned (after or eq to effectors start)
 size_t o_dist = std::distance(neurons().begin(), output_neurons());
 if(effectors() == nend_ or o_dist < e_dist)
  throw EXP(Rpnn::output_neurons_undefined);

 // all output_errors size must be equal to output_neurons and number of target_patters
 o_dist = std::distance(output_neurons(), neurons().end());
 if(o_dist == 0 or
    o_dist != output_errors().size() or
    o_dist != target_patterns().size() or
    target_patterns().front().size() != ips)
  throw EXP(Rpnn::output_config_inconsistent);
}



void Rpnn::converge(size_t epochs) {
 // converge either to solution (error < target) or end of epochs
 DBG(2) DOUT() << "dump before convergence: " << *this << std::endl;
 topology_check();

 if(error_trail_.empty()) {                                     // haven't been trained before
  wbp_->bounce();
  init_neurons_();
 }

 epoch_ = 0;
 size_t patterns = (++neurons().begin())->input_pattern()->size();

 while(not terminate_) {                                        // terminate_ is external signal
  reset_errors();

  for(size_t p = 0; p < patterns; p++) {                        // cycle through all patterns
   activate(p);                                                 // activate entire NN
   compute_error_(p);                                           // accumulate errors for pattern p
   educate_(p);                                                 // train NN for given target ptrn
  }
  DBG(4) DOUT() << "dump while converging: " << *this << std::endl;

  double global_err = global_error();
  if(global_err < target_error() or epoch_ >= epochs) {         // provisioning for LM finder class
   if(wbp_->finish_upon_convergence()) break;
   DBG(0) DOUT() << "re-bouncing, epoch " << epoch_ << ", error: " << global_err
                 << ", target error: " << target_error() << std::endl;
   bounce_weights();                                             // to change target error possibly
  }
  if(stop_on_nan() and isnan(global_err))
   throw EXP(Rpnn::stopped_on_nan_error);

  if(lm_detection() > 0 and is_lm_detected_(global_err))        // provisioning for LM finder class
   if(terminate_) break;                                        // could have been setup by bounce

  for(auto ei = effectors(); ei != neurons().end(); ei++)       // update new weights based on new
   ei->commit_weights();                                        // gradients computed in educate()
  epoch_++;
 }

 terminate_ = false;
 DBG(1) DOUT() << "dump post-convergence: " << *this << std::endl;
}



void Rpnn::init_neurons_(void) {
 // initialize all neurons and their synapses with initial values
 for(auto ei = effectors(); ei != neurons().end(); ei++) {
  for(auto &s: ei->synapses()) {
   s.delta_weight(1.);
   s.gradient(0.);
  }
 }
}



void Rpnn::compute_error_(size_t p) {
 // add current error per each output neuron
 auto on = output_neurons_;
 auto ts = target_patterns().begin();
 for(auto &e: output_errors())
  e += cf_(on++->out(), (*ts++)[p]);
}



void Rpnn::educate_(size_t p) {                                 // p - trained pattern
 // "educate" all effectors: make sure they learn the lesson

 // 1. prepare (clear) bpErrors in all effectors except output neurons
 for(auto ei = effectors(); ei != output_neurons_; ei++)
  ei->bp_err(0.);

 // 2. update only output neurons bpe (bpe = target - output)
 auto ts = target_patterns().begin();
 for(auto oni = output_neurons_; oni != neurons().end(); oni++)
  oni->bp_err((*ts++)[p] - oni->out());

 // 3. update all effectors delta and back propagate error, do it backwards
 for(auto ei = neurons().rbegin(); ei->is_effector(); ++ei) {
  ei->update_delta();
  ei->backpropagate_error();
  ei->compute_gradient(p);
 }
}



bool Rpnn::is_lm_detected_(double err) {
 // detect if we're trapped in the LM
 error_trail_.push_back(err);                                   // push new value
 if(error_trail_.size() < error_trail_.capacity()) return false;// wait until enough errs collected

 size_t tortoise = 0;
 size_t hare = 1;
 auto & et = error_trail_;

 auto is_looping = [&](void) -> bool {                          // detect looping lambda
  for(auto tt = tortoise, th = tt + 1, ht = hare, hh = ht + 1;  // tortoise/hare head/tail
      hh < et.capacity();
      ++tt, ++th, ++ht, ++hh)
   // if |dt-dh / dh| > LMD% then no looping
   if(fabs(((et[th] - et[tt]) - (et[hh] - et[ht])) / (et[hh] - et[ht])) >= RPNN_LMD_PCNT)
    return false;
  return true;
 };

 while(hare - tortoise <= error_trail_.capacity() - hare and
       hare < error_trail_.capacity()) {
  if(fabs(et[tortoise] - et[hare]) / et[hare] < RPNN_LMD_PCNT and   // if values are close enough
     is_looping()) {                                           // then see if it's looping
    DBG(0) DOUT() << "LM trap detected at epoch " << epoch_
                  << ", error: " << global_error()
                  << " (target error: " << target_error() << ")" << std::endl;
    bounce_weights();
    init_neurons_();
    reset_lm_();
    return true;
  }
  ++++hare;
  ++tortoise;
 }

 return false;
}



void Rpnn::normalize_patterns_(vvDouble &ptrn, std::vector<Norm> &np) {
 // normalize data pattern (ptrn) and fill normalization set (np)
 double min{0}, max{1};
 if(&ptrn != &target_patterns()) {                              // normalizing inputs
  // assert: nis_.size() > 0 - ensured by prior checking normalization
  min = nis_.front().base();                                    // in such case base/range are
  max = nis_.front().range() + min;                             // in nis_.front()
 }
 // for inputs normalization bounds (base/range) would be the same across all patterns
 // for targets, they could vary hence will be setup during normalizing
 np.resize(ptrn.size(), Norm(min, max - min));

 for(size_t i = 0; i < ptrn.size(); ++i) {                      // normalize all patterns
  if(&ptrn == &target_patterns()) {                             // ON logistic could vary, hence
   auto on = output_neurons();                                  // require own b/r per each neurn
   std::advance(on, i);
   min = on->transfer(-std::numeric_limits<double>::max(), nullptr);
   max = on->transfer(std::numeric_limits<double>::max(), nullptr);
   np[i].base_range(min, max - min);
  }
  np[i].find_bounds(ptrn[i].begin(), ptrn[i].end());            // find min/max in whole pattern[i]
  for(auto &v: ptrn[i])                                         // normalize all values
   v = np[i].normalize(v);
 }
}





#undef SIZE_T
#undef RPNN_MIN_STEP
#undef RPNN_MAX_STEP
#undef RPNN_DW_FCTOR
#undef RPNN_RANGE
#undef RPNN_BASE









