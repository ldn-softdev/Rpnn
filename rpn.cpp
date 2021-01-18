/*
 *  rpn.currentp
 *
 * Created by Dmitry Lyssenko
 *
 * a cli toy for playing with Resilient backprop NN
 * it features:
 *  - easy way to build any topology (including recursive)
 *  - flexibility for choosing out of multiple logistic functions (for hidden and output neurons)
 *  - 2 cost functions (SSE, Cross-Entropy)
 *  - multiple output classes (including multi-class, using Softmax logistic as output activation)
 *  - local minimum early detection mechanism - drastically increases chances for convergence
 *  - finding better (deeper) local minimum mechanism via multi-threaded search
 *    where no global minimum exist
 *  - auto-enumeration of symbolic inputs (channels must be independent of each other)
 *
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <exception>
#include <regex>
#include "lib/rpnn.hpp"
#include "lib/getoptions.hpp"
#include "lib/signals.hpp"

using namespace std;

#define PRGNAME "Resilient Propagation Neural network (https://github.com/ldn-softdev/Rpnn)"
#define VERSION "1.05"
#define CREATOR "Dmitry Lyssenko"
#define EMAIL "ldn.softdev@gmail.com"


#define SIZE_T(N) static_cast<size_t>(N)

#define ITR first                                           // for emplace pair
#define STATUS second                                       // for emplace pair
#define KEY first                                           // for iterator pair
#define VALUE second                                        // for iterator pair

#define OPT_ABC a                                           // engage alternative (uniform) bouncer
#define OPT_BLM b                                           // engage finding blm
#define OPT_DBG d                                           // debug
#define OPT_TPG t                                           // topology
#define OPT_ERR e                                           // target error
#define OPT_INN n                                           // normalize inputs?
#define OPT_LMF m                                           // local minimim detection factor
#define OPT_COF c                                           // cost function
#define OPT_ETF l                                           // effectors transfer function
#define OPT_OTF o                                           // output neuron transfer function
#define OPT_GRS g                                           // grow synapse
#define OPT_PRS p                                           // prune synapse
#define OPT_DMP f                                           // file to dump NN
#define OPT_RDF r                                           // file to read NN from
#define OPT_SED s                                           // seed for randomizer
#define OPT_RUP u                                           // round up outputs
#define OPT_GSR G                                           // grow synapses recursively
#define OPT_GPM P                                           // generic parameters
#define OPT_SPR S                                           // values separator while reading cin

#define XSTR(X) #X
#define STR(X) XSTR(X)
#define XCHR(X) *#X
#define CHR(X) XCHR(X)

#define EXT_STDEXP 1                                        // std exception
#define EXT_RPNEXP 2                                        // Rpnn exception
#define EXT_CDNTCV 3                                        // Failed convergence





class TwoWayConversion {
 public:
                        TwoWayConversion(void) = default;

    double              operator()(const string & s);
    string              operator()(double);
    bool                empty(void) const
                         { return s2i_.empty(); }
    size_t              size(void) const
                         { return s2i_.size(); }
    bool                roundup_toggle(void) const
                         { return rut_; }
    void                roundup_toggle(bool x)
                         { rut_ = x; }

    SERDES(TwoWayConversion, s2i_, i2s_, rut_)

 protected:
    map<string, size_t> s2i_;
    vector<string> i2s_;

 protected:
    bool                rut_{false};                            // roundup toggle flag
};


double TwoWayConversion::operator()(const string & str) {
 // convert to double, enumerate non-convertible
 if(s2i_.empty())                                               // try stod first
  try { return stod(str); } catch (...) {}

 size_t idx = s2i_.size();
 auto er = s2i_.emplace(str, idx);
 if(er.STATUS == true) i2s_.push_back(move(str));               // new element, update reverse map
 return er.ITR->VALUE;
}


string TwoWayConversion::operator()(double x) {
 // convert to string, round up before conversion if required
 auto to_str = [&](double x, bool rut = true) {
       stringstream ss;
       ss << (roundup_toggle() == rut? floor(x + 0.5): x);
       return ss.str();
      };

 if(i2s_.empty())                                               // no enumeration occurred
  return to_str(x);
 size_t idx = x + 0.5;                                          // return symbolical values
 return idx < i2s_.size()?
         (roundup_toggle()? to_str(x, not roundup_toggle()) : i2s_[idx]):
         to_str(x, roundup_toggle());
}





class Rpn: public Rpnn {
 // housing Rpnn, and some extra classes facilitating all Rpnn functionality
 public:
                        Rpn(void) = delete;
                        Rpn(Getopt &opt): opt_{&opt}
                         { bouncer(blm_); DBG().severity(blm_); }

    Getopt &            opt(void)
                         { return *opt_; }
    const map<const char*, void*> &
                        cfm(void) const
                         { return cfm_; }
    const map<const char*, void*> &
                        tfm(void) const
                         { return tfm_; }

    Rpn &               configure(void);
    Rpn &               resolve(void);
    Rpn &               run(void);


 protected:
    blmFinder           blm_;
    uniformBouncer      ub_;


 private:
    using vvDouble = vector<vector<double>>;

    bool                read_patterns_(vvDouble &ip, vvDouble &tp = dummy_tp);


    Getopt *            opt_;
    vector<TwoWayConversion>
                        cnv_;
    string              sep_;                                   // value REGEX separators

    // map of predefined cost functions
    map<const char*, void*>
                        cfm_ {
        {STRENM(Rpnn::costFunc, Rpnn::Sse), reinterpret_cast<void*>(Rpnn::cf_Sse)},
        {STRENM(Rpnn::costFunc, Rpnn::Xntropy), reinterpret_cast<void*>(Rpnn::cf_Xntropy)}
    };

    // map predefined logistic functions
    map<const char*, void*>
                        tfm_ {
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Sigmoid),
            reinterpret_cast<void*>(rpnnNeuron::tf_Sigmoid)},
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Tanh),
            reinterpret_cast<void*>(rpnnNeuron::tf_Tanh)},
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Tanhfast),
            reinterpret_cast<void*>(rpnnNeuron::tf_Tanhfast)},
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Relu),
            reinterpret_cast<void*>(rpnnNeuron::tf_Relu)},
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Softplus),
            reinterpret_cast<void*>(rpnnNeuron::tf_Softplus)},
        {STRENM(rpnnNeuron::tFunc, rpnnNeuron::Softmax),
            reinterpret_cast<void*>(rpnnNeuron::tf_Softmax)},
    };

    static vvDouble dummy_tp;                               // default param for read_patterns
};

vector<vector<double>> Rpn::dummy_tp;




int main(int argc, char* argv[]) {

 Getopt opt;
 Rpn rpn(opt);

 opt.prolog("\n" PRGNAME "\nVersion " VERSION " (built on " __DATE__ \
            "), developed by " CREATOR " (" EMAIL ")\n");

 opt[CHR(OPT_ABC)].desc("plug in a uniform bouncer (alternative to randomizer)");
 opt[CHR(OPT_BLM)].desc("best local minimum search (0: #threads equals #cores)")
                  .name("threads");
 opt[CHR(OPT_DBG)].desc("turn on debugs (multiple calls increase verbosity)");
 opt[CHR(OPT_TPG)].desc("full mesh topology (enumerated perceptrons)")
                  .bind("1,1").name("perceptrons");
 opt[CHR(OPT_ERR)].desc("convergence target error").bind("0.001").name("target_err");
 opt[CHR(OPT_INN)].desc("input normalization (min=max to disable)").bind("-1,+1").name("min,max");
 opt[CHR(OPT_LMF)].desc("local minimum trap detection (0: disable)").bind("2").name("factor");
 opt[CHR(OPT_COF)].desc("cost function").bind("Sse").name("cost_func");
 opt[CHR(OPT_ETF)].desc("effectors logistic function").bind("Sigmoid").name("transfer");
 opt[CHR(OPT_OTF)].desc("output neurons logistic function").bind("Sigmoid").name("transfer");
 opt[CHR(OPT_GRS)].desc("grow synapse from neuron N to neuron M").name("N,M");
 opt[CHR(OPT_GSR)].desc("recursively interconnect neurons N to M").name("N,M");
 opt[CHR(OPT_PRS)].desc("prune synapse at neuron N to neuron M").name("N,M");
 opt[CHR(OPT_DMP)].desc("file to dump Rpnn brain to").bind("rpn.bin").name("file_name");
 opt[CHR(OPT_RDF)].desc("file to reinstate Rpnn brain from").bind("rpn.bin").name("file_name");
 opt[CHR(OPT_SED)].desc("seed for randomizer (0: auto)").bind("0").name("seed");
 opt[CHR(OPT_RUP)].desc("round up outputs to integer values");
 opt[CHR(OPT_SPR)].desc("value separators (REGEX)").bind(R"(\s,;=)").name("separators");
 opt[CHR(OPT_GPM)].desc("modify generic parameters (PARAM=x,y,..)").name("param");
 opt[0].desc("epochs to run convergence").name("epochs").bind("100000");

 string epilog{R"(
 - parameters N,M are zero based, the index 0 refers to a reserved neuron "the one"
 - factor for option -)" STR(OPT_LMF) R"( is multiple of the total count of synapses (weights)

 available cost functions:
 {CF}
 available logistic functions:
 {LF}
 generic Rpnn parameters (alterable with -)" STR(OPT_GPM) R"():
 {GPM}
 for further details refer to https://github.com/ldn-softdev/Rpnn)"};

 // update epilogue with predefined costs, logistics & params
 auto update_epilogue = [&](auto &cnt, const string &rpl, bool val = false) {
  stringstream ss;
  for(auto &c: cnt)
   { ss << "\to " << c.KEY; if(val) ss << " [" << c.VALUE << "]"; ss << endl; }
  epilog = regex_replace(epilog, std::regex{R"(\{)" + rpl + R"(\})"}, ss.str());
 };
 update_epilogue(rpn.cfm(), "CF");
 update_epilogue(rpn.tfm(), "LF");
 update_epilogue(rpn.gpm(), "GPM", true);
 opt.epilog(epilog.c_str());

 // parse options
 try
  { opt.parse(argc, argv); }
 catch(Getopt::stdException &e)
  { opt.usage(); exit(e.code()); }

 // show tracebacks (upon unlikely crash) if -d given
 Signal sgn;
 if(opt[CHR(OPT_DBG)])
  sgn.install_all();

 DEBUGGABLE()
 DBG().use_ostream(cerr)                                        // debug settings
      .level(opt[CHR(OPT_DBG)]);

 try {
  if(opt[CHR(OPT_RDF)].hits() == 0)
   rpn.configure().resolve();
  else
   rpn.run();
 }
 catch(Rpnn::stdException & e) {
  DBG(0) DOUT() << "exception raised by: " << e.where() << endl;
  cerr << opt.prog_name() << " exception caught: " << e.what() << endl;
  exit(EXT_RPNEXP);
 }
 catch(exception &e)
  { cerr << opt.prog_name() << " caught exception - " << e.what() << endl; exit(EXT_STDEXP); }

 return 0;
}





template<typename T = double>
vector<T> str_to_num(string s, size_t min_req = 0) {
 // parse strings like "1,2,3" into the respective vector
 vector<T> v;

 for(size_t next, current = 0; current != string::npos; current = next) {
  next = s.find(',', current);
  if(next != string::npos) ++next;
  v.push_back(stod(s.substr(current, next - current)));
 }

 if(v.size() < min_req)
  throw std::length_error("minimum size requirement broken");
 return v;
}



Rpn & Rpn::configure(void) {
 // parse and configure rpn from all the options
 // topology:
 full_mesh(str_to_num<int>(opt()[CHR(OPT_TPG)].str()));
 DBG(0) DOUT() << "receptors: " << receptors_count() << endl;
 DBG(0) DOUT() << "effectors: " << effectors_count() << endl;
 DBG(0) DOUT() << "output neurons: " << output_neurons_count() << endl;

 // target error
 target_error(stod(opt()[CHR(OPT_ERR)].str()));
 DBG(0) DOUT() << "target error: " << target_error() << endl;

 // input normalization
 stringstream ss;
 vector<double> norm = str_to_num(opt()[CHR(OPT_INN)].str(), 2);
 //if(norm.front() != norm.back()) {
  normalize(norm.front(), norm.back());
  if(normalizing())
   ss << " [" << input_normalization().front().base() << " to " << std::showpos
      << input_normalization().front().base() + input_normalization().front().range() << "]";
 //}
 DBG(0) DOUT() << std::boolalpha << "normalize inputs: " << normalizing() << ss.str() << endl;

 // local minimum detection
 lm_detection(stoul(opt()[CHR(OPT_LMF)].str()) * synapse_count());
 DBG(0) DOUT() << "LM trail size: " << lm_detection() << endl;

 // cost function
 for(auto &cfe: cfm())
  if(opt()[CHR(OPT_COF)].str() == cfe.KEY)
   { cost_function(reinterpret_cast<Rpnn::c_func*>(cfe.VALUE)); break; }
 for(auto &cfe: cfm())
  if(cfe.VALUE == cost_function())
   { DBG(0) DOUT() << "cost function: cf_" << cfe.KEY << endl; break; }

 // effectors logistic
 if(opt()[CHR(OPT_ETF)].hits() > 0)
  for(auto &tfe: tfm())
   if(opt()[CHR(OPT_ETF)].str() == tfe.KEY) {
    for(auto ei = effectors_itr(); ei != neurons().end(); ++ei)
     ei->transfer_function(reinterpret_cast<rpnnNeuron::t_func*>(tfe.VALUE));
    break;
   }

 // output neurons logistic
 if(opt()[CHR(OPT_OTF)].hits() > 0)
  for(auto &tfe: tfm())
   if(opt()[CHR(OPT_OTF)].str() == tfe.KEY) {
    for(auto on = output_neurons_itr(); on != neurons().end(); ++on)
     on->transfer_function(reinterpret_cast<rpnnNeuron::t_func*>(tfe.VALUE));
    break;
   }

 // grow synapses
 for(auto &gs: opt()[CHR(OPT_GRS)]) {
  vector<size_t> s = str_to_num<size_t>(gs, 2);
  neuron(s[0]).grow_synapses(s[1]);
 }

 // grow synapses recursively between neurons N and M
 for(auto &gs: opt()[CHR(OPT_GSR)]) {
  vector<size_t> n = str_to_num<size_t>(gs, 2);                 // n holds range of neurons
  for(size_t sn = n.front(); sn <= n.back(); ++sn)              // sn/dn: source/destination neuron
   for(size_t dn = n.front(); dn <= n.back(); ++dn)
    if(sn != dn) neuron(sn).grow_synapses(dn);
 }

 // prune synapses
 for(auto &ps: opt()[CHR(OPT_PRS)]) {
  vector<size_t> s = str_to_num<size_t>(ps, 1);
  if(s.size() >= 2) neuron(s[0]).prune_synapses(s[1]);
  else neuron(s[0]).synapses().resize(1);
 }

 // parse GPM
 for(auto &ps: opt()[CHR(OPT_GPM)]) {                           // process each -P
  string pname = regex_replace(ps, std::regex{R"([=:].*)"}, "");// extract parameter's name
  auto fit = gpm().find(pname);                                 // fit = found iterator
  if(fit == gpm().cend())
   throw std::length_error("invalid paramenter");
  vector<double> pval = str_to_num(regex_replace(ps, std::regex{R"(^.*[=:])"}, ""));
  for(auto pi = pval.begin(); fit != gpm().cend() and pi != pval.end(); ++fit, ++pi)
   gpm(fit->KEY, *pi);
 }
 for(auto fit = gpm().begin(); fit != gpm().cend(); ++fit)
  DBG(0) DOUT() << "generic parameter " << fit->KEY << ": " << fit->VALUE << endl;

 // engage BLM
 if(opt()[CHR(OPT_BLM)].hits() == 0) bouncer(native_bouncer());
 else
  blm_.reduce_factor(gpm(STR(BLM_RDCE)))
      .thread_ctl().resize(stod(opt()[CHR(OPT_BLM)].str()));
 DBG(0) DOUT() << "blm (threads) engaged: " << (&bouncer() == &native_bouncer()?
                                                "no": to_string(blm_.thread_ctl().size())) << endl;

 if(opt()[CHR(OPT_ABC)].hits() > 0)
  bouncer().weight_updater(ub_);
 DBG(0)
  DOUT() << "bouncer: " << (opt()[CHR(OPT_ABC)].hits() > 0? "alternative":"native") << endl;

 // seed
 size_t my_seed = stoul(opt()[CHR(OPT_SED)].str());
 if(my_seed > 0) bouncer().seed(my_seed);
 DBG(0) DOUT() << "randomizer seed: "
               << (my_seed == 0?
                   "timer (" + to_string(bouncer().seed()) + ")": to_string(my_seed)) << endl;

 //setup separators
 sep_ = opt()[CHR(OPT_SPR)].str();

 DBG(0) DOUT() << "epochs to run: " << stoul(opt()[0].str()) << endl;

 return *this;
}


Rpn & Rpn::resolve(void) {
 // read inputs, plug into NN, converge, save to file upon successful convergence
 size_t ip = receptors_count(),                                 // number of input patterns
        tp = output_neurons_count();                            // number of target patterns

 // prepare containers for inputs/targets and read into them
 vvDouble inputs(ip);
 vvDouble targets(tp);
 read_patterns_(inputs, targets);

 load_patterns(inputs, targets);

 DBG(0) DOUT() << "training patterns read and loaded, starting convergence..." << endl;
 DBG(1) DOUT() << *this << endl;
 converge(stoul(opt()[0].str()));

 if(&bouncer() == &native_bouncer())                            // blm not engaged
  if(global_error() > target_error()) {
   cout << "Rpnn could not converge for " << epoch()
        << " epochs (err: " << global_error() << ") - not saving" << endl;
   exit(EXT_CDNTCV);
  }

 cout << (&bouncer() == &native_bouncer()?
           "Rpnn has converged at epoch ":
           "Rpnn found best local minimum, combined total epochs ")
       << epoch() << " with error: " << global_error() << endl;

 ofstream file(opt()[CHR(OPT_DMP)].str(), ios::binary);
 file << noskipws << Blob(blm_, *this, cnv_, sep_);             // dump NN to file

 DBG(0) DOUT() << "dumped rpn brains into file: " << opt()[CHR(OPT_DMP)].str() << endl;
 return *this;
}



bool Rpn::read_patterns_(vvDouble &ip, vvDouble &tp) {
 // when both params given read from cin input and target patterns until EOF
 // when only input param is given (interactive input read), read only 1 line into front's vector
 // return false (upon EOF) - meaningful only in the interactive mode
 if(cnv_.empty())                                               // this only would be the case in
  cnv_.resize(ip.size() + tp.size());                           // learning mode

 DBG(0) {
  if(&tp != &Rpn::dummy_tp)
   DOUT() << "start reading training patterns ("
          << ip.size() << " inputs + " << tp.size() << " outputs)..." << endl;
 }

 string str, dbgstr;
 while(getline(cin, str)) {
  str = regex_replace(str, std::regex{"[" + sep_ + "]+"}, " ");
  str = regex_replace(str, std::regex{R"(^ +)"}, "");
  if(str.empty()) continue;
  if(DBG()(0)) dbgstr = str;                                   // for later dbg output

  stringstream ss(str);
  auto cvi = cnv_.begin();
  auto ipi = ip.begin();
  auto tpi = tp.begin();
  while(getline(ss, str, ' ')) {
   if(ipi != ip.end()) {
    ipi->push_back((*cvi++)(str));
    if(&tp != &dummy_tp) ++ipi;                                 // learning mode
    else if(cvi == cnv_.end()) break;                           // training mode, prevent segfault
    continue;
   }
   if(&tp == &dummy_tp) break;                                  // training mode
   if(tpi != tp.end())
    tpi++->push_back((*cvi++)(str));
  }
  if(&tp == &dummy_tp)
   { DBG(0) DOUT() << "read input values: " << dbgstr << endl; return true; }
  if(ipi != ip.end() or tpi != tp.end())
   throw std::length_error("insufficient input");
 }

 DBG(0) DOUT() << "read " << ip.front().size() << " pattern(s)" << endl;
 return false;
}



Rpn & Rpn::run(void) {
 // read-restore rpn from file and activate its inputs

 Blob b(istream_iterator<char>(ifstream{opt()[CHR(OPT_RDF)].str(), ios::binary}>>noskipws),
        istream_iterator<char>{});                              // read from file to blob
 b.restore(blm_, *this, cnv_, sep_);                            // de-serialize blob

 for(auto &c: cnv_) c.roundup_toggle(opt()[CHR(OPT_RUP)].hits() > 0);

 DBG(0) DOUT() << "reinstated rpn brains from file: " << opt()[CHR(OPT_RDF)].str() << endl;
 DBG(1) DOUT() << *this << endl;
 DBG(0) DOUT() << "cnv_.size(): " << cnv_.size() << endl;

 // run input patterns
 size_t ip = receptors_count();
 DBG(0) DOUT() << "receptors_count: " << ip << endl;
 vvDouble inputs(1);

 while(read_patterns_(inputs)) {
  if(inputs.front().size() > ip) inputs.front().resize(ip);

  activate(inputs.front());

  // output activation result(s)
  string dlm("");
  for(size_t ons = output_neurons_count(), i = 0l; i < ons; ++i) {
   cout <<  dlm << cnv_[i + ip](out(i));
   dlm = " ";
  }
  cout << endl;
  inputs.front().clear();
 }

 return *this;
}

















