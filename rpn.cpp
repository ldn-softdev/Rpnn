/*
 *  rpn.currentp
 *
 * Created by Dmitry Lyssenko
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

#define PRGNAME "Resilient Propagation Neural network"
#define VERSION "0.01"
#define CREATOR "Dmitry Lyssenko"
#define EMAIL "ldn.softdev@gmail.com"

#define SIZE_T(N) static_cast<size_t>(N)

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

#define XSTR(X) #X
#define STR(X) XSTR(X)
#define XCHR(X) *#X
#define CHR(X) XCHR(X)

#define EXT_STDEXP 1                                        // std exception
#define EXT_RPNEXP 2                                        // Rpnn exception
#define EXT_CDNTCV 3                                        // Failed convergence


// map of predefined cost functions
map<const char*, void*> Cfm {
        {ENUMS(Rpnn::costFunc, Rpnn::Sse), reinterpret_cast<void*>(Rpnn::cf_Sse)},
        {ENUMS(Rpnn::costFunc, Rpnn::Xntropy), reinterpret_cast<void*>(Rpnn::cf_Xntropy)}
       };

// map predefined logistic functions
map<const char*, void*> Tfm {
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Sigmoid),
            reinterpret_cast<void*>(rpnnNeuron::tf_Sigmoid)},
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Tanh),
            reinterpret_cast<void*>(rpnnNeuron::tf_Tanh)},
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Tanhfast),
            reinterpret_cast<void*>(rpnnNeuron::tf_Tanhfast)},
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Relu),
            reinterpret_cast<void*>(rpnnNeuron::tf_Relu)},
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Softplus),
            reinterpret_cast<void*>(rpnnNeuron::tf_Softplus)},
        {ENUMS(rpnnNeuron::tFunc, rpnnNeuron::Softmax),
            reinterpret_cast<void*>(rpnnNeuron::tf_Softmax)},
       };


void configure_rpn(Rpnn &, Getopt &);
void run_convergence(Rpnn &, Getopt &);
void run_preserved(Rpnn &, Getopt &);





int main(int argc, char* argv[]) {

 Getopt opt;
 opt.prolog("\n" PRGNAME "\nVersion " VERSION " (built on " __DATE__ \
            "), developed by " CREATOR " (" EMAIL ")\n");

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
 opt[0].desc("epochs to run convergence").name("epochs").bind("100000");

 string epilog{R"(
available cost functions:
{CF}
available logistic functions:
{LF}
- parameters N,M are zero based, the index 0 refers to a reserved neuron "the one"
- factor for option -)" STR(OPT_LMF) R"( is multiple of the total count of synapses (weights)
)"};

 // update epilogue with predefined costs & logistics
 stringstream ss;
 for(auto &cf: Cfm) ss << "\to " << cf.first << endl;
 epilog = regex_replace(epilog, std::regex{R"(\{CF\})"}, ss.str());
 ss.str("");
 for(auto &tf: Tfm) ss << "\to " << tf.first << endl;
 epilog = regex_replace(epilog, std::regex{R"(\{LF\})"}, ss.str());
 opt.epilog(epilog.c_str());

 // parse options
 try { opt.parse(argc, argv); }
 catch(Getopt::stdException &e)
  { opt.usage(); exit(e.code()); }

 // show tracebacks (upon unlikely crash) if -d given
 Signal sgn;
 if(opt[CHR(OPT_DBG)])
  sgn.install_all();

 Rpnn rpn;                                                      // our hero
 DEBUGGABLE()
 DBG().use_ostream(cerr)                                        // debug settings
      .level(opt[CHR(OPT_DBG)])
      .severity(rpn);

 try {
  if(opt[CHR(OPT_RDF)].hits() == 0) {
   configure_rpn(rpn, opt);
   run_convergence(rpn, opt);
  }
  else run_preserved(rpn, opt);
 }
 catch(Rpnn::stdException & e) {
  DBG(0) DOUT() << "exception raised by: " << e.where() << endl;
  cerr << opt.prog_name() << " exception caught: " << e.what() << endl;
  exit(EXT_RPNEXP);
 }
 catch(exception &e)
  { cerr << opt.prog_name() << " caught exception " << e.what() << endl; exit(EXT_STDEXP); }

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



void configure_rpn(Rpnn &rpn, Getopt &opt) {
 // parse and configure rpn from all the options
 DEBUGGABLE()
 // topology:
 rpn.full_mesh(str_to_num<int>(opt[CHR(OPT_TPG)].str()));
 DBG(0) DOUT() << "receptors: " << distance(++rpn.neurons().begin(), rpn.effectors()) << endl;
 DBG(0) DOUT() << "effectors: " << distance(rpn.effectors(), rpn.neurons().end()) << endl;
 DBG(0) DOUT() << "output neurons: " << distance(rpn.output_neurons(), rpn.neurons().end()) << endl;

 // target error
 rpn.target_error(stod(opt[CHR(OPT_ERR)].str()));
 DBG(0) DOUT() << "target error: " << rpn.target_error() << endl;

 // input normalization
 vector<double> norm = str_to_num(opt[CHR(OPT_INN)].str(), 2);
 if(norm.front() != norm.back())
  rpn.normalize(norm.front(), norm.back());
 DBG(0) DOUT() << std::boolalpha << "normalize inputs: " << rpn.normalizing() << endl;

 // local minimum detection
 rpn.lm_detection(stoul(opt[CHR(OPT_LMF)].str()) * rpn.synapses_count());
 DBG(0) DOUT() << "LM trail size: " << rpn.lm_detection() << endl;

 // cost function
 for(auto &cfe: Cfm)
  if(opt[CHR(OPT_COF)].str() == cfe.first)
   { rpn.cost_function(reinterpret_cast<Rpnn::c_func*>(cfe.second)); break; }
 for(auto &cfe: Cfm)
  if(cfe.second == rpn.cost_function())
   { DBG(0) DOUT() << "cost function: cf_" << cfe.first << endl; break; }

 // effectors logistic
 if(opt[CHR(OPT_ETF)].hits() > 0)
  for(auto &tfe: Tfm)
   if(opt[CHR(OPT_ETF)].str() == tfe.first) {
    for(auto ei = rpn.effectors(); ei != rpn.neurons().end(); ++ei)
     ei->transfer_function(reinterpret_cast<rpnnNeuron::t_func*>(tfe.second));
    break;
   }

 // output neurons logistic
 if(opt[CHR(OPT_OTF)].hits() > 0)
  for(auto &tfe: Tfm)
   if(opt[CHR(OPT_OTF)].str() == tfe.first) {
    for(auto on = rpn.output_neurons(); on != rpn.neurons().end(); ++on)
     on->transfer_function(reinterpret_cast<rpnnNeuron::t_func*>(tfe.second));
    break;
   }

 // grow synapses
 for(auto &gs: opt[CHR(OPT_GRS)]) {
  vector<size_t> s = str_to_num<size_t>(gs, 2);
  rpn.neuron(s[0]).grow_synapses(s[1]);
 }

 // grow synapses recursively between N and M
 for(auto &gs: opt[CHR(OPT_GSR)]) {
  vector<size_t> n = str_to_num<size_t>(gs, 2);                 // n holds range of neurons
  for(size_t sn = n.front(); sn <= n.back(); ++sn)              // sn/dn: source/destination neuron
   for(size_t dn = n.front(); dn <= n.back(); ++dn)
    if(sn != dn) rpn.neuron(sn).grow_synapses(dn);
 }

 // prune synapses
 for(auto &ps: opt[CHR(OPT_PRS)]) {
  vector<size_t> s = str_to_num<size_t>(ps, 2);
  rpn.neuron(s[0]).decay_synapses(s[1]);
 }

 // seed
 size_t seed = stoul(opt[CHR(OPT_SED)].str());
 if(seed > 0) rpn.bouncer().seed(seed);
 DBG(0) DOUT() << "randomizer seed: "
               << (seed == 0?
                   "timer (" + to_string(rpn.bouncer().seed()) + ")": to_string(seed)) << endl;
 DBG(0) DOUT() << "epochs to run: " << stoul(opt[0].str()) << endl;
 DBG(1) DOUT() << rpn << endl;
}



void run_convergence(Rpnn &rpn, Getopt &opt) {
 // read inputs, plug into NN, converge, save to file upon successful convergence
 size_t ip = distance(++rpn.neurons().begin(), rpn.effectors()),// number of input patterns
        tp = distance(rpn.output_neurons(), rpn.neurons().end());// number of target patterns

 vector<vector<double>> inputs(ip);
 vector<vector<double>> targets(tp);

 DEBUGGABLE()
 DBG(0) DOUT() << "start reading training patterns..." << endl;
 string str;
 while(getline(cin, str)) {
  str = regex_replace(str, std::regex{R"([\s=,]+)"}, " ");
  if(str == " " or str.empty()) continue;                       // blank line
  stringstream ss(str);
  auto ipi = inputs.begin();
  auto tpi = targets.begin();
  while(getline(ss, str, ' ')) {
   if(ipi != inputs.end())
    { ipi++->push_back(stod(str)); continue; }
   if(tpi != targets.end()) tpi++->push_back(stod(str));
  }
  if(ipi != inputs.end() or tpi != targets.end())
   throw std::length_error("inconsistent pattern length");
 }

 rpn.load_patterns(inputs, targets);
 DBG(0) DOUT() << "training patterns read and loaded, starting convergence..." << endl;
 rpn.converge(stoul(opt[0].str()));

 if(rpn.global_error() > rpn.target_error()) {
  cerr << "Rpnn could not converge for " << rpn.epoch()
       << " epochs (err: " << rpn.global_error() << ") - not saving" << endl;
  exit(EXT_CDNTCV);
 }
 cout << "Rpnn has converged at epoch " << rpn.epoch()
      << " with error: " << rpn.global_error() << endl;

 ofstream file(opt[CHR(OPT_DMP)].str(), ios::binary);
 file << noskipws << Blob(rpn);

 DBG(0) DOUT() << "dumped rpn brains into file: " << opt[CHR(OPT_DMP)].str() << endl;
}



void run_preserved(Rpnn &rpn, Getopt &opt) {
 // read-restore rpn from file and activate its inputs
 DEBUGGABLE()

 Blob b(istream_iterator<char>(ifstream{opt[CHR(OPT_RDF)].str(), ios::binary}>>noskipws),
        istream_iterator<char>{});
 b.restore(rpn);
 DBG(0) DOUT() << "reinstated rpn brains from file: " << opt[CHR(OPT_RDF)].str() << endl;
 DBG(1) DOUT() << rpn << endl;

 // run input patterns
 size_t ip = distance(++rpn.neurons().begin(), rpn.effectors());// number of receptors
 vector<double> inputs;
 inputs.reserve(ip);
 
 string str;
 while(getline(cin, str)) {
  str = regex_replace(str, std::regex{R"([\s,]+)"}, " ");
  if(str == " " or str.empty()) continue;                       // blank line
  stringstream ss(str);
  inputs.clear();
  while(getline(ss, str, ' '))                                  // parse read line
   inputs.push_back(stod(str));
  if(inputs.size() != ip)
   throw std::length_error("insufficient inputs");

  rpn.activate(inputs);

  string dlm("");
  for(auto ons = distance(rpn.output_neurons(), rpn.neurons().end()), i = 0l; i < ons; ++i) {
   cout <<  dlm << (opt[CHR(OPT_RUP)].hits() == 0?  rpn.out(i): floor(rpn.out(i) + 0.5));
   dlm = " ";
  }
  cout << endl;
 }
}










