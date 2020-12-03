/*
 * Created by Dmitry Lyssenko
 *
 * few micro classes allowing fast median calculations and normalization
 * all classes is serialize'able and cout-able
 *
 * SYNOPSIS: (TBU)
 *
 */

#pragma once

#include <math.h>       // fmin/fmax
#include <vector>
#include <algorithm>
#include <numeric>      // accumulate
#include <type_traits>
#include "FifoDeque.hpp"
#include "Blob.hpp"     // SERDES interface
#include "Outable.hpp"
#include "dbg.hpp"
#include "extensions.hpp"





// a micro class for a fast median calculation. requires pushing data first and then
// median() call to calculate
//
// by default calculates true median, but also possible to introduce a bias - median(bias)
// bias: normally, when median is calculated it returns a middle value (distance 0.5) of the
// recorded array of values, bias, let specify other distance than default 0.5 (from ]0 to 1[ )

template<typename T>
class Median: public std::vector<T> {
 public:
    #define THROWREASON \
                Empty_data
    ENUMSTR(ThrowReason, THROWREASON)

    #define MMODE \
                Clear_manually, \
                Clear_on_read
    ENUMSTR(Mode, MMODE)

                        Median(void) = default;
                        Median(Mode x): m_{x} {}
    template<typename... Args>
                        Median(Args &&... args):
                         std::vector<T>{std::forward<Args>(args)...} {}

    Median &            clear_on_read(void)
                         { m_ = Clear_on_read; return *this; }
    Median &            clear_manually(void)
                         { m_ = Clear_manually; return *this; }
    Mode                mode(void) const
                         { return m_; }
    double              median(double nth = 0.5);
    double              average(void);

    SERDES(Median, m_)
    COUTABLE(Median, median_mode())
    // COUTABLE interfaces:
    const char *        median_mode(void) const
                         { return ENUMS(Mode, m_); }

    EXCEPTIONS(ThrowReason)

 protected:
    Mode                m_{Clear_manually};

 private:
};

template<typename T>
STRINGIFY(Median<T>::ThrowReason, THROWREASON)
template<typename T>
STRINGIFY(Median<T>::Mode, MMODE)
#undef THROWREASON
#undef MMODE



template<typename T>
double Median<T>::median(double nth) {
 // calculate median
 if(std::vector<T>::empty())
  throw EXP(Median::Empty_data);

 auto dummy = [&] { return true; };
 auto clear_on_exit = [&](bool) { if(mode() == Clear_on_read) std::vector<T>::clear(); };
 GUARD(dummy, clear_on_exit)

 if(std::vector<T>::size() & 0x1) {                             // if size() is odd.
  if(std::vector<T>::size() == 1)
   return std::vector<T>::front();
  else {
   nth_element(std::vector<T>::begin(),
               std::vector<T>::begin() + std::vector<T>::size() * nth, std::vector<T>::end());
   return std::vector<T>::operator[](std::vector<T>::size() * nth);
  }
 }
 // size() is even if here
 nth_element(std::vector<T>::begin(),
             std::vector<T>::begin() + std::vector<T>::size() * nth, std::vector<T>::end());

 double median = std::vector<T>::operator[](std::vector<T>::size() * nth);
 nth_element(std::vector<T>::begin(),
             std::vector<T>::begin() + std::vector<T>::size() * nth - 1, std::vector<T>::end());
 median += std::vector<T>::operator[](std::vector<T>::size() * nth - 1);
 return median /= 2.;
}



template<typename T>
double Median<T>::average(void) {
 // calculate median
 auto dummy = [&] { return true; };
 auto clear_on_exit = [&](bool) { if(mode() == Clear_on_read) std::vector<T>::clear(); };
 GUARD(dummy, clear_on_exit)

 double sum = 0;
 sum = std::accumulate(std::vector<T>::begin(), std::vector<T>::end(), sum);
 return sum / static_cast<double>(std::vector<T>::size());
}





// a micro class for a fast median calculation for fifoDeque container

template<typename T>
class MovingMedian: public fifoDeque<T> {
 public:
    using fifoDouble = fifoDeque<T>;

    #define THROWREASON \
                Empty_data
    ENUMSTR(ThrowReason, THROWREASON)

                        MovingMedian(void) = default;
    explicit            MovingMedian(size_t s): fifoDouble(s) {}
    template<typename... Args>
                        MovingMedian(Args &&... args):
                         fifoDouble{std::forward<Args>(args)...} {}

    MovingMedian &      reset(size_t s = 0)
                         { if(s > 0) fifoDouble::capacity(s); fifoDouble::clear(); return *this; }

    double              median(double nth = 0.5);

    double              average(void) {
                         double sum = 0;
                         for(auto d: *this) sum += d;
                         return fifoDouble::empty()?
                                 0.: sum / static_cast<double>(fifoDouble::size());
                        }

    EXCEPTIONS(ThrowReason)
};

template<typename T>
STRINGIFY(MovingMedian<T>::ThrowReason, THROWREASON)
#undef THROWREASON



template<typename T>
double MovingMedian<T>::median(double nth) {
 // calculate median
 if(fifoDouble::empty())
  throw EXP(MovingMedian<T>::Empty_data);

 if(fifoDouble::size() & 0x1) {                                 // size() is odd.
  if(fifoDouble::size() == 1) return fifoDouble::front();

  nth_element(fifoDouble::begin(),
              fifoDouble::begin() + fifoDouble::size() * nth, fifoDouble::end());
  return fifoDouble::operator[](fifoDouble::size() * nth);
 }

 // size() if even
 nth_element(fifoDouble::begin(),
             fifoDouble::begin() + fifoDouble::size() * nth, fifoDouble::end());
 double median = fifoDouble::operator[](fifoDouble::size() * nth);
 nth_element(fifoDouble::begin(),
             fifoDouble::begin() + fifoDouble::size() * nth - 1, fifoDouble::end());
 median += fifoDouble::operator[](fifoDouble::size() * nth - 1);
 return median /= 2.;
}






class Norm {
 public:
    #define THROWREASON \
                Norm_range_negative
    ENUMSTR(ThrowReason, THROWREASON)

                        Norm(double b = -1.0, double r = 2.0):      // default norm: -1 .. +1
                         nb_(b), nr_(r)
                         { if(nr_ <= 0.) throw EXP(Norm::Norm_range_negative); }

    double              base(void) const
                         { return nb_; }
    double              range(void) const
                         { return nr_; }
    Norm &              base_range(double b, double r) {
                         nb_ = b; nr_ = r;
                         if(nr_ <= 0.) throw EXP(Norm::Norm_range_negative);
                         return *this;
                        }
    double              normalize(double x) const
                         { return (x - base_) / range_ * nr_ + nb_; }
    double              denormalize(double x) const
                         { return (x - nb_) / nr_ * range_ + base_; }
    double              found_min(void) const
                         { return base_; }
    double              found_max(void) const
                         { return base_ + range_; }
    template<typename Itr>
    Norm &              find_bounds(Itr && begin, Itr && end);


    SERDES(Norm, base_, range_, nb_, nr_)
    COUTABLE(Norm, found_min(), found_max(), base(), range())
    EXCEPTIONS(ThrowReason)

 protected:

    double              base_;                                      // data base
    double              range_;                                     // data range
    double              nb_;                                        // base to normalize to
    double              nr_;                                        // range to normalize to
};

STRINGIFY(Norm::ThrowReason, THROWREASON)
#undef THROWREASON



template<typename Itr>
Norm & Norm::find_bounds(Itr && begin, Itr && end) {
 // find min bax of the range and memorize it
 if(begin == end) return *this;
 auto & min = base_;
 auto & max = range_;
 min = *begin;
 max = *begin;
 auto find_min_max = [&](double x) { min = std::min(min, x); max = std::max(max, x); };

 std::for_each(++begin, end, find_min_max);
 range_ = max - min;
 return *this;
}




















