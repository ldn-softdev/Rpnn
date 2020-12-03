/*
 *  fifoDeque.hpp
 *  a trivial tiny wrapper for Deque to similate a fixed size FIFO.
 */

#pragma once

#include <deque>
#include <type_traits>
#include "Blob.hpp"
#include "Outable.hpp"



template <class T>
class fifoDeque: public std::deque<T> {
 public:

                        fifoDeque(void) = default;
    explicit            fifoDeque(size_t s):s_{s} {}
    template<typename... Args>
                        fifoDeque(Args &&... args):
                         std::deque<T>{std::forward<Args>(args)...} {}


    bool                is_full(void) const
                         { return std::deque<T>::size() == capacity(); }
    size_t              capacity(void) const
                         { return s_; }
    fifoDeque &         capacity(size_t x)
                         { s_ = x; return *this; }

    // templated declaration to make an argument a universal reference
    template<class Arg>
    void                push_back(Arg && x) {
                         if(capacity() == 0) return;
                         std::deque<T>::push_back(std::forward<T>(x));
                         while(std::deque<T>::size() > capacity())
                          std::deque<T>::pop_front();
                        }
    template<class Arg>
    void                push_front(Arg && x) {
                         if(capacity() == 0) return;
                         std::deque<T>::push_front(std::forward<T>(x));
                         while(std::deque<T>::size() > capacity())
                          std::deque<T>::pop_back();
                        }

    SERDES(fifoDeque, s_, &fifoDeque::base_ref)
    COUTABLE(fifoDeque, capacity(), fifo())
    // COUTABLE interfaces:
  const std::deque<T> & fifo(void) const
                         { return *this; }

 protected:
    size_t              s_{0};                                  // capacity/max fifo size

    void                base_ref(Blob &b) const
                         { b.append(static_cast<const std::deque<T>&>(*this)); }
    void                base_ref(Blob &b)
                         { b.restore(static_cast<std::deque<T>&>(*this)); }
 private:

};
















