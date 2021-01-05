/*
 * Created by Dmitry Lyssenko.
 *
 * Blob class and SERDES interface (requires c++14 or above)
 *
 * Serdes interface provides serialization/deserialization ability for arbitrary
 * defined user classes. A user class ensures serialization/deserialization operations
 * by inclusion of SERDES macro as a public methods and enumerating all class
 * members which needs to be SERDES'ed.
 *
 * Blob class caters a byte-vector which holds serialized data.
 * There 2 basic interfaces to serialize data into the Blob and de-serialize (restore
 * data) from the blob:
 *
 *      Blob b;
 *      b.append(x, y, etc);
 *      b.restore(x, y, etc);
 *
 * What types of data that can be serialized/deserialized?
 *  1. any fundamental data (bool, char, int, double, etc)
 *  2. C-type arrays and STL containers of the above types
 *  3. any user-defined data classes which are defined with the SERDES interface
 *     - that includes recurrent data structures and those with (recursive) pointers
 *  4. pointers (see below on handling pointers)
 *
 * Blob class features 2 constructors (in addition to default):
 *  - 1. Constructor with data structures to be serialized:
 *
 *      Blob b(x, y, z);            // which is equal to: Blob b; b.append(x, y, z);
 *
 *  - 2. Constructor with iterators, the iterator must be a byte-type, it's particularly
 *       handy with istream_iterator's to load up data into the blob from external
 *       sources:
 *
 *      ifstream fs(file_name, ios::binary);                // input file with serialized data
 *      Blob b( istream_iterator<uint8_t>{fs>>noskipws},    // construct blob from istream
 *              istream_iterator<uint8_t>{} );
 *
 *
 * Other Blob methods:
 *      reset()         // required after append and/or before restore (if used in the same scope)
 *      clear()         // clears blob entirely - use after restore (to free up internal data)
 *      offset()        // returns current offset (after next append/restore operations)
 *      size()          // returns size of the blob itself (not size of the Blob object)
 *      empty()         // check if blob is empty (e.g. after clear())
 *      data()          // returns blob's data (string of serialized bytes)
 *      store()         // returns container (vector) of blob's data
 *      [c]begin()      // [c]begin() and [c]end() methods are re-published from
 *      [c]end()        // from store, for user's convenience
 *
 *
 * SERDES interface explained:
 *
 *  User class becomes SERDES'able when SERDES macro is included inside class definition
 *  as the public method, e.g.:
 *
 *  class SomeClass {
 *   public:
 *      ...
 *      SERDES(SomeClass, i_, s_, v_, ...)  // enumerate all what needs to be SERDES'ed
 *
 *   private:
 *      int                             i_;
 *      std::string                     s_;
 *      std::vector<SerdesableClass>    v_;
 *      ...
 *  };
 *
 *  ... // once a class defined like that, the class becomes SERDES'able:
 *  SomeClass x;
 *  Blob b(x);                          // same as: Blob b; b.append(x);
 *
 * SERDES macro declares 2 public methods and a constructor for the host class, which
 * let [re]storing class object to/from the blob:
 *
 *      serialize(...);
 *      deserialize(...);
 *
 * - serialize() accepts data types by const reference, while deserialize()
 *   does by reference, thus in order to enumerate SERDES'able object by a call,
 *   two methods must be provided: one for serialize() and one for deserialize()
 *   methods, e.g.:
 *
 *  class SomeClass {
 *   public:
 *      ...
 *      int             get_i(void) const { return i; }     // serialize requires const qualifier
 *      int &           get_i(void) { return &i; }          // used in deserialize (no const)
 *
 *      SERDES(SomeClass, get_i(), ...)     // enumerate all what needs to be serdes'ed
 *
 *   private:
 *      int             i_;
 *      ...
 *  };
 *
 *  if Blob is available for SomeClass, then data could be reinstated at the construction
 *  time:
 *
 *      Blob  b;
 *      // ... assume b was read from a file here, containing SomeClass' data;
 *      SomeClass x(b);                 // this constructor is provided by SERDES interface
 *
 *
 * SERDES interface handling POINTERS explained:
 *
 *  - SERDES interface in general handles 3 types of pointers:
 *    1. pointers pointing to any of internal data within the SERDES'ed, a.k.a. internal pointers;
 *       though pointers to individual chars within std::string are not accounted
 *    2. pointers handled by the user's pointer-providers - in this case user provides the way
 *       to [re]store pointers, SERDES only will call provider method in due time
 *    3. Functions (as pointers) and their containers (i.e., variables holding function pointers):
 *       functions only getting listed, while pointer containers will be SERDES'ed
 *
 *  - 1. internal pointers:
 *       when SERDES serializes data, it also memorizes all the addresses of the serialized data,
 *       e.g.: for vector<int>, all addresses of vector's integers will be accounted, as well as
 *       the address of the vector container itself. When facing a pointer, the pointer's address
 *       is checked against the built set of addresses and its reference is getting serialized.
 * *
 *  - 2. pointer providers:
 *       When class handles dynamic resources via pointers (though consider that to be an
 *       obsolete and generally bad practice), then SERDES needs pointer-provider methods:
 *       pointer-provider methods are void type, while accepts a reference to Blob:
 *
 *       class ResourceHandler {
 *        public:
 *           ...
 *           SERDES(ResourceHandler, x_, &ResourceHandler::ptr_provider)
 *
 *           void            ptr_provider(Blob &b) const {       // for serialize (const qualifier)
 *                            if(b.append_cntr(ptr_ != nullptr)) // store state of pointer
 *                             b.append_raw(ptr_, b.append_cntr(strlen(ptr_)));
 *                           }
 *
 *           void            ptr_provider(Blob &b) {             // de-serialize (restore) provider
 *                            bool is_saved;                     // check if ptr was actually saved
 *                            if(not b.restore_cntr(is_saved)) return;
 *                            size_t size;
 *                            ptr_ = new char[b.restore_cntr(size)];
 *                            b.restore_raw(ptr_, size);
 *                           }
 *
 *        private:
 *           someClass           x_;
 *           char []             ptr_;
 *       };
 *
 *       CAUTION: to play safe, always rely on append_raw/append_cntr, restore_raw/restore_cntr
 *                methods in pointer-providers - these methods do nothing but pure SERDES'ing,
 *                while regular append/restore also have some overhead for handling pointers
 *
 *  - 3. functions/function pointers:
 *       ...tbu
 *
 * - Functional implementation of pointers serialization/deserialization:
 *   Serialization of all pointers occurs at the end of top-level SERDES process, i.e., past all
 *   values serialization. This is done in order to ensure that all addresses (internal pointers)
 *   are collected. Two structures facilitate this process:
 *    1. par_ (preserved appended references) - it holds all internal addresses (pointers) mapped
 *       to unique references (the references are preserved in Blob instead of actual addresses),
 *       e.g.: nullptr -> 0, etc
 *    2. prv_ (pointer reference vector) vector holding addresses recorded from pointers, that
 *       at the end of top-level SERDES could be resolved by par_ and saved in Blob
 *   Deserialization occurs similar way: all the pointers restoration is delayed at the end of
 *   SERDES (deserialization phase): when pointer (for restoration) processed, nothing is extracted
 *   from the Blob, however pointer's address is memorized in the vector, and at the end -
 *   all addresses are restored
 *
 *
 * IMPORTANT: SERDES interface requires host class to have a default constructor
 *            (or one with a default argument), thus if none is declared, force
 *            an explicit default one (it could be private though)
 *
 *
 * File operations with Blob:
 *
 * Blob provides ostream operator:
 *
 *      Blob b(x, y, z);
 *      std::ofstream bf(file, std::ios::binary);
 *      bf << b;
 *
 * as well as istream's:
 *
 *      Blob b;
 *      std::ifstream bf(file, std::ios::binary);
 *      bf >> std::noskipws >> b;
 *
 *
 * for more usage examples see "gt_blob.cpp"
 *
 */

#pragma once

#include <vector>
#include <map>
#include <set>
#include <utility>      // std::forward, std::move,
#include <cstdint>      // uint8_t, ...
#include <memory>       // unique/shared_ptr
#include <type_traits>
#include "Outable.hpp"
#include "extensions.hpp"



// SERDES is a simplest form of interface allowing serialization/deserialization
// of any fundamental types of data, SERDES'able classes (those, defined with SERDES
// interface), arrays and containers of afore mentioned data types as well as
// data stored by pointers
// declaration syntax:
//
//      SERDES(MyClass, a, b, c)
//
// SERDES macro requires presence of a default constructor
// The above declaration unrolls into following code:
//
//      MyClass(Blob && __blob__): MyClass()                    // Constructor with r-value blob
//       { deserialize(__blob__);                               // for in-place deserialization
//      MyClass(Blob & __blob__): MyClass()                     // Constructor with blob reference
//       { deserialize(__blob__);                               // for in-place deserialization
//
//      void serialize(Blob &__blob__) const {
//          __blob__.__push_host__(this);
//         __blob__.preserve_addr(this);
//          __blob__.append(a);
//          __blob__.append(b);
//          __blob__.append(c);
//          __blob__.__pop_appended__();
//         if(__blob__.__is_top_serializer__()) __blob__.serialize_ptrs();
//      }
//      void deserialize(Blob &__blob__) {
//          __blob__.__push_host__(this);
//         __blob__.restore_addr(this);
//          __blob__.restore(a);
//          __blob__.restore(b);
//          __blob__.restore(c);
//          __blob__.__pop_restored__();
//         if(__blob__.__is_top_deserializer__()) __blob__.deserialize_ptrs();
//      }
//

#define __SERDES_APPEND__(ARG) __blob__.append(ARG);
#define __SERDES_RESTORE__(ARG) __blob__.restore(ARG);
#define SERDES(CLASS, Args...) \
            CLASS(Blob &&__blob__): CLASS() { deserialize(__blob__); } \
            CLASS(Blob &__blob__): CLASS() { deserialize(__blob__); } \
        void serialize(Blob &__blob__) const { \
         __blob__.__push_host__(this); \
         __blob__.preserve_addr(this); \
         MACRO_TO_ARGS(__SERDES_APPEND__, Args) \
         __blob__.__pop_appended__(); \
         if(__blob__.__is_top_serializer__()) \
          __blob__.serialize_ptrs(); \
        } \
        void deserialize(Blob &__blob__) { \
         __blob__.__push_host__(this); \
         __blob__.restore_addr(this); \
         MACRO_TO_ARGS(__SERDES_RESTORE__, Args) \
         __blob__.__pop_restored__(); \
         if(__blob__.__is_top_deserializer__()) \
          __blob__.deserialize_ptrs(); \
        }

#define ITR first                                               // semantic for emplacment pair
#define STATUS second                                           // instead of first/second
#define KEY first                                               // semantic for map's pair
#define VALUE second                                            // instead of first/second



class Blob {
  // befriending input/output operations:
  // dump blob into output stream
  friend std::ostream & operator<<(std::ostream &os, const Blob & self) {
                         os.write(reinterpret_cast<const char*>(self.data()), self.size());
                         return os;
                        }
  // read into blob from input stream
  // blob size is unknown (until parsed), so will read until end of stream
  friend std::istream & operator>>(std::istream &is, Blob & self) {
                         size_t is_pos = is.tellg(),
                                file_size = is.seekg(0, std::ios_base::end).tellg();
                         self.store().resize(file_size - is_pos);
                         is.seekg(is_pos)                           // restore position
                           .read(reinterpret_cast<char*>(self.data()), file_size - is_pos);
                         return is;
                        }

 public:

    using ptr_pair = std::pair<void*, const void*>;

    #define THROWREASON \
                inconsistent_data_while_appending, \
                inconsistent_data_while_restoring, \
                unknown_pointer_while_serializing, \
                unknown_reference_while_deserializing
    ENUMSTR(ThrowReason, THROWREASON)


                        Blob(void) = default;                   // DC

                        // constructors to dump target data into blob:
                        // Blob(src1, src2, ...);
                        // NB: not all data types are compatible, e.g. native arrays are not;
                        //     use Blob x; x.append(..) form instead
    template<typename... Args>
    explicit            Blob(Args &&... args)
                         { append(std::forward<Args>(args)...); }

                        // constructor to restore Blob
                        // handy to use with istream_iterator's:
    template<class T>
                        Blob(std::istream_iterator<T> first, std::istream_iterator<T> last)
                         { while(first != last) store().push_back(*first++); }


    // User interface to work with Blob:
                        // reset will clear all data structures except blob itself,
                        // as if it was just read
    Blob &              reset(void) {
                         offset_ = 0;
                         cptr_.clear();
                         vptr_.clear();
                         par_.clear(); par_.emplace(nullptr, 0);
                         prv_.clear();
                         rru_.clear(); rru_.emplace(nullptr);
                         rrp_.clear(); rrp_.emplace(0, ptr_pair{nullptr, nullptr});
                         apv_.clear();
                         return *this;
                        }
    Blob &              clear(void)
                         { blob_.clear(); return reset(); }

    size_t              offset(void) const { return offset_; }
    size_t              size(void) const { return blob_.size(); }
    bool                empty(void) const { return blob_.empty(); }

    uint8_t *           data(void) { return blob_.data(); }
    const uint8_t *     data(void) const { return blob_.data(); }
 std::vector<uint8_t> & store(void) { return blob_; }
    const std::vector<uint8_t> &
                        store(void) const { return blob_; }

                        // republish container's iterators
    auto                begin(void) { return store().begin(); }
    auto                begin(void) const { return store().begin(); }
    auto                cbegin(void) const { return store().cbegin(); }
    auto                end(void) { return store().end(); }
    auto                end(void) const { return store().end(); }
    auto                cend(void) const { return store().cend(); }

                        // methods to conclude append/restore operations (commit pointers)
    void                serialize_ptrs(void) {
                         for(auto ptr :prv_) {
                          auto it = par_.find(ptr);
                          if(it == par_.end())
                           throw EXP(unknown_pointer_while_serializing);
                          append_cntr(it->VALUE);
                         }
                        }
    void                deserialize_ptrs(void) {
                         for(auto aptr : apv_) {
                          size_t pref = restore_cntr();        // pref: preserved ptr reference
                          auto it = rrp_.find(pref);
                          if(it == rrp_.end())
                           throw EXP(unknown_reference_while_deserializing);
                          if(aptr.second == nullptr)
                           *aptr.first = it->VALUE.first;
                          else
                           *aptr.second = it->VALUE.second? it->VALUE.second: it->VALUE.first;
                         }
                        }

                        // these methods have to stay public (for SERDES interface),
                        // but normally shouldn't be used by user
    void                __push_host__(const void *ptr) { cptr_.push_back(ptr); }
    void                __push_host__(void *ptr) { vptr_.push_back(ptr); }
    void                __pop_appended__(void) { if(not cptr_.empty()) cptr_.pop_back(); }
    void                __pop_restored__(void) { if(not vptr_.empty()) vptr_.pop_back(); }
    bool                __is_top_serializer__(void) const { return cptr_.empty(); }
    bool                __is_top_deserializer__(void) const { return vptr_.empty(); }


    //
    // ... APPEND [to the blob] / RESTORE [from the blob] methods
    //

    // 0. generic, data-type agnostic:
    void                append_raw(const void *ptr, size_t s) {
                         for(size_t i = 0; i < s; ++i)
                          blob_.push_back(static_cast<const uint8_t*>(ptr)[i]);
                        }

    size_t              append_cntr(size_t s) {
                         uint8_t cs = counter_size_(s);
                         append_atomic_(cs);
                         switch(cs) {
                          case 0: append_atomic_(static_cast<uint8_t>(s)); break;
                          case 1: append_atomic_(static_cast<uint16_t>(s)); break;
                          case 2: append_atomic_(static_cast<uint32_t>(s)); break;
                          case 3: append_atomic_(static_cast<uint64_t>(s)); break;
                          default: throw EXP(inconsistent_data_while_appending);
                         }
                         return s;
                        }

    void                restore_raw(void *ptr, size_t s) {
                         for(size_t i = 0; i < s; ++i)
                          *(static_cast<uint8_t*>(ptr) + i) = blob_.at(offset_++);
                        }

    size_t              restore_cntr(void) {
                         uint8_t cs;
                         restore_atomic_(cs);
                         switch(cs) {
                          case 0: { uint8_t cnt; restore_atomic_(cnt); return cnt; }
                          case 1: { uint16_t cnt; restore_atomic_(cnt); return cnt; }
                          case 2: { uint32_t cnt; restore_atomic_(cnt); return cnt; }
                          case 3: { uint64_t cnt; restore_atomic_(cnt); return cnt; }
                          default: throw EXP(inconsistent_data_while_restoring);
                         }
                        }


    // 0a. variadic: 2 or more args given
    template<typename T, typename Q, typename... Args>
    void                append(const T & first, const Q & second, const Args &... Rest) {
                         cptr_.push_back(this);                 // ensure this is a top-serializer
                         append(first);
                         append(second, Rest...);
                         cptr_.pop_back();
                         if(__is_top_serializer__()) serialize_ptrs();
                        }

    template<typename T, typename Q, typename... Args>
    void                restore(T && first, Q && second, Args &&... rest) {
                         vptr_.push_back(this);                 // ensure this is a top-serializer
                         restore(std::forward<T>(first));
                         restore(std::forward<Q>(second), std::forward<Args>(rest)...);
                         vptr_.pop_back();
                         if(__is_top_deserializer__()) deserialize_ptrs();
                        }


    // 0b. SERDES'able class
    template<typename T>
    typename std::enable_if<std::is_member_function_pointer<decltype(& T::serialize)>::value,
                            const T &>::type
                        append(const T & v)
                         { v.serialize(*this); preserve_addr(&v); return v; }

    template<typename T>
    typename std::enable_if<std::is_member_function_pointer<decltype(& T::deserialize)>::value,
                            T &>::type
                        restore(T & v)
                         { v.deserialize(*this); restore_addr(&v); return v; }


    // 1a. atomic (fundamental) type:
    template<typename T>
    typename std::enable_if<std::is_fundamental<T>::value, const T &>::type
                        append(const T & v) {
                         append_raw(&v, sizeof(T));
                         preserve_addr(&v);
                         return v;
                        }

    template<typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T &>::type
                        restore(T &v) {
                         restore_raw(reinterpret_cast<char*>(&v), sizeof(T));
                         restore_addr(&v);
                         return v;
                        }


    // 1b. atomic (enum) type:
    template<typename T>
    typename std::enable_if<std::is_enum<T>::value, const T &>::type
                        append(const T & v) {
                         append_raw(&v, sizeof(int));
                         preserve_addr(&v);
                         return v;
                        }

    template<typename T>
    typename std::enable_if<std::is_enum<T>::value, T &>::type
                        restore(T &v) {
                         restore_raw(reinterpret_cast<char*>(&v), sizeof(int));
                         restore_addr(&v);
                         return v;
                        }


    // 2. containers
    // 2a. basic string:
    const std::basic_string<char> &
                        append(const std::basic_string<char> & c) {
                         append_cntr(c.size());
                         append_raw(c.data(), c.size());
                         preserve_addr(&c);
                         return c;
                        }

    std::basic_string<char> &
                        restore(std::basic_string<char> & c) {
                         c.resize(restore_cntr());
                         for(auto &v: c) restore_raw(&v, sizeof(char));
                         restore_addr(&c);
                         return c;
                        }


    // 2b. native arrays:
    template<typename T>
    typename std::enable_if<std::is_array<T>::value, const T &>::type
                        append(const T & v) {
                         for(int i = 0, s = sizeof(v) / sizeof(v[0]); i < s; ++i) append(v[i]);
                         return v;
                        }

    template<typename T>
    typename std::enable_if<std::is_array<T>::value, T &>::type
                        restore(T &v) {
                         for(int i = 0, s = sizeof(v) / sizeof(v[0]); i<s; ++i) restore(v[i]);
                         return v;
                        }


    // 2c. trivial container (e.g.: vector/list/dequeue/array):
    template<template<typename, typename> class Container, typename T, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<T>>::value,
                            const Container<T, A> &>::type
                        append(const Container<T, A> & c) {
                         append_cntr(c.size());
                         for(const T &v: c) append(v);
                         preserve_addr(&c);
                         return c;
                        }

    template<template<typename, typename> class Container, typename A, typename T>
    typename std::enable_if<std::is_same<A, std::allocator<T>>::value,
                            Container<T, A> &>::type
                        restore(Container<T, A> & c) {
                         c.resize(restore_cntr());
                         for(auto &v: c) restore(v);
                         restore_addr(&c);
                         return c;
                        }


    // 2d. trivial sorted containers (e.g.: std::set):
    template<template<typename, typename, typename> class Container,
             typename T, typename C, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<T>>::value and
                            std::is_same<C, std::less<T>>::value,
                            const Container<T, C, A> &>::type
                        append(const Container<T, C, A> & c) {
                         append_cntr(c.size());
                         for(auto &v: c) append(v);
                         preserve_addr(&c);
                         return c;
                        }

    template<template<typename, typename, typename> class Container,
             typename T, typename C, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<T>>::value and
                            std::is_same<C, std::less<T>>::value,
                            Container<T, C, A> &>::type
                        restore(Container<T, C, A> & c) {
                         size_t l = restore_cntr();
                         for(size_t i = 0; i < l; ++i) {
                          T v; restore(v);
                          update_addr_(&v, &*c.emplace(std::move(v)).ITR);
                         }
                         restore_addr(&c);
                         return c;
                        }


    // 2e. trivial unordered containers (e.g.: std::unordered_set):
    template<template<typename, typename, typename, typename> class Container,
             typename K, typename H, typename E, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<K>>::value and
                            std::is_same<H, std::hash<K>>::value and
                            std::is_same<E, std::equal_to<K>>::value,
                            const Container<K, H, E, A> &>::type
                        append(const Container<K, H, E, A> & c) {
                         append_cntr(c.size());
                         for(auto &v: c) append(v);
                         preserve_addr(&c);
                         return c;
                        }

    template<template<typename, typename, typename, typename> class Container,
             typename K, typename H, typename E, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<K>>::value and
                            std::is_same<H, std::hash<K>>::value and
                            std::is_same<E, std::equal_to<K>>::value,
                            Container<K, H, E, A> &>::type
                        restore(Container<K, H, E, A> & c) {
                         size_t l = restore_cntr();
                         for(size_t i = 0; i < l; ++i) {
                          K v; restore(v);
                          update_addr_(&v, &*c.emplace(std::move(v)).ITR);
                         }
                         restore_addr(&c);
                         return c;
                        }


    // 2f. unique_ptr/shared_ptr:
    template<typename T>
    const std::unique_ptr<T> &
                        append(const std::unique_ptr<T> & c) {
                         if(append_atomic_(c.get() != nullptr)) append(*c);
                         return c;
                        }

    template<typename T>
   std::unique_ptr<T> & restore(std::unique_ptr<T> & c) {
                         bool ptr_saved;
                         if(not restore_atomic_(ptr_saved)) return c;
                         auto v = new T;
                         restore(*v);
                         c.reset(v);
                         return c;
                        }


    // 3a. key-value sorted containers (e.g.: std::map):
    template<template<typename, typename, typename, typename> class Container,
             typename K, typename V, typename C, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<std::pair<const K, V>>>::value and
                            (std::is_same<C, bool (*)(const K&, const K&)>::value or
                             std::is_same<C, std::less<K>>::value),
                            const Container<K, V, C, A> &>::type
                        append(const Container<K, V, C, A> & c) {
                         append_cntr(c.size());
                         for(auto &v: c)
                          { append(v.KEY); append(v.VALUE); }
                         preserve_addr(&c);
                         return c;
                        }

    template<template<typename, typename, typename, typename> class Container,
             typename K, typename V, typename C, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<std::pair<const K, V>>>::value and
                            (std::is_same<C, bool (*)(const K&, const K&)>::value or
                             std::is_same<C, std::less<K>>::value),
                            Container<K, V, C, A> &>::type
                        restore(Container<K, V, C, A> & c) {
                         size_t l = restore_cntr();
                         for(size_t i = 0; i < l; ++i) {
                          K k; V v{}; restore(k);
                          auto ep = c.emplace(std::move(k), std::move(v));
                          update_addr_((const K *)&k, &ep.ITR->KEY);
                          restore(ep.ITR->VALUE);
                         }
                         restore_addr(&c);
                         return c;
                        }


    // 3b. key-value unordered containers (e.g.: std::unordered_map):
    template<template<typename, typename, typename, typename, typename> class Container,
             typename K, typename V, typename H, typename E, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<std::pair<const K, V>>>::value and
                            std::is_same<H, std::hash<K>>::value and
                            std::is_same<E, std::equal_to<K>>::value,
                            const Container<K, V, H, E, A> &>::type
                        append(const Container<K, V, H, E, A> & c) {
                         append_cntr(c.size());
                         for(auto &v: c)
                          { append(v.KEY); append(v.VALUE); }
                         preserve_addr(&c);
                         return c;
                        }

    template<template<typename, typename, typename, typename, typename> class Container,
             typename K, typename V, typename H, typename E, typename A>
    typename std::enable_if<std::is_same<A, std::allocator<std::pair<const K, V>>>::value and
                            std::is_same<H, std::hash<K>>::value and
                            std::is_same<E, std::equal_to<K>>::value,
                            Container<K, V, H, E, A> &>::type
                        restore(Container<K, V, H, E, A> & c) {
                         size_t l = restore_cntr();
                         for(size_t i = 0; i < l; ++i) {
                          K k; V v{}; restore(k);
                          auto ep = c.emplace(std::move(k), std::move(v));
                          update_addr_(&k, &ep.ITR->KEY);
                          restore(ep.ITR->VALUE);
                         }
                         restore_addr(&c);
                         return c;
                        }

    // 4. pointers
    // 4a. generic, non-func pointers
    template<typename T>
    typename std::enable_if<std::is_pointer<T>::value and not
                            std::is_function<std::remove_pointer_t<T>>::value, void>::type
                        append(T &ptr) {
                         // do not act upon facing a pointer, instead store it in prv_ and
                         // process all of them (form prv_) at the end of SERDES (serialization)
                         prv_.push_back(ptr);
                        }

    template<typename T>
    typename std::enable_if<not std::is_function<std::remove_pointer_t<T>>::value, void>::type
                        restore(T* &ptr) {
                         // do not act upon facing a pointer, instead store it in prv_ and
                         // process all of them (form prv_) at the end of SERDES (deserialization)
                         apv_.emplace_back(reinterpret_cast<void**>(&ptr), nullptr);
                        }

    template<typename T>
    typename std::enable_if<not std::is_function<std::remove_pointer_t<T>>::value, void>::type
                        restore(const T* &ptr) {
                         // do not act upon facing a pointer, instead store it in prv_ and
                         // process all of them (form prv_) at the end of SERDES (deserialization)
                         apv_.emplace_back(nullptr, reinterpret_cast<const void**>(&ptr));
                        }


    // 4b. function pointers
    // 4b.1 bare function pointer (e.g.: func): preserve address only
    template<typename T>
    typename std::enable_if<std::is_function<std::remove_pointer_t<T>>::value and not
                            std::is_pointer<T>::value, void>::type
                        append(T &ptr)
                         { preserve_addr(reinterpret_cast<void*>(ptr)); }
    template<typename T>
    typename std::enable_if<std::is_function<std::remove_pointer_t<T>>::value and not
                            std::is_pointer<T>::value, void>::type
                        restore(T &ptr)
                         { restore_addr(reinterpret_cast<void*>(ptr)); }

    // 4b.2 function pointer storage (e.g.: fuct_ptr)
    template<typename T>
    typename std::enable_if<std::is_pointer<T>::value and
                            std::is_function<std::remove_pointer_t<T>>::value, void>::type
                        append(T &ptr)
                         { prv_.push_back(reinterpret_cast<const void*>(ptr)); }
    template<typename T>
    typename std::enable_if<std::is_pointer<T>::value and
                            std::is_function<std::remove_pointer_t<T>>::value, void>::type
                        restore(T &ptr)
                         { apv_.emplace_back(reinterpret_cast<void**>(&ptr), nullptr); }

    // 4c. pointers callbacks of pointer-providers
    template<typename T>
    typename std::enable_if<std::is_member_function_pointer<void(T::*)(Blob &) const>::value,
                            void>::type
                        append(void (T::*cb)(Blob &) const)
                         { (static_cast<const T*>(cptr_.back())->*cb)(*this); }

    template<typename T>
    typename std::enable_if<std::is_member_function_pointer<void(T::*)(Blob &)>::value,
                            void>::type
                        restore(void (T::*cb)(Blob &))
                         { (static_cast<T*>(vptr_.back())->*cb)(*this); }



    // these methods facilitate building containers for accounting all pointers/references
    void                preserve_addr(const void *ptr) {
                         // create a new reference and attempt to add to par_
                         size_t newref = par_.size();
                         par_.emplace(ptr, newref);
                        }

    void                restore_addr(void *ptr) {
                         // create new ptr reference only if it's a new address (pointer)
                         if(rru_.emplace(ptr).STATUS == false)  // it's already known address
                          return;
                         size_t newref = rru_.size() - 1;       // new reference
                         rrp_.emplace(newref, ptr_pair{ptr, nullptr});  // register new reference
                        }

    void                restore_addr(const void *ptr) {
                         // create new const-ptr reference only if it's a new address (pointer)
                         if(rru_.emplace(ptr).STATUS == false)  // already known address
                          return;
                         size_t newref = rru_.size() - 1;       // register new reference
                         rrp_.emplace(newref, ptr_pair{nullptr, ptr});
                        }


    EXCEPTIONS(ThrowReason)

 protected:
    size_t              offset_{0};
   std::vector<uint8_t> blob_;

 private:
std::vector<const void*>cptr_;                                  // host cons-pointers for append
    std::vector<void*>  vptr_;                                  // host pointers for restore
    // host pointers are used to call user callbacks when processing SERDES arguments
    // they are used as stack: upon each SERDES call (serialize(..)/deserialize(..))
    // host object pointer is pushed into the respective vector and popped at the end

    std::map<const void*, size_t>
                        par_{{nullptr, 0}};                     // preserved appended references
    // these references are built during `append` phase - each (SERDES'able) object and fundamental
    // type's address that SERDES comes across is pushed here and its new reference created;
    std::vector<const void*>
                        prv_;                                // pointers reference vector
    // when SERDES faces pointer (during serialization) it does not act upon it immediately,
    // instead it preserves the address in this container and later (serialize_ptrs()) resolves
    // all the addresses against par_ container into the references which are dumped int the blob

  std::set<const void*> rru_{nullptr};                          // restore ref. of unique pointers
    // rru_ keeps track of all unique pointers entering while in restoring phase
    std::map<size_t, ptr_pair>
                        rrp_{{0, {nullptr, nullptr}}};          // restore reference pointer
    // rrp_ holds mappings for all references to pointers (addresses and const-addresses) faced
    // so far within a blob being restored - reverse meaning of par_
    std::vector<std::pair<void**, const void**>>
                        apv_;                                   // addresses of pointers vector
    // when SERDES faces a pointer (during deserialization) it does not act upon it immediately,
    // instead it preserves its location (i.e. pointer's address) in this container and later
    // (deserialize_ptrs()) resolves the read (from blob) ptr references for each location in apv_
    // via rrp_ lookup and saves it (the real pointer) into the preserved location


    // save up some space when serializing counters. any further optimization
    // of serialized space must be done with the compression algorithms
    template<typename T>
    typename std::enable_if<std::is_fundamental<T>::value, const T &>::type
                        append_atomic_(const T & v)
                         { append_raw(&v, sizeof(T)); return v; }

    template<typename T>
    typename std::enable_if<std::is_fundamental<T>::value, T &>::type
                        restore_atomic_(T &v)
                         { restore_raw(reinterpret_cast<char*>(&v), sizeof(T)); return v; }

    uint8_t             counter_size_(size_t cntr) {
                         #if ( __WORDSIZE == 64 )
                          static size_t bound[3]{1ul<<8, 1ul<<16, 1ul<<32};
                         #endif
                         #if ( __WORDSIZE == 32 )
                          static size_t bound[2]{1ul<<8, 1ul<<16};
                         #endif
                         for(int i = sizeof(bound) / sizeof(bound[0]) - 1; i >= 0; --i)
                          if(cntr >= bound[i])
                           return i + 1;
                         return 0;
                        }

    void                update_addr_(const void *dst, void *src) {
                         rru_.erase(dst);
                         rrp_.erase(rru_.size());
                         restore_addr(src);
                        }

    void                update_addr_(const void *dst, const void *src) {
                         rru_.erase(dst);
                         rrp_.erase(rru_.size());
                         restore_addr(src);
                        }
};

STRINGIFY(Blob::ThrowReason, THROWREASON)
#undef THROWREASON

#undef KEY
#undef STATUS
#undef ITR
#undef VALUE















