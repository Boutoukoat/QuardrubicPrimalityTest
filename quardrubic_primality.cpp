// -----------------------------------------------------------------------
// Quardrubic primality check
// -----------------------------------------------------------------------

#include <algorithm>
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <x86intrin.h>

using namespace std;

typedef unsigned __int128 uint128_t;
typedef signed __int128 int128_t;

static bool uint64_is_perfect_square(uint64_t a);
static bool uint128_is_perfect_square(uint128_t a);

// --------------------------------------------------------------------------------------
//
// C optimized implementations of simple utilities
//
// --------------------------------------------------------------------------------------

// randomness generation with period 2^64
static uint64_t uint64_rnd(void)
{
    static uint64_t s = 0x1234567890123456ull;
    s = s * 137 + 13;
    return (s >> 13) ^ (s << 13) ^ s;
}

// (u + v) mod n
// Assume u+v < 2*n
static inline uint64_t uint64_add_mod(uint64_t u, uint64_t v, uint64_t n)
{
#if 1
    uint64_t t = u + v;
    if (t < u)
    {
        return t - n;
    }
    return (t >= n) ? t - n : t;
#else
    uint128_t t = u + v;
    return t < n ? t : t - n;
#endif
}

// (u) mod n
// (u << 64 + v) mod n
static inline uint64_t uint64_long_mod(uint64_t u, uint64_t v, uint64_t n)
{
#ifdef __x86_64__
    uint64_t r, a;
    asm("divq %4" : "=d"(r), "=a"(a) : "0"(u), "1"(v), "r"(n) : "flags");
    return r;
#else
    uint128_t t = ((uint128_t)u << 64) + v;
    return t % n;
#endif
}

// (u) mod n
static inline uint64_t uint128_long_mod(uint128_t u, uint64_t n)
{
#ifdef __x86_64__
    uint64_t r = (uint64_t)(u >> 64), a = (uint64_t)u;
    asm("divq %4" : "=d"(r), "=a"(a) : "0"(r), "1"(a), "r"(n) : "flags");
    return r;
#else
    return u % n;
#endif
}

// (u) mod m
static inline void uint128_divrem(uint64_t *q, uint64_t *r, uint128_t u, uint64_t n)
{
#ifdef __x86_64__
    uint64_t hi = (uint64_t)(u >> 64), lo = (uint64_t)u;
    asm("divq %4" : "=d"(*r), "=a"(*q) : "0"(hi), "1"(lo), "r"(n) : "flags");
#else
    *q = u / n;
    *r = u % n;
#endif
}

static inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t n)
{
#ifdef __x86_64__
    uint64_t r;
    asm("mulq %3" : "=d"(r), "=a"(a) : "1"(a), "r"(b));
    asm("divq %4" : "=d"(r), "=a"(a) : "0"(r), "1"(a), "r"(n) : "flags");
    return r;
#else
    uint128_t tmp = (uint128_t)a * b;
    tmp %= n;
    return (uint64_t)tmp;
#endif
}

static inline uint64_t square_mod(uint64_t a, uint64_t n)
{
#ifdef __x86_64__
    uint64_t r;
    asm("mulq %2" : "=d"(r), "=a"(a) : "1"(a));
    asm("divq %4" : "=d"(r), "=a"(a) : "0"(r), "1"(a), "r"(n) : "flags");
    return r;
#else
    uint128_t tmp = (uint128_t)a * a;
    tmp %= n;
    return tmp;
#endif
}

// count leading zeroed bits
static inline uint64_t uint64_lzcnt(uint64_t a)
{
#ifdef __x86_64__
    uint64_t r;
    asm("lzcntq %1,%0" : "=r"(r) : "r"(a));
    return r;
#else
    return __builtin_clzll(a);
#endif
}

// count trailing zeroed bits
static inline uint64_t uint64_tzcnt(uint64_t a)
{
#ifdef __x86_64__
    uint64_t r;
    asm("tzcntq %1,%0" : "=r"(r) : "r"(a));
    return r;
#else
    return __builtin_ctzll(a);
#endif
}

// count trailing zeroed bits
static inline uint64_t uint128_tzcnt(uint128_t a)
{
    uint64_t t = (uint64_t)a;
    if (t)
    {
        return uint64_tzcnt(t);
    }
    t = (a >> 64);
    return 64 + uint64_tzcnt(t);
}

// simple floor(log_2) function
// log(1) = 0
// log(2) = 1
// log(3) = 1
// log(4) = 2 ...
static inline uint64_t uint64_log_2(uint64_t a)
{
    return 63 - uint64_lzcnt(a);
}

static inline uint64_t uint128_log_2(uint128_t a)
{
    uint64_t t = (uint64_t)(a >> 64);
    if (t)
    {
        return 127 - uint64_lzcnt(t);
    }
    else
    {
        return 63 - uint64_lzcnt((uint64_t)a);
    }
}

// --------------------------------------------------------------------------------------
//
// Math functions
//
// --------------------------------------------------------------------------------------

// assume a >= 0
// assume odd b > 0
// assume a < 2^62 or b < 2^62
static int uint64_jacobi(uint64_t a, uint64_t b)
{
    static char cols[]
                    [64] = {
                        {1},                                                                               // a=0
                        {1, 1},                                                                            // a=1
                        {(char)-1, (char)-1},                                                              // a=2
                        {0, (char)-1, (char)-1},                                                           // a=3
                        {1, 1, 1, 1},                                                                      // a=4
                        {(char)-1, 0, (char)-1, 1, 1},                                                     // a=5
                        {0, 1, (char)-1, 0, (char)-1, (char)-1},                                           // a=6
                        {1, (char)-1, 0, 1, (char)-1, (char)-1, (char)-1},                                 // a=7
                        {(char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1},                              // a=8
                        {0, 1, 1, 0, 1, 1, 0, 1, 1},                                                       // a=9
                        {1, 0, (char)-1, 1, (char)-1, 1, 0, (char)-1, (char)-1, (char)-1},                 // a=10
                        {(char)-1, 1, 1, 1, 0, (char)-1, (char)-1, (char)-1, 1, (char)-1, (char)-1},       // a=11
                        {0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1},                  // a=12
                        {1, (char)-1, (char)-1, 1, (char)-1, 0, (char)-1, 1, (char)-1, (char)-1, 1, 1, 1}, // a=13
                        {(char)-1, 1, 0, 1, 1, 1, (char)-1, (char)-1, (char)-1,
                         0, (char)-1, 1, (char)-1, (char)-1},                                             // a=14
                        {0, 0, 1, 0, 1, (char)-1, 0, 1, (char)-1, 0, (char)-1, 0, 0, (char)-1, (char)-1}, // a=15
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                                 // a=16
                        {(char)-1, (char)-1, (char)-1, 1, (char)-1, 1, 1, 0, 1, 1, (char)-1, 1, (char)-1, (char)-1,
                         (char)-1, 1, 1}, // a=17
                        {0, (char)-1, 1, 0, (char)-1, (char)-1,
                         0, 1, (char)-1, 0, 1, 1, 0, (char)-1, 1, 0, (char)-1, (char)-1}, // a=18
                        {1, 1, (char)-1, 1, (char)-1, (char)-1, 1, 1, 0, (char)-1, (char)-1,
                         1, 1, (char)-1, 1, (char)-1, (char)-1, (char)-1, (char)-1}, // a=19
                        {(char)-1, 0, (char)-1, 1, 1, (char)-1, 0, (char)-1, 1, 1,
                         (char)-1, 0, (char)-1, 1, 1, (char)-1, 0, (char)-1, 1, 1}, // a=20
                        {0, 1, 0,        0,        (char)-1, (char)-1, 0, 1, (char)-1, 0, (char)-1,
                         1, 0, (char)-1, (char)-1, 0,        0,        1, 0, 1,        1}, // a=21
                        {1, (char)-1, 1, 1,        0,        1, (char)-1, (char)-1, (char)-1, 1, (char)-1, 1, 1,
                         1, (char)-1, 0, (char)-1, (char)-1, 1, (char)-1, (char)-1, (char)-1}, // a=22
                        {(char)-1, (char)-1, 1,        1, 1,        1,        1,        (char)-1,
                         1,        (char)-1, 0,        1, (char)-1, 1,        (char)-1, (char)-1,
                         (char)-1, (char)-1, (char)-1, 1, 1,        (char)-1, (char)-1}, // a=23
                        {0, 1, (char)-1, 0, (char)-1, (char)-1, 0, (char)-1, 1, 0, 1, 1,
                         0, 1, (char)-1, 0, (char)-1, (char)-1, 0, (char)-1, 1, 0, 1, 1},            // a=24
                        {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1}, // a=25
                        {(char)-1, 1,        (char)-1, 1,        1,        0,        (char)-1, 1,        1,
                         1,        1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1,
                         0,        (char)-1, (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1}, // a=26
                        {0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1,
                         0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1}, // a=27
                        {1, (char)-1, 0, 1, (char)-1, (char)-1, (char)-1, (char)-1, 1, 0, (char)-1, 1, 1, 1,
                         1, (char)-1, 0, 1, (char)-1, (char)-1, (char)-1, (char)-1, 1, 0, (char)-1, 1, 1, 1}, // a=28
                        {(char)-1, 1, 1,        1, (char)-1, 1, (char)-1, (char)-1, (char)-1, (char)-1,
                         1,        1, (char)-1, 0, (char)-1, 1, 1,        (char)-1, (char)-1, (char)-1,
                         (char)-1, 1, (char)-1, 1, 1,        1, (char)-1, 1,        1}, // a=29
                        {0, 0, 1, 0, (char)-1, 1,        0, 1,        1, 0, (char)-1, 0, 0, 1,        (char)-1,
                         0, 0, 1, 0, (char)-1, (char)-1, 0, (char)-1, 1, 0, (char)-1, 0, 0, (char)-1, (char)-1}, // a=30
                        {1,        1, (char)-1, 1,        1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,
                         1,        1, (char)-1, 0,        1, (char)-1, (char)-1, (char)-1, 1,        1,        1,
                         (char)-1, 1, (char)-1, (char)-1, 1, (char)-1, (char)-1, (char)-1, (char)-1}, // a=31
                        {(char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1,
                         (char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1,
                         (char)-1, (char)-1, 1, 1, (char)-1, (char)-1, 1, 1}, // a=32
                        {0, (char)-1, (char)-1, 0, 0,        (char)-1, 0, 1, (char)-1, 0, (char)-1, 1,
                         0, 1,        1,        0, 1,        1,        0, 1, (char)-1, 0, (char)-1, 1,
                         0, (char)-1, 0,        0, (char)-1, (char)-1, 0, 1, 1}, // a=33
                        {1,        1,        (char)-1, 1,        1,        (char)-1, 1,       0,        (char)-1,
                         (char)-1, (char)-1, 1,        1,        1,        (char)-1, 1,       (char)-1, 1,
                         (char)-1, (char)-1, (char)-1, 1,        1,        1,        0,       (char)-1, 1,
                         (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1}, // a=34
                        {(char)-1, 0,        0,        1,        (char)-1, 1, 0,        1,        1,
                         0,        1,        0,        (char)-1, 1,        1, 1,        0,        (char)-1,
                         (char)-1, (char)-1, 1,        0,        (char)-1, 0, (char)-1, (char)-1, 0,
                         (char)-1, 1,        (char)-1, 0,        0,        1, (char)-1, (char)-1}, // a=35
                        {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                         0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1}, // a=36
                        {1,        (char)-1, 1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, 1,
                         (char)-1, 1,        1, (char)-1, (char)-1, 1,        (char)-1, 0,        (char)-1, 1,
                         (char)-1, (char)-1, 1, 1,        (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1,
                         1,        1,        1, (char)-1, 1,        1,        1}, // a=37
                        {(char)-1, (char)-1, (char)-1, 1, 1,        1,        1,        1,       0,        1,
                         (char)-1, 1,        (char)-1, 1, 1,        (char)-1, 1,        1,       (char)-1, (char)-1,
                         1,        (char)-1, (char)-1, 1, (char)-1, 1,        (char)-1, 0,       (char)-1, (char)-1,
                         (char)-1, (char)-1, (char)-1, 1, 1,        1,        (char)-1, (char)-1}, // a=38
                        {0,        1, 1,        0,        (char)-1, 0,        0,        (char)-1, 1,        0,
                         1,        1, 0,        (char)-1, 1,        0,        1,        (char)-1, 0,        1,
                         (char)-1, 0, (char)-1, 1,        0,        (char)-1, (char)-1, 0,        (char)-1, 1,
                         0,        0, 1,        0,        (char)-1, (char)-1, 0,        (char)-1, (char)-1}, // a=39
                        {1,        0, (char)-1, 1,        (char)-1, 1,        0, (char)-1, (char)-1, (char)-1,
                         (char)-1, 0, 1,        (char)-1, 1,        (char)-1, 0, 1,        1,        1,
                         1,        0, (char)-1, 1,        (char)-1, 1,        0, (char)-1, (char)-1, (char)-1,
                         (char)-1, 0, 1,        (char)-1, 1,        (char)-1, 0, 1,        1,        1}, // a=40
                        {(char)-1, 1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1,
                         1,        1, 1,        (char)-1, (char)-1, 1,        1,        (char)-1, 1,
                         1,        0, 1,        1,        (char)-1, 1,        1,        (char)-1, (char)-1,
                         1,        1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1,
                         (char)-1, 1, (char)-1, 1,        1}, // a=41
                        {0, (char)-1, 0,        0,        1,        1, 0,        1,        1,       0,        (char)-1,
                         1, 0,        1,        (char)-1, 0,        0, (char)-1, 0,        1,       (char)-1, 0,
                         1, 0,        0,        1,        (char)-1, 0, (char)-1, 1,        0,       (char)-1, (char)-1,
                         0, (char)-1, (char)-1, 0,        0,        1, 0,        (char)-1, (char)-1}, // a=42
                        {1, (char)-1, 1,        1,        (char)-1, 1,        (char)-1, 1,        1,
                         1, (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1,
                         1, 1,        0,        (char)-1, (char)-1, 1,        1,        1,        1,
                         1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1,
                         1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1}, // a=43
                        {(char)-1, 1,        1,        1,        0,        (char)-1, (char)-1, (char)-1, 1,
                         (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, 0,        1,        1,
                         1,        (char)-1, 1,        1,        (char)-1, 1,        1,        1,        0,
                         (char)-1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1,
                         (char)-1, 0,        1,        1,        1,        (char)-1, 1,        1}, // a=44
                        {0, 0, (char)-1, 0, 1, (char)-1, 0, (char)-1, 1, 0, (char)-1, 0, 0, 1, 1,
                         0, 0, (char)-1, 0, 1, (char)-1, 0, (char)-1, 1, 0, (char)-1, 0, 0, 1, 1,
                         0, 0, (char)-1, 0, 1, (char)-1, 0, (char)-1, 1, 0, (char)-1, 0, 0, 1, 1}, // a=45
                        {1,        1,        1,        1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, 1,
                         0,        1,        1,        (char)-1, (char)-1, (char)-1, 1,        1,        (char)-1, 1,
                         (char)-1, 1,        (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1, 1,        1,
                         1,        (char)-1, (char)-1, 0,        (char)-1, 1,        1,        (char)-1, 1,        1,
                         (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, (char)-1}, // a=46
                        {(char)-1, (char)-1, (char)-1, 1,        1,        (char)-1, 1,        1,
                         1,        1,        1,        1,        (char)-1, (char)-1, 1,        (char)-1,
                         1,        1,        1,        (char)-1, 1,        (char)-1, 0,        1,
                         (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, 1,
                         1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1,
                         (char)-1, (char)-1, 1,        1,        1,        (char)-1, (char)-1}, // a=47
                        {0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1,
                         0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1,
                         0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1, (char)-1, 0, 1, 1}, // a=48
                        {1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                         1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1}, // a=49
                        {(char)-1, 0, 1,        1,        (char)-1, (char)-1, 0, 1,        (char)-1, (char)-1,
                         1,        0, (char)-1, (char)-1, 1,        1,        0, (char)-1, 1,        1,
                         (char)-1, 0, 1,        1,        (char)-1, (char)-1, 0, 1,        (char)-1, (char)-1,
                         1,        0, (char)-1, (char)-1, 1,        1,        0, (char)-1, 1,        1,
                         (char)-1, 0, 1,        1,        (char)-1, (char)-1, 0, 1,        (char)-1, (char)-1}, // a=50
                        {0, 1,        1,        0, (char)-1, 1,        0, 0,        (char)-1, 0, (char)-1, 1,
                         0, 1,        1,        0, 1,        (char)-1, 0, 1,        (char)-1, 0, 1,        1,
                         0, (char)-1, (char)-1, 0, 1,        (char)-1, 0, 1,        (char)-1, 0, (char)-1, (char)-1,
                         0, (char)-1, 1,        0, 1,        0,        0, (char)-1, 1,        0, (char)-1, (char)-1,
                         0, (char)-1, (char)-1}, // a=51
                        {1, (char)-1, (char)-1, 1, (char)-1, 0, (char)-1, 1, (char)-1, (char)-1, 1, 1, 1,
                         1, (char)-1, (char)-1, 1, (char)-1, 0, (char)-1, 1, (char)-1, (char)-1, 1, 1, 1,
                         1, (char)-1, (char)-1, 1, (char)-1, 0, (char)-1, 1, (char)-1, (char)-1, 1, 1, 1,
                         1, (char)-1, (char)-1, 1, (char)-1, 0, (char)-1, 1, (char)-1, (char)-1, 1, 1, 1}, // a=52
                        {(char)-1, (char)-1, 1,        1,        1,        1,        1,        1,        (char)-1,
                         (char)-1, (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,
                         (char)-1, (char)-1, 1,        (char)-1, 1,        1,        (char)-1, 0,        (char)-1,
                         1,        1,        (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1,
                         (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        1,
                         1,        1,        1,        1,        (char)-1, (char)-1, 1,        1}, // a=53
                        {0, 1, (char)-1, 0,        (char)-1, (char)-1, 0,        (char)-1, 1,        0,        1,
                         1, 0, 1,        (char)-1, 0,        (char)-1, (char)-1, 0,        (char)-1, 1,        0,
                         1, 1, 0,        1,        (char)-1, 0,        (char)-1, (char)-1, 0,        (char)-1, 1,
                         0, 1, 1,        0,        1,        (char)-1, 0,        (char)-1, (char)-1, 0,        (char)-1,
                         1, 0, 1,        1,        0,        1,        (char)-1, 0,        (char)-1, (char)-1}, // a=54
                        {1,        0, (char)-1, 1,        0,        1,        0, 1,        1,        (char)-1,
                         1,        0, 1,        (char)-1, (char)-1, 0,        0, (char)-1, 1,        (char)-1,
                         (char)-1, 0, 1,        1,        1,        (char)-1, 0, 1,        (char)-1, (char)-1,
                         (char)-1, 0, 1,        1,        (char)-1, 1,        0, 0,        1,        1,
                         (char)-1, 0, (char)-1, 1,        (char)-1, (char)-1, 0, (char)-1, 0,        (char)-1,
                         1,        0, (char)-1, (char)-1, (char)-1}, // a=55
                        {(char)-1, 1,        0,        1,        1,        1,        (char)-1, (char)-1,
                         (char)-1, 0,        (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1,
                         0,        (char)-1, (char)-1, (char)-1, 1,        1,        1,        0,
                         1,        (char)-1, 1,        1,        (char)-1, 1,        0,        1,
                         1,        1,        (char)-1, (char)-1, (char)-1, 0,        (char)-1, 1,
                         (char)-1, (char)-1, 1,        (char)-1, 0,        (char)-1, (char)-1, (char)-1,
                         1,        1,        1,        0,        1,        (char)-1, 1,        1}, // a=56
                        {0, (char)-1, 1,        0, (char)-1, (char)-1, 0, (char)-1, 0,        0, (char)-1, 1,
                         0, 1,        (char)-1, 0, (char)-1, (char)-1, 0, 1,        1,        0, (char)-1, 1,
                         0, 1,        1,        0, 1,        1,        0, 1,        (char)-1, 0, 1,        1,
                         0, (char)-1, (char)-1, 0, (char)-1, 1,        0, 1,        (char)-1, 0, 0,        (char)-1,
                         0, (char)-1, (char)-1, 0, 1,        (char)-1, 0, 1,        1}, // a=57
                        {1,        (char)-1, 1,        1,        1,        (char)-1, (char)-1, (char)-1, 1,
                         1,        1,        1,        1,        0,        (char)-1, 1,        (char)-1, 1,
                         (char)-1, (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1,
                         1,        (char)-1, 1,        1,        1,        (char)-1, 1,        1,        (char)-1,
                         1,        1,        (char)-1, 1,        (char)-1, 1,        0,        (char)-1, (char)-1,
                         (char)-1, (char)-1, (char)-1, 1,        1,        1,        (char)-1, (char)-1, (char)-1,
                         1,        (char)-1, (char)-1, (char)-1}, // a=58
                        {(char)-1, 1,        (char)-1, 1,        1,        (char)-1, (char)-1, 1,        (char)-1,
                         1,        1,        1,        (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1,
                         1,        1,        1,        1,        1,        1,        (char)-1, 1,        1,
                         1,        0,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1,
                         (char)-1, (char)-1, (char)-1, 1,        1,        1,        (char)-1, (char)-1, 1,
                         (char)-1, (char)-1, (char)-1, 1,        (char)-1, 1,        1,        (char)-1, (char)-1,
                         1,        (char)-1, 1,        (char)-1, (char)-1}, // a=59
                        {0, 0,        1,        0, 1,        (char)-1, 0, 1,        (char)-1, 0, (char)-1, 0,
                         0, (char)-1, (char)-1, 0, 0,        (char)-1, 0, (char)-1, 1,        0, (char)-1, 1,
                         0, 1,        0,        0, 1,        1,        0, 0,        1,        0, 1,        (char)-1,
                         0, 1,        (char)-1, 0, (char)-1, 0,        0, (char)-1, (char)-1, 0, 0,        (char)-1,
                         0, (char)-1, 1,        0, (char)-1, 1,        0, 1,        0,        0, 1,        1}, // a=60
                        {1,        1,        (char)-1, 1,        (char)-1, 1,        1,        (char)-1, 1,
                         (char)-1, (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1,
                         1,        1,        (char)-1, 1,        1,        1,        (char)-1, (char)-1, (char)-1,
                         1,        (char)-1, 0,        (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,
                         1,        1,        (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1,
                         (char)-1, 1,        1,        (char)-1, (char)-1, 1,        (char)-1, 1,        1,
                         (char)-1, 1,        (char)-1, 1,        1,        1,        1}, // a=61
                        {(char)-1, (char)-1, (char)-1, 1,        (char)-1, 1,        1,        (char)-1, 1,
                         1,        1,        1,        (char)-1, 1,        0,        1,        1,        1,
                         (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        1,        1,        1,
                         (char)-1, 1,        1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1,
                         (char)-1, 1,        1,        1,        (char)-1, 1,        (char)-1, (char)-1, (char)-1,
                         0,        (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, 1,        (char)-1,
                         (char)-1, 1,        (char)-1, 1,        1,        1,        (char)-1, (char)-1}, // a=62
                        {0, (char)-1, 0, 0, (char)-1, (char)-1, 0,        (char)-1, 1, 0, (char)-1, 1, 0, 1,
                         1, 0,        0, 1, 0,        (char)-1, (char)-1, 0,        1, 0, 0,        1, 1, 0,
                         1, (char)-1, 0, 1, (char)-1, 0,        (char)-1, (char)-1, 0, 0, (char)-1, 0, 1, 1,
                         0, (char)-1, 0, 0, (char)-1, (char)-1, 0,        (char)-1, 1, 0, (char)-1, 1, 0, 1,
                         1, 0,        0, 1, 0,        (char)-1, (char)-1}, // a=63
                    };
    static char rows[][64] = {
        {1, 1},                                                                            // b=1
        {(char)-1, 0, 1},                                                                  // b=3
        {(char)-1, (char)-1, 1, 0, 1},                                                     // b=5
        {1, (char)-1, 1, (char)-1, (char)-1, 0, 1},                                        // b=7
        {1, 0, 1, 1, 0, 1, 1, 0, 1},                                                       // b=9
        {(char)-1, 1, 1, 1, (char)-1, (char)-1, (char)-1, 1, (char)-1, 0, 1},              // b=11
        {(char)-1, 1, 1, (char)-1, (char)-1, (char)-1, (char)-1, 1, 1, (char)-1, 1, 0, 1}, // b=13
        {1, 0, 1, 0, 0, (char)-1, 1, 0, 0, (char)-1, 0, (char)-1, (char)-1, 0, 1},         // b=15
        {1, (char)-1, 1, (char)-1, (char)-1, (char)-1, 1, 1, (char)-1, (char)-1, (char)-1, 1, (char)-1, 1, 1, 0,
         1}, // b=17
        {(char)-1, (char)-1, 1, 1, 1, 1, (char)-1, 1, (char)-1, 1, (char)-1, (char)-1, (char)-1, (char)-1, 1, 1,
         (char)-1, 0, 1}, // b=19
        {(char)-1, 0, 1, 1, 0, 0, (char)-1, 0, (char)-1, (char)-1, 0,
         (char)-1, 0, 0, 1, 1, 0, (char)-1, 1, 0,        1}, // b=21
        {1,        1,        1, (char)-1, 1, (char)-1, 1,        1,        (char)-1, (char)-1, 1, 1,
         (char)-1, (char)-1, 1, (char)-1, 1, (char)-1, (char)-1, (char)-1, (char)-1, 0,        1}, // b=23
        {1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1},               // b=25
        {(char)-1, 0,        1, (char)-1, 0,        1, (char)-1, 0,        1, (char)-1, 0,        1, (char)-1, 0,
         1,        (char)-1, 0, 1,        (char)-1, 0, 1,        (char)-1, 0, 1,        (char)-1, 0, 1}, // b=27
        {(char)-1, (char)-1, 1,        1,        1,        1,        (char)-1, 1,        (char)-1, (char)-1,
         (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1,
         1,        1,        1,        1,        (char)-1, (char)-1, 1,        0,        1}, // b=29
        {1,        (char)-1, 1,        1,        (char)-1, 1,        1,        1, 1,        (char)-1, (char)-1,
         (char)-1, 1,        (char)-1, 1,        (char)-1, 1,        1,        1, (char)-1, (char)-1, (char)-1,
         (char)-1, 1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, 0, 1}, // b=31
        {1,        0, 1, (char)-1, 0, (char)-1, 1,        0, (char)-1, 0, 0, (char)-1, (char)-1, 0, 1, 1, 0, (char)-1,
         (char)-1, 0, 0, (char)-1, 0, 1,        (char)-1, 0, (char)-1, 1, 0, 1,        1,        0, 1}, // b=33
        {(char)-1, 1, 1, 0, (char)-1, 0,        (char)-1, 1, 0,        1,        1,        1,
         0,        0, 1, 1, (char)-1, (char)-1, 0,        0, (char)-1, (char)-1, (char)-1, 0,
         (char)-1, 1, 0, 1, 0,        (char)-1, (char)-1, 1, (char)-1, 0,        1}, // b=35
        {(char)-1, 1,        1,        (char)-1, (char)-1, 1,        (char)-1, 1,        1,        1,
         1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, 1,
         (char)-1, (char)-1, (char)-1, 1,        1,        1,        1,        (char)-1, 1,        (char)-1,
         (char)-1, 1,        1,        (char)-1, 1,        0,        1}, // b=37
        {1, 0,        1, 1,        0,        (char)-1, 1, 0,        1,        1, 0, 0, (char)-1, 0,
         1, (char)-1, 0, (char)-1, 1,        0,        1, (char)-1, 0,        1, 0, 0, (char)-1, (char)-1,
         0, (char)-1, 1, 0,        (char)-1, (char)-1, 0, (char)-1, (char)-1, 0, 1}, // b=39
        {1,        (char)-1, 1,        1,        (char)-1, (char)-1, 1,        1, 1, (char)-1, (char)-1,
         (char)-1, (char)-1, (char)-1, 1,        (char)-1, 1,        (char)-1, 1, 1, (char)-1, 1,
         (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1, 1, 1,        (char)-1,
         (char)-1, 1,        1,        (char)-1, 1,        1,        0,        1}, // b=41
        {(char)-1, (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1, 1,        1,        1,        (char)-1,
         1,        1,        1,        1,        1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, 1,
         1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1,
         1,        1,        (char)-1, 1,        (char)-1, 1,        1,        (char)-1, 0,        1}, // b=43
        {(char)-1, 0, 1, 0, 0, (char)-1, (char)-1, 0, 0, 1, 0, (char)-1, 1, 0, 1,
         (char)-1, 0, 1, 0, 0, (char)-1, (char)-1, 0, 0, 1, 0, (char)-1, 1, 0, 1,
         (char)-1, 0, 1, 0, 0, (char)-1, (char)-1, 0, 0, 1, 0, (char)-1, 1, 0, 1}, // b=45
        {1,        1,        1,        (char)-1, 1,        1, 1,        1,        (char)-1, (char)-1,
         1,        (char)-1, 1,        (char)-1, 1,        1, 1,        (char)-1, (char)-1, 1,
         (char)-1, (char)-1, 1,        1,        (char)-1, 1, 1,        (char)-1, (char)-1, (char)-1,
         1,        (char)-1, 1,        (char)-1, 1,        1, (char)-1, (char)-1, (char)-1, (char)-1,
         1,        (char)-1, (char)-1, (char)-1, (char)-1, 0, 1}, // b=47
        {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
         1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1}, // b=49
        {(char)-1, 0,        1,        1,        0, (char)-1, (char)-1, 0,        (char)-1, 1, 0,        1, 1,        0,
         1,        0,        0,        1,        1, 0,        (char)-1, 1,        0,        1, (char)-1, 0, (char)-1, 1,
         0,        (char)-1, (char)-1, 0,        0, (char)-1, 0,        (char)-1, (char)-1, 0, (char)-1, 1, 0,        1,
         1,        0,        (char)-1, (char)-1, 0, 1,        (char)-1, 0,        1}, // b=51
        {(char)-1, (char)-1, 1,        (char)-1, 1,        1,        (char)-1, 1,        1,        1,        (char)-1,
         1,        (char)-1, 1,        1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, (char)-1,
         1,        1,        (char)-1, (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1,
         (char)-1, 1,        1,        1,        (char)-1, 1,        (char)-1, 1,        1,        1,        (char)-1,
         1,        1,        (char)-1, 1,        (char)-1, (char)-1, 1,        0,        1}, // b=53
        {1,        (char)-1, 1,        0,        (char)-1, 1,        1,        1,        0,        0, (char)-1,
         1,        1,        0,        1,        1,        1,        (char)-1, 0,        (char)-1, 0, (char)-1,
         (char)-1, 0,        1,        (char)-1, 1,        (char)-1, 0,        1,        1,        0, 1,
         0,        1,        (char)-1, (char)-1, (char)-1, 0,        (char)-1, (char)-1, 1,        0, 0,
         (char)-1, (char)-1, (char)-1, 1,        0,        (char)-1, 1,        (char)-1, (char)-1, 0, 1}, // b=55
        {1,        0, 1,        (char)-1, 0, 1,        1,        0, (char)-1, (char)-1, 0, (char)-1,
         1,        0, 1,        (char)-1, 0, 0,        (char)-1, 0, (char)-1, (char)-1, 0, 1,
         (char)-1, 0, 1,        1,        0, (char)-1, 1,        0, (char)-1, (char)-1, 0, (char)-1,
         0,        0, (char)-1, 1,        0, 1,        (char)-1, 0, (char)-1, (char)-1, 0, 1,
         1,        0, (char)-1, 1,        0, 1,        1,        0, 1}, // b=57
        {(char)-1, 1,        1,        1,        (char)-1, 1,        (char)-1, 1,        (char)-1, (char)-1,
         1,        (char)-1, (char)-1, 1,        1,        1,        (char)-1, 1,        1,        1,
         1,        (char)-1, (char)-1, 1,        1,        1,        1,        1,        (char)-1, (char)-1,
         (char)-1, (char)-1, (char)-1, 1,        1,        (char)-1, (char)-1, (char)-1, (char)-1, 1,
         (char)-1, (char)-1, (char)-1, 1,        1,        (char)-1, 1,        1,        (char)-1, 1,
         (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, 0,        1}, // b=59
        {(char)-1, 1, 1,        1,        (char)-1, (char)-1, (char)-1, 1,        (char)-1, (char)-1, 1,
         1,        1, 1,        1,        (char)-1, (char)-1, 1,        1,        (char)-1, 1,        (char)-1,
         (char)-1, 1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, (char)-1, 1,
         (char)-1, 1, (char)-1, (char)-1, 1,        (char)-1, 1,        1,        (char)-1, (char)-1, 1,
         1,        1, 1,        1,        (char)-1, (char)-1, 1,        (char)-1, (char)-1, (char)-1, 1,
         1,        1, (char)-1, 1,        0,        1}, // b=61
        {1, 0,        1, (char)-1, 0,        0, 1, 0, (char)-1, 1, 0,        (char)-1, 0, 0,
         1, (char)-1, 0, (char)-1, (char)-1, 0, 1, 1, 0,        1, (char)-1, 0,        0, 1,
         0, (char)-1, 1, 0,        (char)-1, 0, 0, 1, (char)-1, 0, (char)-1, (char)-1, 0, 1,
         1, 0,        1, (char)-1, 0,        0, 1, 0, (char)-1, 1, 0,        (char)-1, 0, 0,
         1, (char)-1, 0, (char)-1, (char)-1, 0, 1}, // b=63
    };

    if (a >= b)
    {
        a %= b;
    }

    // 0 <= a < b with b odd
    if (a < 3)
    {
        // (0/b) = (b == 1)
        if (a == 0)
        {
            return (b == 1) ? 1 : 0;
        }
        // (1/b) = 1
        if (a == 1)
        {
            return 1;
        }
        // (2/b) = (b % 8 == 3 || b % 8 == 5) ? -1 : 1;
        return (((b >> 2) ^ (b >> 1)) & 1) ? -1 : 1;
    }

    // 3 <= a < b with b odd
    if (a < 64)
    {
        if (a & 2)
        {
            uint64_t k = ((b - 3) >> 1) % (2 * a);
            return k < a ? cols[a][k] : -cols[a][k - a];
        }
        else
        {
            uint64_t k = ((b - 3) >> 1) % a;
            return cols[a][k];
        }
    }

    // 64 <= a < b with b odd and a < 2^62
    int t = 1;
    if ((a & 3) == 0)
    {
        b = b % a;
    }
    else if ((a & 3) == 2 && b < 3 * a)
    {
        if (b < 2 * a)
        {
            t = (b & 2) ? -t : t;
            b = b - a;
        }
        else
        {
            t = (b & 1) ? -t : t;
            b = b - 2 * a;
        }
    }
    else
    {
        b = b % (4 * a);
        if (b >= 2 * a)
        {
            b -= 2 * a;
            t = (a & 2) ? -t : t;
        }
    }

    // 64 <= a, b odd
    if (b < 64)
    {
        uint64_t k = (a - 2) % b;
        return (t == -1) ? -rows[b >> 1][k] : rows[b >> 1][k];
    }

    unsigned c = (b >> 2) ^ (b >> 1);
    while (a)
    {
        unsigned v = uint64_tzcnt(a);
        a >>= v;
        t = (c & v & 1) ? -t : t;

        if (a < b)
        {
            uint64_t k = a;
            a = b;
            b = k;
            t = ((a & b & 3) == 3) ? -t : t;
            c = (b >> 2) ^ (b >> 1);
        }

        a -= b;
    }
    return (b == 1) ? t : 0;
}

static int int64_kronecker(int64_t a, int64_t b)
{
    int v;
    int t = 1;
    if (b == 0)
    {
        return (a == -1 || a == 1) ? 1 : 0;
    }

    // make b positive
    // K(a,-b) = K(a,b)*sgn(a)
    if (b < 0)
    {
        b = -b;
        t = (a < 0) ? -t : t;
    }

    // make a positive
    // K(-a,b) = K(a,b)*(-1)^(b'>>1)
    if (a < 0)
    {
        a = -a;
        v = uint64_tzcnt(b);
        t = ((b >> v) & 2) ? -t : t;
    }

    // make b odd
    if ((b & 1) == 0)
    {
        if ((a & 1) == 0)
        {
            return 0;
        }
        v = uint64_tzcnt(b);
        if (v & 1)
        {
            t = (((a >> 1) ^ (a >> 2)) & 1) ? -t : t;
        }
        b >>= v;
    }

    // make a odd
    if ((a & 1) == 0)
    {
        if (a == 0)
        {
            return (b == 1) ? 1 : 0;
        }
        v = uint64_tzcnt(a);
        if (v & 1)
        {
            t = (((b >> 1) ^ (b >> 2)) & 1) ? -t : t;
        }
        a >>= v;
    }

    v = uint64_jacobi(a, b);
    return (t < 0) ? -v : v;
}

static int int128_kronecker(int128_t a, int128_t b)
{
    unsigned v;
    int t = 1;
    if (a == 0)
    {
        return (b == -1 || b == 1) ? 1 : 0;
    }

    if (b == 0)
    {
        return (a == -1 || a == 1) ? 1 : 0;
    }

    // K(a,-b) = K(a,b)*sgn(a)
    if (b < 0)
    {
        b = -b;
        t = (a < 0) ? -t : t;
    }
    // K(-a,b) = K(a,b)*(-1)^(b'>>1)
    if (a < 0)
    {
        a = -a;
        v = uint128_tzcnt(b);
        t = ((b >> v) & 2) ? -t : t;
    }

    // make b odd
    if ((b & 1) == 0)
    {
        if ((a & 1) == 0)
        {
            return 0;
        }
        v = uint128_tzcnt(b);
        if (v & 1)
        {
            t = (((a >> 1) ^ (a >> 2)) & 1) ? -t : t;
        }
        b >>= v;
    }

    unsigned c = (b >> 2) ^ (b >> 1);
    while ((a | b) >> 62 != 0)
    {
        v = uint128_tzcnt(a);
        a >>= v;
        t = (c & v & 1) ? -t : t;

        if (a < b)
        {
            uint128_t k = a;
            a = b;
            b = k;
            t = ((a & b & 3) == 3) ? -t : t;
            c = (b >> 2) ^ (b >> 1);
        }
        a -= b;

        if (a < 3)
        {
            if (a == 0)
            {
                return (b == 1) ? t : 0;
            }
            if (a == 1)
            {
                return t;
            }
            return (c & 1) ? -t : t;
        }
    }

    // make b odd
    if ((b & 1) == 0)
    {
        if ((a & 1) == 0)
        {
            return 0;
        }
        v = uint128_tzcnt(b);
        if (v & 1)
        {
            t = (((a >> 1) ^ (a >> 2)) & 1) ? -t : t;
        }
        b >>= v;
    }

    if (b == 1)
    {
        return t;
    }

    int r = uint64_jacobi((uint64_t)a, (uint64_t)b);
    return (t < 0) ? -r : r;
}

// return smallest factor of n < 2^64 , exact if n < 157*157, return 1 if none is found.
static uint64_t uint64_small_factor(uint64_t n)
{
    if (n <= 152)
    {
        // return smallest factor of n <= 151, or 1 if none is found.
        uint8_t stooopid_factor_table[] = {
            1, 1, 1, 1, 2, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 1, 2, 1,  2, 3, 2, 1, 2, 5, 2, 3, 2,  1, 2,
            1, 2, 3, 2, 5, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 1, 2, 7, 2,  3, 2, 1, 2, 5, 2, 3, 2, 1,  2, 1,
            2, 3, 2, 5, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 7, 2, 1, 2, 3,  2, 1, 2, 5, 2, 3, 2, 1, 2,  7, 2,
            3, 2, 5, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 3, 2,  1, 2, 5, 2, 3, 2, 7, 2, 11, 2, 3,
            2, 5, 2, 1, 2, 3, 2, 1, 2, 7, 2, 3, 2, 1, 2, 1, 2, 3, 2, 11, 2, 5, 2, 3, 2, 1, 2, 1, 2};
        return stooopid_factor_table[n];
    }
    if (!(n & 1))
        return 2;
    if ((uint64_t)(n * 0xaaaaaaaaaaaaaaabull) <= 0x5555555555555555ull)
        return 3;
    if ((uint64_t)(n * 0xcccccccccccccccdull) <= 0x3333333333333333ull)
        return 5;
    if ((uint64_t)(n * 0x6db6db6db6db6db7ull) <= 0x2492492492492492ull)
        return 7;
    if ((uint64_t)(n * 0x2e8ba2e8ba2e8ba3ull) <= 0x1745d1745d1745d1ull)
        return 11;
    if ((uint64_t)(n * 0x4ec4ec4ec4ec4ec5ull) <= 0x13b13b13b13b13b1ull)
        return 13;
    if ((uint64_t)(n * 0xf0f0f0f0f0f0f0f1ull) <= 0x0f0f0f0f0f0f0f0full)
        return 17;
    if ((uint64_t)(n * 0x86bca1af286bca1bull) <= 0x0d79435e50d79435ull)
        return 19;
    if ((uint64_t)(n * 0xd37a6f4de9bd37a7ull) <= 0x0b21642c8590b216ull)
        return 23;
    if ((uint64_t)(n * 0x34f72c234f72c235ull) <= 0x08d3dcb08d3dcb08ull)
        return 29;
    if ((uint64_t)(n * 0xef7bdef7bdef7bdfull) <= 0x0842108421084210ull)
        return 31;
    if (n < 37 * 37)
        return 1; // prime
    if ((uint64_t)(n * 0x14c1bacf914c1badull) <= 0x06eb3e45306eb3e4ull)
        return 37;
    if ((uint64_t)(n * 0x8f9c18f9c18f9c19ull) <= 0x063e7063e7063e70ull)
        return 41;
    if ((uint64_t)(n * 0x82fa0be82fa0be83ull) <= 0x05f417d05f417d05ull)
        return 43;
    if ((uint64_t)(n * 0x51b3bea3677d46cfull) <= 0x0572620ae4c415c9ull)
        return 47;
    if ((uint64_t)(n * 0x21cfb2b78c13521dull) <= 0x04d4873ecade304dull)
        return 53;
    if ((uint64_t)(n * 0xcbeea4e1a08ad8f3ull) <= 0x0456c797dd49c341ull)
        return 59;
    if ((uint64_t)(n * 0x4fbcda3ac10c9715ull) <= 0x04325c53ef368eb0ull)
        return 61;
    if ((uint64_t)(n * 0xf0b7672a07a44c6bull) <= 0x03d226357e16ece5ull)
        return 67;
    if ((uint64_t)(n * 0x193d4bb7e327a977ull) <= 0x039b0ad12073615aull)
        return 71;
    if ((uint64_t)(n * 0x7e3f1f8fc7e3f1f9ull) <= 0x0381c0e070381c0eull)
        return 73;
    if ((uint64_t)(n * 0x9b8b577e613716afull) <= 0x033d91d2a2067b23ull)
        return 79;
    if ((uint64_t)(n * 0xa3784a062b2e43dbull) <= 0x03159721ed7e7534ull)
        return 83;
    if ((uint64_t)(n * 0xf47e8fd1fa3f47e9ull) <= 0x02e05c0b81702e05ull)
        return 89;
    if ((uint64_t)(n * 0xa3a0fd5c5f02a3a1ull) <= 0x02a3a0fd5c5f02a3ull)
        return 97;
    if (n < 101 * 101)
        return 1; // prime
    if ((uint64_t)(n * 0x3a4c0a237c32b16dull) <= 0x0288df0cac5b3f5dull)
        return 101;
    if ((uint64_t)(n * 0xdab7ec1dd3431b57ull) <= 0x027c45979c95204full)
        return 103;
    if ((uint64_t)(n * 0x77a04c8f8d28ac43ull) <= 0x02647c69456217ecull)
        return 107;
    if ((uint64_t)(n * 0xa6c0964fda6c0965ull) <= 0x02593f69b02593f6ull)
        return 109;
    if ((uint64_t)(n * 0x90fdbc090fdbc091ull) <= 0x0243f6f0243f6f02ull)
        return 113;
    if ((uint64_t)(n * 0x7efdfbf7efdfbf7full) <= 0x0204081020408102ull)
        return 127;
    if ((uint64_t)(n * 0x03e88cb3c9484e2bull) <= 0x01f44659e4a42715ull)
        return 131;
    if ((uint64_t)(n * 0xe21a291c077975b9ull) <= 0x01de5d6e3f8868a4ull)
        return 137;
    if ((uint64_t)(n * 0x3aef6ca970586723ull) <= 0x01d77b654b82c339ull)
        return 139;
    if ((uint64_t)(n * 0xdf5b0f768ce2cabdull) <= 0x01b7d6c3dda338b2ull)
        return 149;
    if ((uint64_t)(n * 0x6fe4dfc9bf937f27ull) <= 0x01b2036406c80d90ull)
        return 151;
    return 1; // no small factor < 157
}

// binary gcd
static uint64_t uint64_gcd(uint64_t u, uint64_t v)
{
    uint64_t t, k;

    if (u < v)
    {
        t = u;
        u = v;
        v = t;
    }
    if (v == 0)
        return u;

    // strip trailing zeroes
    k = uint64_tzcnt(u | v);
    v >>= k;

    // Stein algorithm, from odd number to odd number
    u >>= uint64_tzcnt(u);
    do
    {
        v >>= uint64_tzcnt(v);

        if (u > v)
        {
            t = u;
            u = v;
            v = t;
        }
        v -= u;
    } while (v);
    return u << k;
}

// binary modular inverse 1/x mod m, x and m coprime
// r = (1/x) mod m

static uint64_t uint64_mod_inv(uint64_t x, uint64_t m)
{
    if (x < 2)
        return x;
    if (m < 3)
        return 0;
    uint64_t a = x, b = m, u = 1, v = 0;
    while (a != 0)
    {
        unsigned za = uint64_tzcnt(a);
        a >>= za;
        while (za--)
        {
            u += (u & 1) ? m : 0;
            u >>= 1;
        }
        if (a < b)
        {
            uint64_t t = a;
            uint64_t s = u;
            a = b;
            u = v;
            b = t;
            v = s;
        }
        a -= b;
        u = (u >= v) ? u - v : u + m - v;
    }
    return b == 1 ? v : 0;
}

static bool uint64_is_perfect_square(uint64_t a)
{
    if (0xffedfdfefdecull & (1ull << (a % 48)))
        return false;
    if (0xfdfdfdedfdfcfdecull & (1ull << (a % 64)))
        return false;
    if (0x7bfdb7cfedbafd6cull & (1ull << (a % 63)))
        return false;
    if (0x7dcfeb79ee35ccull & (1ull << (a % 55)))
        return false;
    if (0x8ec196bf5a60dc4ull & (1ull << (a % 61)))
        return false;
    if (0x5d49de7c1846d44ull & (1ull << (a % 59)))
        return false;
    if (0xd228fccfc512cull & (1ull << (a % 53)))
        return false;
    if (0x7bcae4d8ac20ull & (1ull << (a % 47)))
        return false;
    if (0x4a77c5c11acull & (1ull << (a % 43)))
        return false;
    if (0x4c7d4af8c8ull & (1ull << (a % 41)))
        return false;
    if (0x9a1dee164ull & (1ull << (a % 37)))
        return false;
    if (0x6de2b848ull & (1ull << (a % 31)))
        return false;
    if (0xc2edd0cull & (1ull << (a % 29)))
        return false;
    if (0x7acca0ull & (1ull << (a % 23)))
        return false;
    if (0x4f50cull & (1ull << (a % 19)))
        return false;
    if (0x5ce8ull & (1ull << (a % 17)))
        return false;
    if (0x9e4ull & (1ull << (a % 13)))
        return false;

    // approximation of square root with floating point accuracy
    double d = (double)a;
    d = exp(log(d) / 2.0); // square root
    double dl = d * 0.999999;
    double dh = d * 1.000001;
    uint64_t c, m;
    // binary search (1 more bit of square root per iteration)
    uint64_t r = (uint64_t)d;
    uint64_t l = (uint64_t)dl;
    uint64_t h = (uint64_t)dh;
    while (l <= h)
    {
        m = (l + h) >> 1;
        c = m * m;
        if (c == a)
        {
            return true; // perfect square
        }
        if (c < a)
        {
            l = m + 1;
            r = m;
        }
        else
        {
            h = m - 1;
        }
    }
    c = r * r; // check perfect square
    return (c == a);
}

// conversion of a large number into a basis-10 string.
// returns the output string length.
static unsigned uint128_sprint(char *ptr, uint128_t x)
{
    char linef[256];
    char *pb = linef;
    char *pt = ptr;

    if (x == 0)
    {
        *pb++ = '0';
    }
    while (x)
    {
        *pb++ = '0' + (char)(x % 10);
        x /= 10;
    }
    while (pb > linef)
    {
        *(pt++) = *(--pb);
    }
    *pt = 0;
    return pt - ptr;
}

#include "quardrubic_primality.h"
#include "quardrubic_primality_alloc.h"
#include "quardrubic_primality_precompute.h"

static void mpz_mod_add(mpz_t r, mpz_t a, mpz_t b, mod_precompute_t *p)
{
    mpz_add(r, a, b);
    mpz_mod_slow_reduce(r, p);
}

// x % (2^b -1)
static uint64_t mpz_mod_mersenne(mpz_t x, uint64_t b)
{
    uint64_t mask = (1ull << b) - 1;
    unsigned s = mpz_size(x);
    const mp_limb_t *array = mpz_limbs_read(x);
    uint128_t t, r = 0;

    if ((b & (b - 1)) == 0)
    {
        // an exact power of 2
        for (unsigned i = 0; i < s; i += 1)
        {
            r += array[i];
        }
    }
    else
    {
        // add, shift first, and reduce after
        unsigned c = 0;
        for (unsigned i = 0; i < b; i += 1)
        {
            t = 0;
            for (unsigned j = i; j < s; j += b)
            {
                t += array[j];
            }
            t = (t & mask) + (t >> b);
            t = t << c;
            r += t;
            r = (r & mask) + (r >> b);
            c += 64;
            while (c >= b)
            {
                c -= b;
            }
        }
    }
    // reduce mod 2^b - 1
    while (r >> 64)
    {
        r = (r & mask) + (r >> b);
    }
    return (uint64_t)r;
}

// detect a perfect square after efficient sieving
static bool mpz_perfect_square(mpz_t n)
{
    // detects more than 99% of non-perfect cubes.
    mpz_t ignore;
    mpz_init(ignore);
    uint64_t a = mpz_mod_ui(ignore, n, 64ull * 63ull * 55ull * 61ull * 59ull * 53ull * 47ull * 43ull);
    uint64_t b = mpz_mod_ui(ignore, n, 41ull * 37ull * 31ull * 29ull * 23ull * 19ull * 17ull * 13ull);
    mpz_clear(ignore);

    if (0xffedfdfefdecull & (1ull << (a % 48)))
        return false;
    if (0xfdfdfdedfdfcfdecull & (1ull << (a % 64)))
        return false;
    if (0x7bfdb7cfedbafd6cull & (1ull << (a % 63)))
        return false;
    if (0x7dcfeb79ee35ccull & (1ull << (a % 55)))
        return false;
    if (0x8ec196bf5a60dc4ull & (1ull << (a % 61)))
        return false;
    if (0x5d49de7c1846d44ull & (1ull << (a % 59)))
        return false;
    if (0xd228fccfc512cull & (1ull << (a % 53)))
        return false;
    if (0x7bcae4d8ac20ull & (1ull << (a % 47)))
        return false;
    if (0x4a77c5c11acull & (1ull << (a % 43)))
        return false;
    if (0x4c7d4af8c8ull & (1ull << (b % 41)))
        return false;
    if (0x9a1dee164ull & (1ull << (b % 37)))
        return false;
    if (0x6de2b848ull & (1ull << (b % 31)))
        return false;
    if (0xc2edd0cull & (1ull << (b % 29)))
        return false;
    if (0x7acca0ull & (1ull << (b % 23)))
        return false;
    if (0x4f50cull & (1ull << (b % 19)))
        return false;
    if (0x5ce8ull & (1ull << (b % 17)))
        return false;
    if (0x9e4ull & (1ull << (b % 13)))
        return false;

    // compute the rounded-down integer square root, check if the solution is exact.
    mpz_t u;
    mpz_init(u);
    uint64_t e = mpz_root(u, n, 2); // e = squareroot(n), e non zero if computation is exact.
    mpz_clear(u);
    return e ? true : false;
}

typedef enum sieve_e
{
    COMPOSITE_FOR_SURE,
    PRIME_FOR_SURE,
    UNDECIDED
} sieve_t;

// sieve small primes
static sieve_t uint64_composite_sieve(uint64_t a)
{
    if (a <= 152)
    {
        // return COMPOSITE_FOR_SURE for small numbers which are composite for sure , without checking further.
        sieve_t stooopid_prime_table[] = {
            PRIME_FOR_SURE,     PRIME_FOR_SURE,     PRIME_FOR_SURE,     PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE,
            COMPOSITE_FOR_SURE, COMPOSITE_FOR_SURE, PRIME_FOR_SURE,     COMPOSITE_FOR_SURE, PRIME_FOR_SURE,
            COMPOSITE_FOR_SURE};
        return stooopid_prime_table[a];
    }

    // divisibility is based on Barrett modular reductions by constants, use modular multiplications
    if ((uint64_t)(a * 0xaaaaaaaaaaaaaaabull) <= 0x5555555555555555ull)
        return COMPOSITE_FOR_SURE; // divisible by 3
    if ((uint64_t)(a * 0xcccccccccccccccdull) <= 0x3333333333333333ull)
        return COMPOSITE_FOR_SURE; // divisible by 5
    if ((uint64_t)(a * 0x6db6db6db6db6db7ull) <= 0x2492492492492492ull)
        return COMPOSITE_FOR_SURE; // divisible by 7
    if ((uint64_t)(a * 0x2e8ba2e8ba2e8ba3ull) <= 0x1745d1745d1745d1ull)
        return COMPOSITE_FOR_SURE; // divisible by 11
    if ((uint64_t)(a * 0x4ec4ec4ec4ec4ec5ull) <= 0x13b13b13b13b13b1ull)
        return COMPOSITE_FOR_SURE; // divisible by 13
    if ((uint64_t)(a * 0xf0f0f0f0f0f0f0f1ull) <= 0x0f0f0f0f0f0f0f0full)
        return COMPOSITE_FOR_SURE; // divisible by 17
    if ((uint64_t)(a * 0x86bca1af286bca1bull) <= 0x0d79435e50d79435ull)
        return COMPOSITE_FOR_SURE; // divisible by 19
    if ((uint64_t)(a * 0xd37a6f4de9bd37a7ull) <= 0x0b21642c8590b216ull)
        return COMPOSITE_FOR_SURE; // divisible by 23
    if ((uint64_t)(a * 0x34f72c234f72c235ull) <= 0x08d3dcb08d3dcb08ull)
        return COMPOSITE_FOR_SURE; // divisible by 29
    if ((uint64_t)(a * 0xef7bdef7bdef7bdfull) <= 0x0842108421084210ull)
        return COMPOSITE_FOR_SURE; // divisible by 31
    if (a < 37 * 37)
        return PRIME_FOR_SURE;
    if ((uint64_t)(a * 0x14c1bacf914c1badull) <= 0x06eb3e45306eb3e4ull)
        return COMPOSITE_FOR_SURE; // divisible by 37
    if ((uint64_t)(a * 0x8f9c18f9c18f9c19ull) <= 0x063e7063e7063e70ull)
        return COMPOSITE_FOR_SURE; // divisible by 41
    if ((uint64_t)(a * 0x82fa0be82fa0be83ull) <= 0x05f417d05f417d05ull)
        return COMPOSITE_FOR_SURE; // divisible by 43
    if ((uint64_t)(a * 0x51b3bea3677d46cfull) <= 0x0572620ae4c415c9ull)
        return COMPOSITE_FOR_SURE; // divisible by 47
    if ((uint64_t)(a * 0x21cfb2b78c13521dull) <= 0x04d4873ecade304dull)
        return COMPOSITE_FOR_SURE; // divisible by 53
    if ((uint64_t)(a * 0xcbeea4e1a08ad8f3ull) <= 0x0456c797dd49c341ull)
        return COMPOSITE_FOR_SURE; // divisible by 59
    if ((uint64_t)(a * 0x4fbcda3ac10c9715ull) <= 0x04325c53ef368eb0ull)
        return COMPOSITE_FOR_SURE; // divisible by 61
    if ((uint64_t)(a * 0xf0b7672a07a44c6bull) <= 0x03d226357e16ece5ull)
        return COMPOSITE_FOR_SURE; // divisible by 67
    if ((uint64_t)(a * 0x193d4bb7e327a977ull) <= 0x039b0ad12073615aull)
        return COMPOSITE_FOR_SURE; // divisible by 71
    if ((uint64_t)(a * 0x7e3f1f8fc7e3f1f9ull) <= 0x0381c0e070381c0eull)
        return COMPOSITE_FOR_SURE; // divisible by 73
    if ((uint64_t)(a * 0x9b8b577e613716afull) <= 0x033d91d2a2067b23ull)
        return COMPOSITE_FOR_SURE; // divisible by 79
    if ((uint64_t)(a * 0xa3784a062b2e43dbull) <= 0x03159721ed7e7534ull)
        return COMPOSITE_FOR_SURE; // divisible by 83
    if ((uint64_t)(a * 0xf47e8fd1fa3f47e9ull) <= 0x02e05c0b81702e05ull)
        return COMPOSITE_FOR_SURE; // divisible by 89
    if ((uint64_t)(a * 0xa3a0fd5c5f02a3a1ull) <= 0x02a3a0fd5c5f02a3ull)
        return COMPOSITE_FOR_SURE; // divisible by 97
    if (a < 101 * 101)
        return PRIME_FOR_SURE;
    if ((uint64_t)(a * 0x3a4c0a237c32b16dull) <= 0x0288df0cac5b3f5dull)
        return COMPOSITE_FOR_SURE; // divisible by 101
    if ((uint64_t)(a * 0xdab7ec1dd3431b57ull) <= 0x027c45979c95204full)
        return COMPOSITE_FOR_SURE; // divisible by 103
    if ((uint64_t)(a * 0x77a04c8f8d28ac43ull) <= 0x02647c69456217ecull)
        return COMPOSITE_FOR_SURE; // divisible by 107
    if ((uint64_t)(a * 0xa6c0964fda6c0965ull) <= 0x02593f69b02593f6ull)
        return COMPOSITE_FOR_SURE; // divisible by 109
    if ((uint64_t)(a * 0x90fdbc090fdbc091ull) <= 0x0243f6f0243f6f02ull)
        return COMPOSITE_FOR_SURE; // divisible by 113
    if ((uint64_t)(a * 0x7efdfbf7efdfbf7full) <= 0x0204081020408102ull)
        return COMPOSITE_FOR_SURE; // divisible by 127
    if ((uint64_t)(a * 0x03e88cb3c9484e2bull) <= 0x01f44659e4a42715ull)
        return COMPOSITE_FOR_SURE; // divisible by 131
    if ((uint64_t)(a * 0xe21a291c077975b9ull) <= 0x01de5d6e3f8868a4ull)
        return COMPOSITE_FOR_SURE; // divisible by 137
    if ((uint64_t)(a * 0x3aef6ca970586723ull) <= 0x01d77b654b82c339ull)
        return COMPOSITE_FOR_SURE; // divisible by 139
    if ((uint64_t)(a * 0xdf5b0f768ce2cabdull) <= 0x01b7d6c3dda338b2ull)
        return COMPOSITE_FOR_SURE; // divisible by 149
    if ((uint64_t)(a * 0x6fe4dfc9bf937f27ull) <= 0x01b2036406c80d90ull)
        return COMPOSITE_FOR_SURE; // divisible by 151

    // no factor less than 157
    if (a < 157 * 157)
        return PRIME_FOR_SURE; // prime
    return UNDECIDED;
}

static sieve_t mpz_composite_sieve(mpz_t n)
{
    // detect small 64-bit numbers
    unsigned sz = mpz_sizeinbase(n, 2);
    if (sz <= 64)
    {
        uint64_t a = mpz_get_ui(n);
        sieve_t sv = uint64_composite_sieve(a);
        return sv;
    }
    else
    {
        // large number, do the modular reduction in 2 steps
        // step 1 :
        //   reduce by a multiple of small factors
        // step 2:
        //    divisibility is based on Barrett modular reductions by constants, use modular multiplications

        uint64_t a;
        // 2^60-1 is divisible by 3,5,7,11,13,31,41,61,151 ...
        a = mpz_mod_mersenne(n, 60);

        if ((uint64_t)(a * 0xaaaaaaaaaaaaaaabull) <= 0x5555555555555555ull)
            return COMPOSITE_FOR_SURE; // divisible by 3
        if ((uint64_t)(a * 0xcccccccccccccccdull) <= 0x3333333333333333ull)
            return COMPOSITE_FOR_SURE; // divisible by 5
        if ((uint64_t)(a * 0x6db6db6db6db6db7ull) <= 0x2492492492492492ull)
            return COMPOSITE_FOR_SURE; // divisible by 7
        if ((uint64_t)(a * 0x2e8ba2e8ba2e8ba3ull) <= 0x1745d1745d1745d1ull)
            return COMPOSITE_FOR_SURE; // divisible by 11
        if ((uint64_t)(a * 0x4ec4ec4ec4ec4ec5ull) <= 0x13b13b13b13b13b1ull)
            return COMPOSITE_FOR_SURE; // divisible by 13
        if ((uint64_t)(a * 0xef7bdef7bdef7bdfull) <= 0x0842108421084210ull)
            return COMPOSITE_FOR_SURE; // divisible by 31
        if ((uint64_t)(a * 0x8f9c18f9c18f9c19ull) <= 0x063e7063e7063e70ull)
            return COMPOSITE_FOR_SURE; // divisible by 41
        if ((uint64_t)(a * 0x4fbcda3ac10c9715ull) <= 0x04325c53ef368eb0ull)
            return COMPOSITE_FOR_SURE; // divisible by 61
        if ((uint64_t)(a * 0x6fe4dfc9bf937f27ull) <= 0x01b2036406c80d90ull)
            return COMPOSITE_FOR_SURE; // divisible by 151

        // 2^56-1 is divisible by 3, 5, 17, 29, 43, 113, 127, .....
        a = mpz_mod_mersenne(n, 56);
        if ((uint64_t)(a * 0xf0f0f0f0f0f0f0f1ull) <= 0x0f0f0f0f0f0f0f0full)
            return COMPOSITE_FOR_SURE; // divisible by 17
        if ((uint64_t)(a * 0x34f72c234f72c235ull) <= 0x08d3dcb08d3dcb08ull)
            return COMPOSITE_FOR_SURE; // divisible by 29
        if ((uint64_t)(a * 0x82fa0be82fa0be83ull) <= 0x05f417d05f417d05ull)
            return COMPOSITE_FOR_SURE; // divisible by 43
        if ((uint64_t)(a * 0x90fdbc090fdbc091ull) <= 0x0243f6f0243f6f02ull)
            return COMPOSITE_FOR_SURE; // divisible by 113
        if ((uint64_t)(a * 0x7efdfbf7efdfbf7full) <= 0x0204081020408102ull)
            return COMPOSITE_FOR_SURE; // divisible by 127

        // 2^36-1 is divisible by 3,5,7,19,37,73,109, ...
        a = mpz_mod_mersenne(n, 36);
        if ((uint64_t)(a * 0x86bca1af286bca1bull) <= 0x0d79435e50d79435ull)
            return COMPOSITE_FOR_SURE; // divisible by 19
        if ((uint64_t)(a * 0x14c1bacf914c1badull) <= 0x06eb3e45306eb3e4ull)
            return COMPOSITE_FOR_SURE; // divisible by 37
        if ((uint64_t)(a * 0x7e3f1f8fc7e3f1f9ull) <= 0x0381c0e070381c0eull)
            return COMPOSITE_FOR_SURE; // divisible by 73
        if ((uint64_t)(a * 0xa6c0964fda6c0965ull) <= 0x02593f69b02593f6ull)
            return COMPOSITE_FOR_SURE; // divisible by 109

        // 2^44-1 is divisible by 3, 5, 23, 89, .....
        a = mpz_mod_mersenne(n, 44);
        if ((uint64_t)(a * 0xd37a6f4de9bd37a7ull) <= 0x0b21642c8590b216ull)
            return COMPOSITE_FOR_SURE; // divisible by 23
        if ((uint64_t)(a * 0xf47e8fd1fa3f47e9ull) <= 0x02e05c0b81702e05ull)
            return COMPOSITE_FOR_SURE; // divisible by 89
    }

    // no trivial small factor detected
    return UNDECIDED; // might be prime
}

// Iterate a third order linear recurrence using "double and add" steps
// Computes (s,t,u)^e mod (n, x^3-ax-a)
// Makes output s,t,u < n
static void mpz_exponentiate(mpz_t s, mpz_t t, mpz_t u, mpz_t e, uint64_t a, mod_precompute_t *p, bool first)
{
    unsigned bit;
    unsigned new_size = (p->n + 256) * 2;
    mpz_t s2, t2, u2, st, tu, us, tmp, tmp_s2, tmp_t2, tmp_u2;
    mpz_init2(tmp, new_size);
    mpz_init2(tmp_s2, new_size);
    mpz_init2(tmp_t2, new_size);
    mpz_init2(tmp_u2, new_size);
    mpz_init2(s2, new_size);
    mpz_init2(t2, new_size);
    mpz_init2(u2, new_size);
    mpz_init2(us, new_size);
    mpz_init2(st, new_size);
    mpz_init2(tu, new_size);

    bit = mpz_sizeinbase(e, 2) - 1;

    mpz_mod_to_montg(s, p);
    mpz_mod_to_montg(t, p);
    mpz_mod_to_montg(u, p);

    while (bit--)
    {
        // Double
        mpz_mul(s2, s, s);
        mpz_mul(t2, t, t);
        mpz_mul(u2, u, u);
        mpz_sub(st, s, t);
        mpz_sub(tu, t, u);
        mpz_sub(us, u, s);
        mpz_mul(tmp_s2, st, st);
        mpz_mul(tmp_t2, tu, tu);
        mpz_mul(tmp_u2, us, us);
        mpz_sub(st, s2, tmp_s2);
        mpz_add(st, st, t2);
        mpz_sub(tu, t2, tmp_t2);
        mpz_add(tu, tu, u2);
        mpz_sub(us, u2, tmp_u2);
        mpz_add(us, us, s2);
        mpz_mul_ui(s2, s2, a);
        mpz_mul_ui(st, st, a);
        if (mpz_tstbit(e, bit))
        {
            // Add
            mpz_add(tmp_u2, us, t2);
            mpz_add(u, tmp_u2, s2);
            mpz_mul_ui(u, u, a);
            mpz_add(tmp_s2, s2, st);
            mpz_add(s, tmp_s2, tu);
            mpz_add(tmp_t2, u2, st);
            mpz_add(t, tmp_t2, u);
        }
        else
        {
            mpz_add(tmp_s2, s2, us);
            mpz_add(s, tmp_s2, t2);
            mpz_add(tmp_t2, s2, st);
            mpz_add(t, tmp_t2, tu);
            mpz_add(u, u2, st);
        }
        // numbers to reduce here are < 6 * a * O(input_numbers)^2
        mpz_mod_fast_reduce(s, tmp, p);
        mpz_mod_fast_reduce(t, tmp, p);
        mpz_mod_fast_reduce(u, tmp, p);
    }

    mpz_mod_from_montg(s, tmp, p);
    mpz_mod_from_montg(t, tmp, p);
    mpz_mod_from_montg(u, tmp, p);

    mpz_mod_slow_reduce(s, p);
    mpz_mod_slow_reduce(t, p);
    mpz_mod_slow_reduce(u, p);

    mpz_clears(s2, t2, u2, st, tu, us, tmp_s2, tmp_t2, tmp_u2, 0);
    mpz_clear(tmp);
}

static bool uint64_quardrubic_primality(uint64_t n, bool verbose = false)
{
    if (n >> 61)
    {
        // the quardrubic test might overflow for numbers > 61 bits along the additions
        // Better use another slower deterministic test for numbers <= 64 bits
        // More precise constraint is n < 2^64/6
        mpz_t v;
        mpz_init_set_ui(v, n);
        bool r = mpz_quardrubic_primality(v, verbose);
        mpz_clear(v);
        return r;
    }

    if ((n & 1) == 0)
    {
        return n == 2; // even
    }
    if (n == 1)
    {
        return false; // one is not prime
    }

    // sieve composites with small factors
    sieve_t sv = uint64_composite_sieve(n);
    switch (sv)
    {
    case COMPOSITE_FOR_SURE:
        if (verbose)
        {
            printf("Number has a small factor\n");
        }
        return false; // composite
    case PRIME_FOR_SURE:
        if (verbose)
        {
            printf("Small number has no small factor\n");
        }
        return true; // prime
    case UNDECIDED:
    default:
        break;
    }

    if (uint64_is_perfect_square(n))
    {
        if (verbose)
        {
            printf("Number is a perfect square\n");
        }
        return false; // composite
    }

    uint64_t a6, a, g;
    int64_t sa;

    a = 9;

    while (1)
    {
        a += 10;
        if (verbose)
        {
            printf("try a = %lu\n", a);
        }
        sa = 1 + 4 * a;
        int j = int64_kronecker(sa, n);
        if (j == 0)
        {
            if (verbose)
            {
                printf("Number has a small factor\n");
            }
            return false; // n composite
        }
        if (j == 1)
        {
            continue; // try another a
        }

        if (!uint64_quardrubic_primality(a, verbose))
        {
            continue; // try another a
        }

        a6 = a * a * a;
        a6 *= a6;
        g = uint64_gcd(a6 - 1, n);
        if (g != 1)
        {
            if (g == n)
            {
                continue; // a^6 - 1 is a multiple of n, try another a
            }
            if (verbose)
            {
                printf("Number has a small factor\n");
            }
            return false; // n composite
        }

        // ----------------------------------------------------
        // Check (1*x+0)^n mod (x^2-x-a) == -x+1  mod n
        //

        uint64_t s, t, bit;
        s = 1;
        t = 0;
        bit = uint64_log_2(n);
        while (bit--)
        {
            uint128_t s2 = square_mod(s, n);
            s2 *= a;
            uint128_t t2 = t;
            t2 *= t;
            uint128_t st2 = s + t;
            st2 *= s + t;
            uint128_t ss = st2 - t2;
            uint128_t tt = t2 + s2;
            if ((n >> bit) & 1)
            {
                s = uint128_long_mod(ss, n);
                ss = tt + s;
                tt = s;
                tt *= a;
            }
            s = uint128_long_mod(ss, n);
            t = uint128_long_mod(tt, n);
        }
        return (s + 1 == n && t == 1);
    }
}

bool mpz_quardrubic_primality(mpz_t n, bool verbose)
{
    if (verbose)
    {
        gmp_printf("Testing a %lu digits number\n", mpz_sizeinbase(n, 10));
    }
    if (mpz_cmp_ui(n, 1ull << 61) < 0)
    {
        // the quardrubic test will run in 64 bits calculations
        return uint64_quardrubic_primality(mpz_get_ui(n), verbose);
    }

    if (mpz_tstbit(n, 0) == 0)
    {
        if (verbose)
        {
            printf("Number is even\n");
        }
        return (mpz_cmp_ui(n, 2) == 0); // even
    }

    // detects small primes, small composites
    // detects smooth composites
    sieve_t sv = mpz_composite_sieve(n);
    switch (sv)
    {
    case COMPOSITE_FOR_SURE:
        if (verbose)
        {
            printf("Number has a small factor\n");
        }
        return false; // composite
    case PRIME_FOR_SURE:
        if (verbose)
        {
            printf("Small number has no small factor\n");
        }
        return true; // prime
    case UNDECIDED:
    default:
        break;
    }

    if (mpz_perfect_square(n))
    {
        if (verbose)
        {
            printf("Number is a perfect square\n");
        }
        return false; // composite
    }

    uint64_t a;
    bool res = false;
    mpz_t tmp;
    mpz_init(tmp);

    a = 1;

    while (1)
    {
        a += 2;
        if (verbose)
        {
            printf("try a = %lu\n", a);
        }
        mpz_set_ui(tmp, 1 + 4 * a);
        int j = mpz_kronecker(tmp, n);
        if (j == 0)
        {
            if (mpz_cmp(n, tmp) == 0)
            {
                continue; // try another a
            }
            if (verbose)
            {
                printf("Number has a small factor\n");
            }
            res = false; // n composite
            goto done;
        }
        if (j == 1)
        {
            continue; // try another a
        }

        if (!uint64_quardrubic_primality(a))
        {
            continue; // try another a
        }
        mpz_set_ui(tmp, a * a * a);
        mpz_mul(tmp, tmp, tmp);
        mpz_sub_ui(tmp, tmp, 1);
        mpz_gcd(tmp, tmp, n);
        if (mpz_cmp_ui(tmp, 1) != 0)
        {
            if (mpz_cmp(n, tmp) == 0)
            {
                continue; // try another a
            }
            if (verbose)
            {
                printf("Number has a small factor\n");
            }
            res = false; // n composite
            goto done;
        }

        // -----------------------------------------------
        // Check (1*x+0)^n mod (x^2-x-a) == -x+1  mod n
        //

        uint64_t bits_a = 64 - __builtin_clzll(a); // rounded-up log2
        struct mod_precompute_t *p = mpz_mod_precompute(n, bits_a, verbose);
        mpz_t bs, bt, s2, t2, st2;
        mpz_inits(bs, bt, s2, t2, st2, 0);

// square, multiply and divide operations
//
//	? V=Mod(s*x+t, x^2-x-a)
//      %1 = Mod(s*x + t, x^2 - x - a)
//      ? V^2
//      %2 = Mod((s^2 + 2*t*s)*x + (a*s^2 + t^2), x^2 - x - a)
//      ? V*x
//      %3 = Mod((s + t)*x + a*s, x^2 - x - a)
//      ? V/x
//      %4 = Mod(-1/-a*t*x + (s + 1/-a*t), x^2 - x - a)

        // NAF exponent (3*n xor n)
        mpz_t n1, n3, ia;
        mpz_inits(n1, n3, ia, 0);
        mpz_mul_ui(n3, n, 3);
        mpz_xor(n1, n3, n);

        // fast 1/a mod n when n and a are coprime, computed from 1/n mod a
        uint64_t in = mpz_mod_ui(tmp, n, a);
        mpz_mul_ui(tmp, n, a - uint64_mod_inv(in, a));
        mpz_add_ui(tmp, tmp, 1);
        mpz_divexact_ui(ia, tmp, a);

        uint64_t bit = mpz_sizeinbase(n1, 2) - 1;
        mpz_set_ui(bs, 1);
        mpz_set_ui(bt, 0);
        mpz_mod_to_montg(bs, p);
        // mpz_mod_to_montg(bt, p);

        // (TODO) unroll the 3 first bits

        while (bit-- > 1)
        {
            // square (bs * x + bt)^2
            mpz_mul(s2, bs, bs);
            mpz_mul(t2, bt, bt);
            mpz_add(tmp, bs, bt);
            mpz_mul(st2, tmp, tmp);
            mpz_sub(bs, st2, t2);
            mpz_mul_ui(tmp, s2, a);
            mpz_add(bt, tmp, t2);
            if (mpz_tstbit(n1, bit))
            {
                if (mpz_tstbit(n3, bit))
                {
                    // multiply (bs * x + bt) by x
                    mpz_add(tmp, bs, bt);
                    mpz_mul_ui(bt, bs, a);
                    mpz_set(bs, tmp);
                }
                else
                {
                    // divide (bs * x + bt) by x
                    mpz_mul(tmp, bt, ia);
                    mpz_sub(bt, bs, tmp);
                    mpz_mod_positive_reduce(bt, bs, p);
                    mpz_set(bs, tmp);
                }
            }
            mpz_mod_fast_reduce(bs, tmp, p);
            mpz_mod_fast_reduce(bt, tmp, p);
        }

        mpz_mod_from_montg(bs, tmp, p);
        mpz_mod_from_montg(bt, tmp, p);

        mpz_mod_slow_reduce(bs, p);
        mpz_mod_slow_reduce(bt, p);

        // check bs*x + bt == (n-1)*x + 1
        mpz_add_ui(tmp, bs, 1);
        res = (mpz_cmp(tmp, n) == 0 && mpz_cmp_ui(bt, 1) == 0);

        mpz_clears(n1, n3, ia, 0);
        mpz_clears(bs, bt, s2, t2, st2, 0);
        mpz_mod_uncompute(p);
        goto done;
    }

done:
    mpz_clear(tmp);
    return res;
}

// ------------------------------------------------------------------------------
// Simple foolguard unit tests
// ------------------------------------------------------------------------------

void quardrubic_primality_self_test(void)
{
    // ---------------------------------------------------------------------------------
    if (1)
    {
        uint64_t a, b, c;

        // ---------------------------------------------------------------------------------
        printf("Gcd ...\n");
        a = 12;
        b = 15;
        c = uint64_gcd(a, b);
        assert(c == 3);

        a = 120;
        b = 150;
        c = uint64_gcd(a, b);
        assert(c == 30);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Perfect square ...\n");
        mpz_t mb;
        bool mc;
        mpz_init_set_ui(mb, 1);
        mpz_mul_2exp(mb, mb, 11213);
        mpz_sub_ui(mb, mb, 1);
        mpz_mul(mb, mb, mb);
        mc = mpz_perfect_square(mb);
        assert(mc == true);
        mpz_sub_ui(mb, mb, 1);
        mc = mpz_perfect_square(mb);
        assert(mc == false);
        mpz_clear(mb);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Fast reduction mod 2^b-1 ...\n");
        mpz_t ma, tmp;
        mpz_init(tmp);
        mpz_init_set_ui(ma, 11);
        for (unsigned b = 1; b < 10; b++)
        {
            mpz_mul(ma, ma, ma);
        }
        for (unsigned b = 1; b < 64; b++)
        {
            uint64_t mm = (1ull << b) - 1;
            uint64_t r = mpz_mod_mersenne(ma, b);
            assert(r % mm == mpz_mod_ui(tmp, ma, mm));
        }
        mpz_clears(ma, tmp, 0);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Sieve ...\n");
        mpz_t ma;
        assert(uint64_composite_sieve(101) == PRIME_FOR_SURE);
        assert(uint64_composite_sieve(1661) == COMPOSITE_FOR_SURE);
        assert(uint64_composite_sieve(281474976710677ull) == UNDECIDED);

        // 2^127 - 1 (a prime)
        mpz_init_set_ui(ma, 1);
        mpz_mul_2exp(ma, ma, 127);
        mpz_sub_ui(ma, ma, 1);
        assert(mpz_composite_sieve(ma) == UNDECIDED);
        // 2^127 + 1
        mpz_add_ui(ma, ma, 2);
        assert(mpz_composite_sieve(ma) == COMPOSITE_FOR_SURE);
        mpz_clear(ma);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Slow modular reduction\n");
        mpz_t x, ma, mb, tmp;
        mpz_inits(x, ma, mb, tmp, 0);
        mpz_set_ui(ma, 0x1);
        mpz_mul_2exp(ma, ma, 127);
        mpz_add_ui(ma, ma, 0x1d);
        // verify 2*(modulus +1) == 2
        mod_precompute_t *p = mpz_mod_precompute(ma);
        mpz_add_ui(mb, ma, 0x1);
        mpz_mod_add(x, mb, mb, p);
        assert(mpz_cmp_ui(x, 2) == 0);
        mpz_mod_slow_reduce(x, p);
        assert(mpz_cmp_ui(x, 2) == 0);
        // clears temp structure
        mpz_mod_uncompute(p);

        // ---------------------------------------------------------------------------------
        printf("Fast modular reduction\n");
        // verify generic modulus
        mpz_set_ui(ma, 0x1cc);
        p = mpz_mod_precompute(ma);
        assert(p->special_case == false);
        mpz_set_ui(mb, 0x2000ull * 0x1cc);
        mpz_mod_fast_reduce(mb, tmp, p);
        mpz_mod_slow_reduce(mb, p);
        assert(mpz_get_ui(mb) == 0);
        // clears temp structure
        mpz_mod_uncompute(p);

        // verify proth modulus
        mpz_t mx;
        mpz_init(mx);
        mpz_set_ui(ma, 0xcdefabcdcdefabcdull);
        mpz_mul_2exp(ma, ma, 64);
        mpz_add_ui(ma, ma, 1);
        p = mpz_mod_precompute(ma);
        assert(p->special_case == true);
        assert(p->montg == true);
        assert(p->proth == true);
        assert(p->n2 == 64);
        assert(p->n == 128);
        mpz_set_ui(ma, 0xabcdef01abcdef01ull);
        mpz_mul(ma, ma, ma);
        mpz_set_ui(mb, 0x1234567812345678ull);
        mpz_mul(mb, mb, mb);
        mpz_mul(x, ma, mb);
        mpz_mod(x, x, p->m);
        mpz_mod_to_montg(ma, p);
        mpz_mod_to_montg(mb, p);
        mpz_mul(mx, ma, mb);
        mpz_mod_fast_reduce(mx, tmp, p);
        mpz_mod_from_montg(mx, tmp, p);
        assert(mpz_cmp(x, mx) == 0);
        // clears temp structure
        mpz_clear(mx);
        mpz_mod_uncompute(p);

        // verify (modulus + 0x17)^2 == 17*17
        mpz_set_ui(ma, 1);
        mpz_mul_2exp(ma, ma, 127);
        p = mpz_mod_precompute(ma);
        assert(p->special_case == false);
        mpz_add_ui(mb, ma, 17);
        mpz_mul(x, mb, mb);
        mpz_mod_fast_reduce(x, tmp, p);
        mpz_mod_slow_reduce(x, p);
        assert(mpz_get_ui(x) == 17 * 17);

        // verify (modulus + 0x19)^4 == 17*17*17*17
        mpz_add_ui(mb, ma, 19);
        mpz_mul(x, mb, mb);
        mpz_mul(x, x, x);
        mpz_mod_fast_reduce(x, tmp, p);
        mpz_mod_slow_reduce(x, p);
        assert(mpz_get_ui(x) == 19 * 19 * 19 * 19);

        // clears temp structure
        mpz_mod_uncompute(p);
        p = 0;
        mpz_clears(x, ma, mb, tmp, 0);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Quardrubic inner loop (uint64)\n");
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Small primes (uint64)\n");
        assert(uint64_quardrubic_primality(16777259ull) == true);
        assert(uint64_quardrubic_primality(281474976710677ull) == true);

        // ---------------------------------------------------------------------------------
        printf("Small composites (uint64)\n");
        assert(uint64_quardrubic_primality(16777265ull) == false);
        assert(uint64_quardrubic_primality(281474976710683ull) == false);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        // minimum values of q where 2^j+q is prime
        uint32_t smallq[] = {
            1,   1,   1,   3,   1,   5,   3,   3,   1,  9,   7,   5,   3,   17,  27,  3,   1,   29,  3,   21,  7,   17,
            15,  9,   43,  35,  15,  29,  3,   11,  3,  11,  15,  17,  25,  53,  31,  9,   7,   23,  15,  27,  15,  29,
            7,   59,  15,  5,   21,  69,  55,  21,  21, 5,   159, 3,   81,  9,   69,  131, 33,  15,  135, 29,  13,  131,
            9,   3,   33,  29,  25,  11,  15,  29,  37, 33,  15,  11,  7,   23,  13,  17,  9,   75,  3,   171, 27,  39,
            7,   29,  133, 59,  25,  105, 129, 9,   61, 105, 7,   255, 277, 81,  267, 81,  111, 39,  99,  39,  33,  147,
            27,  51,  25,  281, 43,  71,  33,  29,  25, 9,   451, 41,  277, 165, 67,  27,  7,   29,  51,  17,  169, 39,
            67,  27,  27,  33,  85,  155, 87,  155, 37, 5,   217, 5,   175, 27,  85,  51,  91,  69,  147, 45,  253, 95,
            27,  15,  45,  69,  97,  299, 7,   107, 19, 21,  117, 141, 85,  83,  87,  147, 49,  129, 105, 77,  7,   9,
            427, 75,  87,  309, 15,  165, 49,  215, 27, 159, 205, 303, 57,  35,  129, 5,   133, 65,  27,  35,  21,  107,
            15,  101, 235, 351, 67,  15,  7,   581, 33, 203, 375, 47,  33,  71,  57,  75,  7,   251, 423, 129, 163, 185,
            217, 81,  49,  189, 735, 119, 735, 483, 3,  249, 67,  105, 357, 431, 43,  81,  25,  249, 67,  29,  115, 261,
            69,  59,  133, 315, 337, 63,  81,  119, 25, 65,  421, 39,  79,  95,  297, 155, 73,  435, 223, 0};

        printf("Medium primes (mpz)\n");
        mpz_t ma, mb;
        mpz_inits(ma, mb, 0);

        for (int j = 0; smallq[j]; j++)
        {
            // primes (2^j + q) must be catched
            mpz_init_set_ui(ma, 1);
            mpz_mul_2exp(ma, ma, j);
            mpz_add_ui(ma, ma, smallq[j]);
            bool res = mpz_quardrubic_primality(ma);
            if (!res)
            {
                printf("%d %d\n", (int)j, (int)smallq[j]);
                assert(res == true);
            }
        }
        mpz_clears(ma, mb, 0);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        uint32_t smallp[] = {2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37,  41,  43,
                             47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 0};
        printf("Medium composites (mpz)\n");
        mpz_t ma, mb;
        mpz_inits(ma, mb, 0);

        for (int j = 0; smallp[j]; j++)
        {
            // Composites p * (2^127-1) must be catched
            mpz_init_set_ui(ma, 1);
            mpz_mul_2exp(ma, ma, 127);
            mpz_sub_ui(ma, ma, 1);
            mpz_mul_ui(ma, ma, smallp[j]);
            bool res = mpz_quardrubic_primality(ma);
            if (res)
            {
                printf("%d %d\n", (int)j, (int)smallp[j]);
                assert(res == false);
            }
        }
        mpz_clears(ma, mb, 0);
    }

    // ---------------------------------------------------------------------------------
    if (1)
    {
        printf("Large primes (mpz)\n");
        mpz_t ma, mb;
        mpz_inits(ma, mb, 0);

        // large proth prime 333*2^448+1
        mpz_set_ui(ma, 333);
        mpz_mul_2exp(ma, ma, 448);
        mpz_add_ui(ma, ma, 1);
        assert(mpz_quardrubic_primality(ma) == true);

        // Large Riesel prime 100000000000037*2^5982-1
        mpz_set_ui(ma, 1);
        mpz_mul_2exp(ma, ma, 5982);
        mpz_mul_ui(ma, ma, 100000000000037ul);
        mpz_sub_ui(ma, ma, 1ul);
        assert(mpz_quardrubic_primality(ma) == true);

        // 11111...6442446...11111 (1001-digits) The smallest zeroless titanic palindromic prime
        // https://t5k.org/curios/page.php?number_id=3797
        char titanic[1002];
        memset(titanic, '1', 1001);
        titanic[497] = '6';
        titanic[498] = '4';
        titanic[499] = '4';
        titanic[500] = '2';
        titanic[501] = '4';
        titanic[502] = '4';
        titanic[503] = '6';
        titanic[1001] = 0;
        mpz_set_str(ma, titanic, 10);
        // gmp_printf("%Zd\n", ma);
        assert(mpz_quardrubic_primality(ma) == true);

        // 11111...0...3781 (1001-digits) The smallest Cyclops titanic prime.
        // https://t5k.org/curios/page.php?number_id=18967
        memset(titanic, '1', 1001);
        titanic[500] = '0';
        titanic[997] = '3';
        titanic[998] = '7';
        titanic[999] = '8';
        titanic[1000] = '1';
        titanic[1001] = 0;
        mpz_set_str(mb, titanic, 10);
        assert(mpz_quardrubic_primality(mb) == true);

        // ---------------------------------------------------------------------------------
        printf("Large composites (mpz)\n");
        // a semiprime out of the 2 previous tests, no small factors.
        mpz_mul(ma, mb, ma);
        assert(mpz_quardrubic_primality(ma) == false);

        // a large square
        mpz_mul(ma, mb, mb);
        assert(mpz_quardrubic_primality(ma) == false);

        // a large cube
        mpz_mul(ma, ma, mb);
        assert(mpz_quardrubic_primality(ma) == false);

        mpz_clears(ma, mb, 0);
    }
}
