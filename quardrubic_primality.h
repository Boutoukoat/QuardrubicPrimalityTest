#pragma once

// -----------------------------------------------------------------------
// Quardrubic primality test
//
// mpz_quardrubic_primality():
//    true: composite for sure
//    false: might be prime
//
// quardrubic_primality_self_test()
//    simplified unit tests to detect a possible compiler/platform issue.
//    assert when fail (this should not happen).
// -----------------------------------------------------------------------

#include "gmp.h"
#include <stdbool.h>

bool mpz_quardrubic_primality(mpz_t v, bool verbose = false);
void quardrubic_primality_self_test(void);
