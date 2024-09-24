#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    if (i+VECTOR_WIDTH <= N) maskAll = _pp_init_ones();
    else maskAll = _pp_init_ones(N - i);

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float val;
  __pp_vec_int exp;
  __pp_mask maskAll, maskExpZero, maskMax;
  __pp_vec_float result;
  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    if (i+VECTOR_WIDTH <= N) maskAll = _pp_init_ones();
    else maskAll = _pp_init_ones(N - i);

    // val = values[i]; exp = exponents[i];
    _pp_vload_float(val, values + i, maskAll);
    _pp_vload_int(exp, exponents + i, maskAll);

    __pp_vec_int zeros = _pp_vset_int(0);
    _pp_veq_int(maskExpZero, exp, zeros, maskAll);
    // result = 1.f; when exp == 0
    if (_pp_cntbits(maskExpZero) > 0) _pp_vset_float(result, 1.f, maskExpZero);

    __pp_mask maskExpNotZero = _pp_mask_not(maskExpZero);
    // float result = x;
    if (_pp_cntbits(maskExpNotZero) > 0) _pp_vmove_float(result, val, maskExpNotZero);
    __pp_vec_int ones = _pp_vset_int(1);
    __pp_vec_int count;
    // count = y - 1;
    if (_pp_cntbits(maskExpNotZero) > 0) {
      _pp_vsub_int(count, exp, ones, maskExpNotZero);
      _pp_vgt_int(maskExpNotZero, count, zeros, maskExpNotZero);
    }

    while (_pp_cntbits(maskExpNotZero) > 0) {
      // result *= x;
      _pp_vmult_float(result, result, val, maskExpNotZero);

      // count--;
      _pp_vsub_int(count, count, ones, maskExpNotZero);
      _pp_vgt_int(maskExpNotZero, count, zeros, maskExpNotZero);
    }

    // clamp result to 9.999999f
    __pp_vec_float max = _pp_vset_float(9.999999f);
    _pp_vgt_float(maskMax, result, max, maskAll);
    if (_pp_cntbits(maskMax) > 0) _pp_vmove_float(result, max, maskMax);

    // output[i] = result;
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  float sum = 0.0;
  int width;
  __pp_vec_float val, tmp;
  __pp_mask maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    width = VECTOR_WIDTH;
    _pp_vload_float(val, values + i, maskAll);

    while (width > 1) {
      width >>= 1; // width /= 2
      _pp_hadd_float(tmp, val);
      _pp_interleave_float(val, tmp);
    }
    sum += val.value[0];
  }

  return sum;
}