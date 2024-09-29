# Parallel Programming HW1 Questions
## Q1-1
> Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

| `VECTOR_WIDTH` | vector utilization |
| -------- | -------- | 
| 2    | 84.0 %  | 
| 4    | 76.1 %   |
| 8    | 71.6 %  |
| 16   | 69.5 %   |

It decreases as `VECTOR_WIDTH` becomes larger.

### Reason
Utilization is defined as the number of '1's in the mask when doing operations. 

When doing exponentiation,
we keeps subtracting 1 and multiplying, 
then the total number of operations 
is determined by the MAX exponent 
in the vector with length `VECTOR_WIDTH`.

If `VECTOR_WIDTH` is larger, 
for example 16, if one of the exponent is 10, 
then others (15) are all 0. 
But others (15 in total) still need to do the same number of operations as the one with 10. 
So the utilization is lower.

If `VECTOR_WIDTH` is smaller, 
for example 2, if one of the exponent is 10, 
then others (1) are all 0. 
Even if others (1 in total) still need to do the same number of operations as the one with 10. 
It wastes not that much. 
So the utilization is higher.

## Q2-1
> Fix the code to make sure it uses aligned moves for the best performance.

Orginal aligned size is 16 bytes,
but AVX2 register size is 256 bits (32 bytes).

Therefore, the aligned size should be set to 32 bytes instead of 16.

```c
a = (float *)__builtin_assume_aligned(a, 32);
b = (float *)__builtin_assume_aligned(b, 32);
c = (float *)__builtin_assume_aligned(c, 32);
```

## Q2-2
> What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers?

| compile option | median elapsed time after running 50 times (s) | speedup |
| -------- | -------- | -------- |
| No    | 6.937711 | 1x |
| `VECTORIZE=1`    | 1.724773 | about 4x |
| `VECTORIZE=1 AVX2=1`   | 0.862713 | about 8x |

Original size of one float is 32 bits.

Because the speedup of `VECTORIZE=1` is about 4x, 
I think the default vector registers on the PP machines are 32*4 = 128 bits. 

And the speedup of `VECTORIZE=1 AVX2=1` is about 8x, 
so the AVX2 vector registers are 32*8 = 256 bits.

### Code for testing different compile options:
```python
import os
import subprocess
import statistics

os.system("make clean && make VECTORIZE=1 AVX2=1")
cnt = 0
run_sec = []
while cnt < 50:
    out = subprocess.run(['srun', './test_auto_vectorize', '-t', '1'], capture_output=True)
    # avoid segmentation fault error
    if out.returncode == 0:
        cnt += 1
        out = out.stdout.decode('utf-8')
        out = out.split('\n')[-2]
        out = out.split('sec')[0]
        run_sec.append(float(out))
print(statistics.median(run_sec))
```

## Q2-3
> Provide a theory for why the compiler is generating dramatically different assemblies.

I also tried GCC 12.4.0 to compile the code, 
and both of them generated almost the same assembly containing SIMD instructions.

Therefore, I think it is because the compiler (clang 11.1.0) is not smart enough.
It just converts the code to assembly step by step, 
not knowing both codes do the same as getting a max value.

By using if and else statement, the compiler knows that the code is getting a max value,
and it can generate the assembly with SIMD instructions such as `maxps`.

Because GCC 12.4.0 is able to know both codes doing the same things 
and 11.1.0 is not the latest version in clang,
I think this is probably be optimized in clang newer versions.
