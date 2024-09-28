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



## Q2-2
> What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers?



## Q2-3
> Provide a theory for why the compiler is generating dramatically different assemblies.


