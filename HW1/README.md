# Questions
## Q1-1
> Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

It decreases as `VECTOR_WIDTH` becomes larger.

### Reason
Utilization is defined as the number of '1's in the mask when doing operations. 

When doing exponentiation, we keeps subtracting 1 and multiplying, then the total number of operations is determined by the MAX exponent in the vector with length `VECTOR_WIDTH`.

If `VECTOR_WIDTH` is larger, for example 16, if one of the exponent is 10, then others (15) are all 0. 
But others (15 in total) still need to do the same number of operations as the one with 10. So the utilization is lower.

If `VECTOR_WIDTH` is smaller, for example 2, if one of the exponent is 10, then others (1) are all 0. 
Even if others (1 in total) still need to do the same number of operations as the one with 10. It wastes not that much. So the utilization is higher.

## Q2-1

## Q2-2

## Q2-3
