#include <iostream>
#include <pthread.h>
#include <vector>
#include <immintrin.h>
#include <time.h>

using namespace std;

struct thread_data {
    long long tosses;
    int tid;
};

struct alignas(64) padded_long {
    long long value;
};

__m256i SIMD_xorshift(__m256i state) {
    // uint32_t xorshift32(uint32_t state) {
    //     state ^= state << 13;
    //     state ^= state >> 17;
    //     state ^= state << 5;
    //     return state;
    // }
    // Do the xorshift32 operation on each 32-bit integer in the state
    const __m256i shift13 = _mm256_slli_epi32(state, 13);
    state = _mm256_xor_si256(state, shift13);
    const __m256i shift17 = _mm256_srli_epi32(state, 17);
    state = _mm256_xor_si256(state, shift17);
    const __m256i shift5 = _mm256_slli_epi32(state, 5);
    state = _mm256_xor_si256(state, shift5);

    return state;
}

__m256 narrowRange(__m256i bits) {
    const __m256i mantissaMask = _mm256_set1_epi32(0x7FFFFF);
    const __m256i mantissa = _mm256_and_si256(bits, mantissaMask);

    // Convert to a floating-point number in the range [1.0, 2.0)
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 val = _mm256_or_ps(_mm256_castsi256_ps(mantissa), one);

    // Scale the number from [1.0, 2.0) to [0.0, 1.0)
    val = _mm256_sub_ps(val, one);

    // Scale and shift to map [0.0, 1.0) to [-1.0, 1.0)
    val = _mm256_fmsub_ps(val, _mm256_set1_ps(2.0f), _mm256_set1_ps(1.0f));
    
    return val;
}

vector<padded_long> *num_in_circle_part;

void *tossing(void *data) {
    struct thread_data *td = (struct thread_data *)data;
    int tid = td->tid;
    long long num_of_toss = td->tosses;
    long long number_in_circle = 0;

    time_t timer;
    long seed = time(&timer);
    __m256i state = _mm256_set_epi32(seed, seed + 1, seed + 2, seed + 3, seed + 4, seed + 5, seed + 6, seed + 7);

    __m256 one = _mm256_set1_ps(1.0);
    for (long long toss = 0; toss < num_of_toss; toss += 8) {
        state = SIMD_xorshift(state);
        __m256 x = narrowRange(state);
        state = SIMD_xorshift(state);
        __m256 y = narrowRange(state);

        // Compute squared distance
        __m256 x_squared = _mm256_mul_ps(x, x);
        __m256 y_squared = _mm256_mul_ps(y, y);
        __m256 distance_squared = _mm256_add_ps(x_squared, y_squared);

        // Compare distance_squared <= 1.0f
        __m256 mask = _mm256_cmp_ps(distance_squared, one, _CMP_LE_OS);

        // Count the number of points inside the circle
        int inside_mask = _mm256_movemask_ps(mask);
        number_in_circle += _mm_popcnt_u32(inside_mask);
    }

    (*num_in_circle_part)[tid].value = number_in_circle;
    return nullptr;
}

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(0);

    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <number of threads> <number of tosses>\n";
        return 1;
    }
    long long number_in_circle = 0;
    int number_of_threads = atoi(argv[1]);
    long long number_of_tosses = atoll(argv[2]);
    float pi_estimate;

    vector<pthread_t> threads(number_of_threads);
    num_in_circle_part = new vector<padded_long>(number_of_threads);
    vector<thread_data> td(number_of_threads);
    long long tosses_per_thread = number_of_tosses / number_of_threads;

    for (int i = 0; i < number_of_threads; i++) {
        td[i].tosses = tosses_per_thread;
        td[i].tid = i;
        pthread_create(&threads[i], nullptr, tossing, (void *)&td[i]);
    }

    for (int i = 0; i < number_of_threads; i++) {
        pthread_join(threads[i], nullptr);
        number_in_circle += (*num_in_circle_part)[i].value;
    }

    pi_estimate = 4.0 * number_in_circle / static_cast<float>(number_of_tosses);
    cout << pi_estimate << '\n';

    delete num_in_circle_part;
    return 0;
}
