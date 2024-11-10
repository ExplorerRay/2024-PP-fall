#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#define UINT32_MAX  (0xffffffff)

u_int32_t xorshift32(u_int32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long local_tosses = tosses / world_size;
    long long number_in_circle = 0;
    long long total_number_in_circle = 0;
    time_t timer;
    long seed = time(&timer) + world_rank;

    for (long long toss = 0; toss < local_tosses; toss++)
    {
        double x = seed / static_cast<double>(UINT32_MAX) * 2 - 1;
        seed = xorshift32(seed);
        double y = seed / static_cast<double>(UINT32_MAX) * 2 - 1;
        seed = xorshift32(seed);
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) {
            number_in_circle++;
        }
    }

    // Create MPI window
    MPI_Win_create(&total_number_in_circle, sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // For both master and workers
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    MPI_Accumulate(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
    MPI_Win_unlock(0, win);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4 * total_number_in_circle / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
