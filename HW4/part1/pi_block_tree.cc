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

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long local_tosses = tosses / world_size;
    long long number_in_circle = 0;
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

    // TODO: binary tree reduction
    long long total_number_in_circle = number_in_circle;
    int step = 1;
    while (step < world_size) {
        if (world_rank % (2 * step) == 0) {
            if (world_rank + step < world_size) {
                long long recv_number_in_circle;
                MPI_Recv(&recv_number_in_circle, 1, MPI_LONG_LONG, world_rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_number_in_circle += recv_number_in_circle;
            }
        } else {
            int target = world_rank - step;
            MPI_Send(&total_number_in_circle, 1, MPI_LONG_LONG, target, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * total_number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
