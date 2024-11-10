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

    // init MPI
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

    long long recv_number_in_circle;

    if (world_rank > 0)
    {
        // TODO: handle workers
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        recv_number_in_circle = number_in_circle;
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&number_in_circle, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_number_in_circle += number_in_circle;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = (4 * recv_number_in_circle) / ((double) tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
