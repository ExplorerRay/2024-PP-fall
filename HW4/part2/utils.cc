#include <mpi.h>
#include <fstream>
#include <iostream>

#define BLK_SIZE 64

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from in
//
// in:        input stream of the matrix file
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    in >> *n_ptr >> *m_ptr >> *l_ptr;
    *a_mat_ptr = new int[*n_ptr * *m_ptr];
    *b_mat_ptr = new int[*m_ptr * *l_ptr];

    for (int i = 0; i < *n_ptr; i++)
        for (int j = 0; j < *m_ptr; j++)
            in >> (*a_mat_ptr)[i * *m_ptr + j];

    for (int i = 0; i < *m_ptr; i++)
        for (int j = 0; j < *l_ptr; j++)
            in >> (*b_mat_ptr)[i * *l_ptr + j];
}

// Just matrix multiplication (your should output the result in this function)
//
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rows_per_process = n / world_size;
    int remain_rows = n % world_size;

    int distributed_rows = rows_per_process + ((world_rank < remain_rows) ? 1 : 0);
    int *local_a = new int[distributed_rows * m];
    int *local_c = new int[distributed_rows * l]();
    int *c_mat = nullptr;
    int *sendcounts = new int[world_size];
    int *send_displs = new int[world_size];
    int *recvcounts = new int[world_size];
    int *recv_displs = new int[world_size];

    if (world_rank == 0) {
        c_mat = new int[n * l];
    }
    for (int i = 0; i < world_size; i++){
        sendcounts[i] = ((i < remain_rows) ? (rows_per_process + 1) : rows_per_process) * m;
        send_displs[i] = (i == 0) ? 0 : send_displs[i - 1] + sendcounts[i - 1];
    }

    // Scatter rows of A to all processes
    MPI_Scatterv(a_mat, sendcounts, send_displs, MPI_INT, local_a, sendcounts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast B to all processes
    MPI_Bcast(const_cast<int*>(b_mat), m * l, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation of matrix multiplication
    int local_rows = (world_rank < remain_rows) ? (rows_per_process + 1) : rows_per_process;
    for (int i = 0; i < local_rows; i += BLK_SIZE) {
        for (int j = 0; j < l; j += BLK_SIZE) {
            for (int k = 0; k < m; k += BLK_SIZE) {
                for (int ii = i; ii < std::min(i + BLK_SIZE, local_rows); ii++) {
                    for (int jj = j; jj < std::min(j + BLK_SIZE, l); jj++) {
                        for (int kk = k; kk < std::min(k + BLK_SIZE, m); kk++) {
                            local_c[ii * l + jj] += local_a[ii * m + kk] * b_mat[kk * l + jj];
                        }
                    }
                }
            }
        }
    }

    // Gather the results from all processes
    for (int i = 0; i < world_size; i++){
        recvcounts[i] = ((i < remain_rows) ? (rows_per_process + 1) : rows_per_process) * l;
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Gatherv(local_c, recvcounts[world_rank], MPI_INT, c_mat, recvcounts, recv_displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Output the result
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                std::cout << c_mat[i * l + j] << " ";
            }
            std::cout << '\n';
        }
        delete[] c_mat;
    }

    delete[] local_a;
    delete[] local_c;
    delete[] sendcounts;
    delete[] send_displs;
    delete[] recvcounts;
    delete[] recv_displs;
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat) {
    delete[] a_mat;
    delete[] b_mat;
}
