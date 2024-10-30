#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  // initialize score_new
  double *score_new = (double *)malloc(sizeof(double) * numNodes);
  bool converged = false;
  double global_diff, dangling_sum;
  while(!converged) {
    dangling_sum = 0.0;
    #pragma omp parallel for reduction(+:dangling_sum)
    for (int i = 0; i < numNodes; i++) {
      if (outgoing_size(g, i) == 0) {
        dangling_sum += solution[i];
      }
    }

    #pragma omp parallel for
    for(int i = 0; i < numNodes; i++) {
      const Vertex* in_begin = incoming_begin(g, i);
      const Vertex* in_end = incoming_end(g, i);
      score_new[i] = 0.0;
      // there is dependency between solution, so use score_new
      for(const Vertex* it = in_begin; it != in_end; it++) {
        score_new[i] += solution[*it] / outgoing_size(g, *it);
      }

      score_new[i] = damping * score_new[i]
        + (1.0 - damping) / numNodes
        + damping * dangling_sum / numNodes;
    }

    // check convergence
    global_diff = 0.0;
    #pragma omp parallel for reduction(+:global_diff)
    for(int i = 0; i < numNodes; i++) {
      global_diff += std::abs(score_new[i] - solution[i]);
      // assign score_new to solution for next iteration
      solution[i] = score_new[i];
    }
    converged = (global_diff < convergence);
  }
  free(score_new);
}
