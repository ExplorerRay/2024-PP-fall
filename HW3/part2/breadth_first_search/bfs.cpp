#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define CHUNK_SIZE 16384

// #define VERBOSE

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

void old_top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                distances[outgoing] = distances[node] + 1;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}

int top_down_step(
    Graph g,
    const int prev_distance,
    int *distances)
{
    int end_count = 0;
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) reduction(+:end_count)
    for(int i = 0; i < g->num_nodes; i++){
        if(distances[i] == prev_distance){
            int start_edge = g->outgoing_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[i + 1];

            for(int neighbor = start_edge; neighbor < end_edge; neighbor++){
                int outgoing = g->outgoing_edges[neighbor];
                if(distances[outgoing] == NOT_VISITED_MARKER){
                    distances[outgoing] = prev_distance + 1;
                    end_count++;
                }
            }
        }
    }
    return end_count;
}

void bfs_top_down(Graph graph, solution *sol)
{
    if (graph->num_edges <= 4 * graph->num_nodes)
    {
        vertex_set list1;
        vertex_set list2;
        vertex_set_init(&list1, graph->num_nodes);
        vertex_set_init(&list2, graph->num_nodes);

        vertex_set *frontier = &list1;
        vertex_set *new_frontier = &list2;

        // initialize all nodes to NOT_VISITED
        for (int i = 0; i < graph->num_nodes; i++)
            sol->distances[i] = NOT_VISITED_MARKER;

        // setup frontier with the root node
        frontier->vertices[frontier->count++] = ROOT_NODE_ID;
        sol->distances[ROOT_NODE_ID] = 0;

        while (frontier->count != 0){
            vertex_set_clear(new_frontier);
            old_top_down_step(graph, frontier, new_frontier, sol->distances);

            // swap pointers
            vertex_set *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
    }
    else{
        int end_count = 1;
        int prev_dist = 0;

        // initialize all nodes to NOT_VISITED
        for (int i = 0; i < graph->num_nodes; i++)
            sol->distances[i] = NOT_VISITED_MARKER;

        sol->distances[ROOT_NODE_ID] = 0;

        while (end_count > 0)
        {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

            end_count = top_down_step(graph, prev_dist, sol->distances);
            prev_dist++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        }
    }

}

int bottom_up_step(
    Graph g,
    const int prev_distance,
    int *distances)
{
    const int now_distance = prev_distance + 1;
    int end_cnt = 0;

    // iterate all unvisited nodes
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) reduction(+:end_cnt)
    for (int i = 0; i < g->num_nodes; i++)
    {
        int node = i;
        if (distances[node] != NOT_VISITED_MARKER)
            continue;

        int start_edge = g->incoming_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[node + 1];

        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int incoming = g->incoming_edges[neighbor];
            // check if the incoming node is in the frontier
            if (distances[incoming] == prev_distance)
            {
                distances[node] = now_distance;
                end_cnt++;
                break;
            }
        }
    }
    return end_cnt;
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    int prev_dist = 0;

    // initialize all nodes to NOT_VISITED except the root node
    sol->distances[ROOT_NODE_ID] = prev_dist;
    for (int i = 1; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    int end_count = 1;
    while (end_count > 0) {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        end_count = bottom_up_step(graph, prev_dist, sol->distances);
        prev_dist++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", new_frontier->count, end_time - start_time);
#endif
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    int alpha = 14;
    int beta = 24;
    int end_count = 1;
    int prev_dist = 0;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    while (end_count > 0)
    {
        if (end_count*alpha > (graph->num_nodes - end_count))
        {
            end_count = bottom_up_step(graph, prev_dist, sol->distances);
        }
        else
        {
            end_count = top_down_step(graph, prev_dist, sol->distances);
        }
        prev_dist++;

    }
}
