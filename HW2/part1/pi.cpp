#include <iostream>
#include <random>
#include <ctime>
#include <pthread.h>
#include <cstdlib>
#include <unistd.h>

using namespace std;

struct thread_data{
    long long tosses;
    int tid;
};

long long *num_in_circle_part;

void *tossing(void *data){
    struct thread_data *td = (struct thread_data *)data;
    int tid = td->tid;
    long long num_of_toss = td->tosses;
    long long number_in_circle = 0;
    random_device rd;
    mt19937_64 generater(rd());
    uniform_real_distribution<double> unif(-1.0, 1.0);
    for(long long toss = 0; toss < num_of_toss; toss++){
        double x = unif(generater);
        double y = unif(generater);
        double distance_squared = x*x + y*y;
        if(distance_squared <= 1.0){
            number_in_circle++;
        }
    }
    
    num_in_circle_part[tid] = number_in_circle;
    return nullptr;
}

int main(int argc, char *argv[]){
    if (argc != 3){
        cout << "Usage: " << argv[0] << "<number of threads> <number of tosses>\n";
        return 1;
    }
    long long number_in_circle = 0;
    int number_of_threads = atoi(argv[1]);
    long long number_of_tosses = atoll(argv[2]);
    double pi_estimate;
    random_device rd;
    mt19937_64 generater(rd());
    uniform_real_distribution<double> unif(-1.0, 1.0);

    pthread_t threads[number_of_threads];
    num_in_circle_part = (long long *)malloc(sizeof(long long) * number_of_threads);
    struct thread_data *td = (struct thread_data *)malloc(sizeof(struct thread_data) * number_of_threads);
    long long tosses_per_thread = number_of_tosses / number_of_threads;
    for(int i = 0; i < number_of_threads; i++){
        td[i].tosses = tosses_per_thread;
        td[i].tid = i;
        pthread_create(&threads[i], NULL, &tossing, (void *)&td[i]);
    }
    for(int i = 0; i < number_of_threads; i++){
        pthread_join(threads[i], NULL);
        number_in_circle += num_in_circle_part[i];
    }
    pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);
    cout << pi_estimate << '\n';
    return 0;
}
