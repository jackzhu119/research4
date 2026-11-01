/* Ordinary bounded native CPU/signal probe only; no quantum code or integrals. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <omp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static _Atomic int stop_flag = 0;
static volatile sig_atomic_t soft_signals = 0;

static void soft_handler(int sig) {
    (void)sig;
    ++soft_signals;
    const char message[] = "OBSERVED_SIGXCPU\n";
    (void)write(STDOUT_FILENO, message, sizeof(message)-1);
}

static double read_clock(clockid_t which) {
    struct timespec value;
    if (clock_gettime(which, &value) != 0) return -1.0;
    return (double)value.tv_sec + value.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    const int threads = argc == 2 ? atoi(argv[1]) : 0;
    if (threads != 1 && threads != 4) return 3;
    setvbuf(stdout, NULL, _IONBF, 0);
    struct rlimit limits;
    if (getrlimit(RLIMIT_CPU, &limits)) return 4;
    struct sigaction action = {0};
    action.sa_handler = soft_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGXCPU, &action, NULL)) return 5;
    printf("START threads=%d actual_soft=%llu actual_hard=%llu process=%.9f prof=%.9f sched_encoded=%.9f\n",
           threads, (unsigned long long)limits.rlim_cur, (unsigned long long)limits.rlim_max,
           read_clock(CLOCK_PROCESS_CPUTIME_ID), read_clock((clockid_t)-8), read_clock((clockid_t)-6));
    /* Independently stop at2.25 process CPU seconds even if RLIMIT is delayed. */
    #pragma omp parallel num_threads(threads)
    {
        volatile double accumulator = 1.001;
        while (!atomic_load(&stop_flag)) {
            for (int i=0; i<100000; ++i) accumulator = accumulator * 1.00000001 + 0.00000001;
            if (read_clock(CLOCK_PROCESS_CPUTIME_ID) >= 2.25) atomic_store(&stop_flag, 1);
        }
    }
    printf("SELF_STOP process=%.9f prof=%.9f sched_encoded=%.9f observed_soft_signals=%d\n",
           read_clock(CLOCK_PROCESS_CPUTIME_ID), read_clock((clockid_t)-8),
           read_clock((clockid_t)-6), (int)soft_signals);
    return 0;
}
