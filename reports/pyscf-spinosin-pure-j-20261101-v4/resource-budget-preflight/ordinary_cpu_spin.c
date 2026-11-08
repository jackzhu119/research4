/* Non-quantum, independently bounded CPU-guard test child only. */
#define _POSIX_C_SOURCE 200809L
#include <omp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
static _Atomic int stop_flag=0;
static void term_handler(int ignored) {
    (void)ignored;
    const char text[]="ORDINARY_CHILD_OBSERVED_TERM\n";
    (void)write(1,text,sizeof(text)-1);
}
static double process_cpu(void) {
    struct timespec value;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&value);
    return value.tv_sec+value.tv_nsec/1e9;
}
int main(void) {
    setvbuf(stdout,NULL,_IONBF,0);
    struct sigaction action={0};action.sa_handler=term_handler;
    sigemptyset(&action.sa_mask);sigaction(SIGTERM,&action,NULL);
    printf("ORDINARY_NATIVE4_SELF_STOP_AT_3_CPU_SECONDS\n");
    #pragma omp parallel num_threads(4)
    {
        volatile double value=1.0;
        while (!atomic_load(&stop_flag)) {
            for (int i=0;i<100000;i++) value=value*1.00000001+0.00000001;
            if (process_cpu()>=3.0) atomic_store(&stop_flag,1);
        }
    }
    printf("ORDINARY_INDEPENDENT_SELF_STOP %.9f\n",process_cpu());
    return 0;
}
