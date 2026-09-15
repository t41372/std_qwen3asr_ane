// Read the target process's current and peak residency through public libproc.
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    errno = 0;
    long value = strtol(argv[1], &end, 10);
    if (errno || *end || value <= 0 || value > INT_MAX) return 2;
    struct rusage_info_v4 usage = {0};
    if (proc_pid_rusage((int)value, RUSAGE_INFO_V4, (rusage_info_t *)&usage)) {
        perror("proc_pid_rusage");
        return 1;
    }
    printf("{\"rss_bytes\":%" PRIu64 ",\"phys_footprint_bytes\":%" PRIu64
           ",\"peak_phys_footprint_bytes\":%" PRIu64 "}\n",
           usage.ri_resident_size, usage.ri_phys_footprint,
           usage.ri_lifetime_max_phys_footprint);
    return 0;
}
