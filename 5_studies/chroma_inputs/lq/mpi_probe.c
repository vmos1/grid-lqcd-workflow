/* mpi_probe.c -- the MPI gate of chroma_baseg_mds.sh on lq2. The launcher compiles it inside the
 * job with the Chroma environment (Open MPI 3.1.5 of NVHPC 23.11, the MPI the hmc binary loads)
 * and starts it with hmc's own srun line and wrapper, before the trajectory. It answers three
 * questions in seconds:
 *   1. Can srun --mpi=pmix start this Open MPI (its PMIx 2.2.3) across all nodes?
 *   2. Is every rank where QDP-JIT will put it? One line per rank: host, the local-rank variables
 *      QDP-JIT reads in order (PMI_RANK, OMPI_COMM_WORLD_LOCAL_RANK, SLURM_LOCALID; its device is
 *      that value mod the visible GPU count), the node communicator size, the cores it may use.
 *   3. Which transport carries inter-node messages? Rank 0 ping-pongs 8 B and 4 MiB (about one
 *      48^3 halo face) with the next rank on its node and with the first rank on another node.
 *      openib on lq2's mlx5 HCAs gives several GB/s; the TCP fallback far less.
 * Rank 0 ends with one PROBE_SUMMARY line, which the launcher parses. */
#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HOSTLEN 64

static const char *env_or(const char *k)
{
    const char *v = getenv(k);
    return v ? v : "unset";
}

/* One-way time per message of n bytes between ranks a and b, in seconds (rank a's view).
 * Every rank calls it: the barriers are collective. */
static double pingpong(int me, int a, int b, char *buf, int n, int iters)
{
    double t = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (me == a || me == b) {
        int other = (me == a) ? b : a;
        for (int i = -2; i < iters; i++) { /* two warm-up rounds */
            double t0 = MPI_Wtime();
            if (me == a) {
                MPI_Send(buf, n, MPI_CHAR, other, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_CHAR, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf, n, MPI_CHAR, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, n, MPI_CHAR, other, 0, MPI_COMM_WORLD);
            }
            if (i >= 0) t += MPI_Wtime() - t0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return t / iters / 2.0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPI_Comm shm;
    int shm_size;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    MPI_Comm_size(shm, &shm_size);

    char host[HOSTLEN];
    memset(host, 0, sizeof host);
    gethostname(host, HOSTLEN - 1);
    char *hosts = malloc((size_t)np * HOSTLEN);
    MPI_Allgather(host, HOSTLEN, MPI_CHAR, hosts, HOSTLEN, MPI_CHAR, MPI_COMM_WORLD);

    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof set, &set);
    int first = -1, last = -1;
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &set)) {
            if (first < 0) first = c;
            last = c;
        }

    printf("PROBE rank %d/%d host %s PMI_RANK %s OMPI_COMM_WORLD_LOCAL_RANK %s SLURM_LOCALID %s "
           "CUDA_VISIBLE_DEVICES %s node_comm %d cpus %d-%d (%d)\n",
           me, np, host, env_or("PMI_RANK"), env_or("OMPI_COMM_WORLD_LOCAL_RANK"),
           env_or("SLURM_LOCALID"), env_or("CUDA_VISIBLE_DEVICES"), shm_size, first, last,
           CPU_COUNT(&set));
    fflush(stdout);

    /* Rank 0's partners: the next rank on its host, the first rank on another host. Every rank
     * computes the same two numbers from the gathered host names. */
    int intra = -1, inter = -1;
    for (int r = 1; r < np; r++) {
        int same = strcmp(hosts + (size_t)r * HOSTLEN, hosts) == 0;
        if (same && intra < 0) intra = r;
        if (!same && inter < 0) inter = r;
    }
    const int big = 4 << 20, small = 8;
    char *buf = calloc((size_t)big, 1);
    double ti_s = intra > 0 ? pingpong(me, 0, intra, buf, small, 200) : 0.0;
    double ti_b = intra > 0 ? pingpong(me, 0, intra, buf, big, 20) : 0.0;
    double te_s = inter > 0 ? pingpong(me, 0, inter, buf, small, 200) : 0.0;
    double te_b = inter > 0 ? pingpong(me, 0, inter, buf, big, 20) : 0.0;

    int smin, smax;
    MPI_Reduce(&shm_size, &smin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&shm_size, &smax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (me == 0) {
        printf("PROBE_SUMMARY ranks %d node_comm_min %d node_comm_max %d", np, smin, smax);
        if (intra > 0)
            printf(" intra_pair 0-%d intra_lat_us %.2f intra_GBs %.2f", intra, ti_s * 1e6,
                   big / ti_b / 1e9);
        else
            printf(" intra_pair none intra_lat_us na intra_GBs na");
        if (inter > 0)
            printf(" inter_pair 0-%d inter_lat_us %.2f inter_GBs %.2f", inter, te_s * 1e6,
                   big / te_b / 1e9);
        else
            printf(" inter_pair none inter_lat_us na inter_GBs na");
        printf("\n");
        fflush(stdout);
    }

    free(buf);
    free(hosts);
    MPI_Comm_free(&shm);
    MPI_Finalize();
    return 0;
}
