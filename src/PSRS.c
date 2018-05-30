
//*********************************************************************************
//
// Filename : 'PSRS.c'
//
// Function : Parallel sorting by regular sampling (using quick sort for local sorting)
//
// Author : Xingzhong Li
//
// Date : 2018/05
//
//*********************************************************************************

# define NDEBUG
# include <stdio.h>
# include <stdlib.h>
# include <assert.h>
# include <time.h>
# include <mpi.h>

//================================================================================
// 
// Function name : cmp
//
// Function : input parameter for qsort (sort integer)
//
//================================================================================
int cmp (const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main (int argc, char *argv[])
{
    if (argc != 2)
    {
        printf ("\n Error : the number of input parameters is wrong ! \n");
        exit (EXIT_FAILURE);
    }

    long n = atol (argv[1]);                            // size of array
    long *a_all = NULL;                                 // array a

    long i, j, k;
    int size, rank;
    double begin, end, t;

    // initialize MPI environment :
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    // phase 1 : Initialization
    if (rank == 0)
    {
        a_all = (long *)calloc (n, sizeof (long));
        assert (a_all != NULL);

        srand (time (NULL));
        for (i = 0; i < n; i++)
            a_all[i] = rand ();

//        a_all[0] = 15; a_all[1] = 46; a_all[2] = 48; a_all[3] = 93; a_all[4] = 39; a_all[5] = 6; a_all[6] = 72; a_all[7] = 91; a_all[8] = 14;
//        a_all[9] = 36; a_all[10] = 69; a_all[11] = 40; a_all[12] = 89; a_all[13] = 61; a_all[14] = 97; a_all[15] = 12; a_all[16] = 21; a_all[17] = 54;
//        a_all[18] = 53; a_all[19] = 97; a_all[20] = 84; a_all[21] = 58; a_all[22] = 32; a_all[23] = 27; a_all[24] = 33; a_all[25] = 72; a_all[26] = 20;
    }

    // begin clock :
    begin = MPI_Wtime ();

    // phase 2 : Scatter data, local sort and regular samples collecte
    long n_per = n / size;
    long *a = (long *)calloc (n_per, sizeof (long));
    assert (a != NULL);

    MPI_Scatter (a_all, n_per, MPI_LONG, a, n_per, MPI_LONG, 0, MPI_COMM_WORLD);
    qsort (a, n_per, sizeof (long), cmp);

    long *samples = (long *)calloc (size, sizeof (long));
    assert (samples != NULL);
    for (i = 0; i < size; i++)
        samples[i] = a[i * size];

    // phase 3 : Gather and merge samples, choose and broadcast (size - 1) pivots
    long *samples_all;
    if (rank == 0)
    {
        samples_all = (long *)calloc ((size * size), sizeof (long));
        assert (samples_all != NULL);
    }
    MPI_Gather (samples, size, MPI_LONG, samples_all, size, MPI_LONG, 0, MPI_COMM_WORLD);

    long *pivots = (long *)calloc ((size - 1), sizeof (long));
    assert (pivots != NULL);
    if (rank == 0)
    {
        qsort (samples_all, (size * size), sizeof (long), cmp);
        for (i = 0; i < (size - 1); i++)
            pivots[i] = samples_all[(i + 1) * size];
    }
    MPI_Bcast (pivots, (size - 1), MPI_LONG, 0, MPI_COMM_WORLD);

    // phase 4 : Local data is partitioned
    int index = 0;
    int *partition_size = (int *)calloc (size, sizeof (int));
    assert (partition_size != NULL);
    for (i = 0; i < n_per; i++)
    {
        if (a[i] > pivots[index]) 
        {
            index += 1;
        }
        if (index == (size - 1))
        {
            partition_size[index] = n_per - i;
            break;
        }
        partition_size[index]++;
    }

    // phase 5 : All ith classes are gathered and merged
    int *new_partition_size = (int *)calloc (size, sizeof (int));
    assert (new_partition_size != NULL);
    MPI_Alltoall (partition_size, 1, MPI_INT, new_partition_size, 1, MPI_INT, MPI_COMM_WORLD);

    int totalsize = 0;
    for (i = 0; i < size; i++)
        totalsize += new_partition_size[i];
    long *new_partitions = (long *)calloc (totalsize, sizeof (long));
    assert (new_partitions != NULL);

    int *send_dis = (int *)calloc (size, sizeof (int));
    assert (send_dis != NULL);
    int *recv_dis = (int *)calloc (size, sizeof (int));
    assert (recv_dis != NULL);
    send_dis[0] = 0;
    recv_dis[0] = 0;
    for (i = 1; i < size; i++)
    {
        send_dis[i] = send_dis[i - 1] + partition_size[i - 1];
        recv_dis[i] = recv_dis[i - 1] + new_partition_size[i - 1];
    }
    MPI_Alltoallv (a, partition_size, send_dis, MPI_LONG, new_partitions, new_partition_size, recv_dis, MPI_LONG, MPI_COMM_WORLD);
    qsort (new_partitions, totalsize, sizeof (long), cmp);

    // phase 6 : Root processor collects all the data
    int *recv_count;
    if (rank == 0)
    {
        recv_count = (int *)calloc (size, sizeof (int));
        assert (recv_count != NULL);
    }
    MPI_Gather (&totalsize, 1, MPI_INT, recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        recv_dis[0] = 0;
        for (i = 1; i < size; i++)
            recv_dis[i] = recv_dis[i - 1] + recv_count[i - 1];
    }
    MPI_Gatherv (new_partitions, totalsize, MPI_LONG, a_all, recv_count, recv_dis, MPI_LONG, 0, MPI_COMM_WORLD);

    // end clock :
    end = MPI_Wtime ();
    t = (end - begin);
    MPI_Barrier (MPI_COMM_WORLD);

    // find the maximum running time :
    double *t_all;
    if (rank == 0)
    {
        t_all = (double *)calloc (size, sizeof (double));
        assert (t_all != NULL);
    }
    MPI_Gather (&t, 1, MPI_DOUBLE, t_all, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        double t_max = t_all[0];
        for (i = 1; i < size; i++)
        {
            if (t_max < t_all[i]) 
                t_max = t_all[i];
        }
        printf ("\n t_max = %lf \n", t_max);
    }

    // output sorting result :
//    if (rank == 0)
//    {
//        for (i = 0; i < n; i++)
//            printf (" i = %2ld \t %ld \n", i, a_all[i]);
//    }

    // free memory :
    if (rank == 0)
    {
        free (a_all);
        free (samples_all);
        free (t_all);
    }
    free (a);
    free (samples);
    free (pivots);
    free (partition_size);
    free (new_partition_size);
    free (new_partitions);
    free (send_dis);
    free (recv_dis);
    free (recv_count);

    return 0;
}
