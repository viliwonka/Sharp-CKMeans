/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */

using System.Collections.Generic;
using System;
using System.Threading.Tasks;
using number = System.Decimal;

namespace Sharp.CKMeans
{
    public static class DynamicProgramming
    {
        public static void FillMatrix(number[] x, number[] w, number[][] S, int[][] J, Method method, DissimilarityType criterion)
        {
            /*
            x: One dimension vector to be clustered, must be sorted (in any order).
            S: K x N matrix. S[q][i] is the sum of squares of the distance from
            each x[i] to its cluster mean when there are exactly x[i] is the
            last point in cluster q
            J: K x N backtrack matrix

            NOTE: All vector indices in this program start at position 0
            */

            int K = S.Length;
            int N = S[0].Length;

            number[] sum_x = new number[N];
            number[] sum_x_sq = new number[N];

            number[] sum_w = null;
            number[] sum_w_sq = null;

            int[] jseq = new int[N];

            number shift = x[N / 2]; // median. used to shift the values of x to
                                     //  improve numerical stability

            if (w == null || w.Length == 0)
            {
                // equally weighted
                sum_x[0] = x[0] - shift;
                sum_x_sq[0] = (x[0] - shift) * (x[0] - shift);
            }
            else
            { // unequally weighted
                sum_x[0] = w[0] * (x[0] - shift);
                sum_x_sq[0] = w[0] * (x[0] - shift) * (x[0] - shift);

                sum_w = new number[N];
                sum_w_sq = new number[N];

                sum_w[0] = w[0];
                sum_w_sq[0] = w[0] * w[0];
            }

            S[0][0] = 0;
            J[0][0] = 0;

            for (int i = 1; i < N; ++i)
            {
                if (w == null || w.Length == 0)
                { // equally weighted
                    sum_x[i] = sum_x[i - 1] + x[i] - shift;
                    sum_x_sq[i] = sum_x_sq[i - 1] + (x[i] - shift) * (x[i] - shift);
                }
                else
                { // unequally weighted
                    sum_x[i] = sum_x[i - 1] + w[i] * (x[i] - shift);
                    sum_x_sq[i] = sum_x_sq[i - 1] + w[i] * (x[i] - shift) * (x[i] - shift);
                    sum_w[i] = sum_w[i - 1] + w[i];
                    sum_w_sq[i] = sum_w_sq[i - 1] + w[i] * w[i];
                }

                // Initialize for q = 0
                S[0][i] = WithinCluster.Dissimilarity(criterion, 0, i, sum_x, sum_x_sq, sum_w, sum_w_sq); // ssq(0, i, sum_x, sum_x_sq, sum_w);
                J[0][i] = 0;
            }
            /*
            #if DEBUG
                        for (int i = 0; i < x.Length; ++i)
                        {
                            Console.Write(x[i] + ",");
                        }
                        Console.WriteLine();
            #endif
            */
            for (int q = 1; q < K; ++q)
            {
                int imin;
                if (q < K - 1)
                {
                    imin = Math.Max(1, q);
                }
                else
                {
                    // No need to compute S[K-1][0] ... S[K-1][N-2]
                    imin = N - 1;
                }
                /*
                # ifdef DEBUG
                                // std::cout << std::endl << "q=" << q << ":" << std::endl;
                #endif
                */
                // fill_row_k_linear_recursive(imin, N-1, 1, q, jseq, S, J, sum_x, sum_x_sq);
                // fill_row_k_linear(imin, N-1, q, S, J, sum_x, sum_x_sq);
                if (method == Method.Linear)
                {
                    Fill.SMAWK(imin, N - 1, q, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, criterion);
                }
                else if (method == Method.LogLinear)
                {
                    Fill.LogLinear(imin, N - 1, q, q, N - 1, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, criterion);
                }
                else if (method == Method.Quadratic)
                {
                    Fill.Quadratic(imin, N - 1, q, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, criterion);
                }
                else
                {
                    throw new Exception("ERROR: unknown method " + method + "!");
                }

                /*
                     #if DEBUG

                fill_row_q_log_linear(imin, N - 1, q, q, N - 1, SS, JJ, sum_x, sum_x_sq, sum_w, sum_w_sq, criterion);

                for (int i = imin; i < N; ++i)
                {
                    if (S[q][i] != SS[q][i] || J[q][i] != JJ[q][i])
                    {
                        std::cout << "ERROR: q=" << q << ", i=" << i << std::endl;
                        std::cout << "\tS=" << S[q][i] << "\tJ=" << J[q][i] << std::endl;
                        std::cout << "Truth\tSS=" << SS[q][i] << "\tJJ=" << JJ[q][i];
                        std::cout << std::endl;
                        assert(false);

                    }
                    else
                    {

                            std::cout << "OK: q=" << q << ", i=" << i << std::endl;
                            std::cout << "\tS=" << S[q][i] << "\tJ=" << J[q][i] << std::endl;
                            std::cout << "Truth\tSS=" << SS[q][i] << "\tJJ=" << JJ[q][i];
                            std::cout << std::endl;

                            }

                        }
                #endif
                */
            }

/*
# ifdef DEBUG
            std::cout << "Linear & log-linear code returned identical dp index matrix."
                      << std::endl;
#endif
*/
        }

        public static void Backtrack(number[] x, int[][] J, int[] cluster, number[] centers, number[] withinss, number[] count)
        {
            int K = J.Length;
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int q = K - 1; q >= 0; --q)
            {
                cluster_left = J[q][cluster_right];

                for (int i = cluster_left; i <= cluster_right; ++i)
                    cluster[i] = q;

                number sum = 0;

                for (int i = cluster_left; i <= cluster_right; ++i)
                    sum += x[i];

                centers[q] = sum / (cluster_right - cluster_left + 1);

                for (int i = cluster_left; i <= cluster_right; ++i)
                    withinss[q] += (x[i] - centers[q]) * (x[i] - centers[q]);

                count[q] = (cluster_right - cluster_left + 1);

                if (q > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }

        public static void BacktrackL1(number[] x, int[][] J, int[] cluster, number[] centers, number[] withinss, number[] count)
        {
            int K = J.Length;
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int q = (K) - 1; q >= 0; --q)
            {
                cluster_left = J[q][cluster_right];

                for (int i = cluster_left; i <= cluster_right; ++i)
                    cluster[i] = q;

                centers[q] = x[(cluster_right + cluster_left) >> 1];

                for (int i = cluster_left; i <= cluster_right; ++i)
                    withinss[q] += Math.Abs(x[i] - centers[q]);

                count[q] = (cluster_right - cluster_left + 1);

                if (q > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }

        public static void BacktrackL2Y(number[] x, number[] y, int[][] J, int[] cluster, number[] centers, number[] withinss, number[] count)
        {
            int K = J.Length;
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int q = (K) - 1; q >= 0; --q)
            {
                cluster_left = J[q][cluster_right];

                for (int i = cluster_left; i <= cluster_right; ++i)
                    cluster[i] = q;

                number sum = 0;
                number sum_y = 0;

                for (int i = cluster_left; i <= cluster_right; ++i)
                {
                    sum += x[i];
                    sum_y += y[i];
                }

                centers[q] = sum / (cluster_right - cluster_left + 1);

                number mean_y = sum_y / (cluster_right - cluster_left + 1);

                for (int i = cluster_left; i <= cluster_right; ++i)
                    withinss[q] += (y[i] - mean_y) * (y[i] - mean_y);

                count[q] = (cluster_right - cluster_left + 1);

                if (q > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }

        public static void Backtrack(number[] x, int[][] J, int[] count, int K)
        {
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int q = K - 1; q >= 0; --q)
            {
                cluster_left = J[q][cluster_right];
                count[q] = cluster_right - cluster_left + 1;
                if (q > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }

        public static void BacktrackWeighted(number[] x, number[] y, int[][] J, int[] cluster, number[] centers, number[] withinss, number[] weights)
        {
            int K = J.Length;
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int k = K - 1; k >= 0; --k)
            {
                cluster_left = J[k][cluster_right];

                for (int i = cluster_left; i <= cluster_right; ++i)
                    cluster[i] = k;

                number sum = 0;
                number weight = 0;

                for (int i = cluster_left; i <= cluster_right; ++i)
                {
                    sum += x[i] * y[i];
                    weight += y[i];
                }

                centers[k] = sum / weight;

                for (int i = cluster_left; i <= cluster_right; ++i)
                    withinss[k] += y[i] * (x[i] - centers[k]) * (x[i] - centers[k]);

                weights[k] = weight;

                if (k > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }

        public static void BacktrackWeighted(number[] x, number[] y, int[][] J, int[] count, number[] weights, int K)
        {
            int N = J[0].Length;
            int cluster_right = N - 1;
            int cluster_left;

            // Backtrack the clusters from the dynamic programming matrix
            for (int k = K - 1; k >= 0; --k)
            {
                cluster_left = J[k][cluster_right];
                count[k] = cluster_right - cluster_left + 1;

                weights[k] = 0;
                for (int i = cluster_left; i <= cluster_right; ++i)
                {
                    weights[k] += y[i];
                }

                if (k > 0)
                {
                    cluster_right = cluster_left - 1;
                }
            }
        }
    }
}
