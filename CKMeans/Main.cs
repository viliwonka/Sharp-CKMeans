/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using number = System.Decimal;

namespace Sharp.CKMeans
{

    public static class Main
    {

        public class CKResult
        {
            public int[] Clusters { get; private set; }
            public number[] Centers { get; private set; }
            public number[] Withinss { get; private set; }
            public number[] Sizes { get; private set; }
            public double[] BIC { get; private set; }

            public CKResult(int[] clusters, number[] centers, number[] withinss, number[] size, double[] bic)
            {
                Clusters = clusters;
                Centers = centers;
                Withinss = withinss;
                Sizes = size;
                BIC = bic;
            }
        }

        //L2
        public static CKResult CKMeans1D(number[] x, number[] y, int Kmin, int Kmax, Method method = Method.LogLinear)
        {
            int[] clusters;
            number[] centers;
            number[] withinss;
            number[] size;
            double[] BIC;

            KMeans(x, y, Kmin, Kmax, out clusters, out centers, out withinss, out size, out BIC, method, DissimilarityType.L2);

            CKResult result = new CKResult(clusters, centers, withinss, size, BIC);

            return result;
        }

        //L1
        public static CKResult CKMedian1D(number[] x, number[] y, int Kmin, int Kmax, Method method = Method.LogLinear)
        {
            int[] clusters;
            number[] centers;
            number[] withinss;
            number[] size;
            double[] BIC;

            KMeans(x, y, Kmin, Kmax, out clusters, out centers, out withinss, out size, out BIC, method, DissimilarityType.L1);

            CKResult result = new CKResult(clusters, centers, withinss, size, BIC);

            return result;
        }

        //L2Y
        public static CKResult CKSegs1D(number[] x, number[] y, int Kmin, int Kmax, Method method = Method.LogLinear)
        {
            int[] clusters;
            number[] centers;
            number[] withinss;
            number[] size;
            double[] BIC;

            KMeans(x, y, Kmin, Kmax, out clusters, out centers, out withinss, out size, out BIC, method, DissimilarityType.L2Y);

            CKResult result = new CKResult(clusters, centers, withinss, size, BIC);

            return result;
        }

        private static void KMeans(number[] x, number[] y, int Kmin, int Kmax, out int[] clusters, out number[] centers, out number[] withinss, out number[] size, out double[] BIC, Method method, DissimilarityType criterion)
        {
            // Input:
            // x -- an array of double precision numbers, not necessarily sorted
            // Kmin -- the minimum number of clusters expected
            // Kmax -- the maximum number of clusters expected
            // NOTE: All vectors in this program is considered starting at position 0.

            int N = x.Length;

            clusters = new int[N];
            BIC = new double[Kmax - Kmin];

            int[] order = new int[N];

            for (int i = 0; i < order.Length; ++i)
            {
                order[i] = i;
            }

            bool is_sorted = true;

            for (int i = 0; i < N - 1; ++i)
            {
                if (x[i] > x[i + 1])
                {
                    is_sorted = false;
                    break;
                }
            }

            number[] x_sorted = null;

            number[] y_sorted = null;
            bool is_equally_weighted = true;

            if (!is_sorted)
            {
                x_sorted = new number[x.Length];

                Array.Copy(x, x_sorted, x.Length);
                Array.Sort(x_sorted, order);

                for (int i = 0; i < x_sorted.Length; i++)
                {
                    x_sorted[i] = x[order[i]];

                }
            }
            else
            {
                x_sorted = x;
            }

            // check to see if unequal weight is provided
            if (y != null)
            {
                is_equally_weighted = true;
                for (int i = 1; i < N; ++i)
                {
                    if (y[i] != y[i - 1])
                    {
                        is_equally_weighted = false;
                        break;
                    }
                }
            }

            if (!is_equally_weighted)
            {
                y_sorted = new number[N];

                for (int i = 0; i < N; ++i)
                {
                    y_sorted[i] = y[order[i]];
                }
            }
            else
            {
                y = null;
            }

            int nUnique = 1;

            if(N == 0)
            {
                nUnique = 0;
            }

            if (N > 1)
            {
                for (int i = 1; i < N; i++)
                {
                    if (x_sorted[i - 1] != x_sorted[i])
                        nUnique++;
                }
            }

            Kmax = nUnique < Kmax ? nUnique : Kmax;

            if (nUnique > 1)
            { // The case when not all elements are equal.

                number[][] S = new number[Kmax][];

                for (int i = 0; i < Kmax; i++)
                {
                    S[i] = new number[N];
                }

                int[][] J = new int[Kmax][];

                for (int i = 0; i < Kmax; i++)
                {
                    J[i] = new int[N];
                }

                int Kopt;

                DynamicProgramming.FillMatrix(x_sorted, y_sorted, S, J, method, criterion);

                // Fill in dynamic programming matrix
                if (is_equally_weighted)
                {

                    Kopt = NonWeighted.SelectLevels(x_sorted, J, Kmin, Kmax, BIC);
                }
                else
                {

                    switch (criterion)
                    {
                        case DissimilarityType.L2Y:
                            Kopt = NonWeighted.SelectLevels(y_sorted, J, Kmin, Kmax, BIC);
                        break;

                        default:
                            Kopt = Weighted.SelectLevels(x_sorted, y_sorted, J, Kmin, Kmax, BIC);

                        break;
                    }
                }

                centers = new number[Kopt];
                withinss = new number[Kopt];
                size = new number[Kopt];

                if (Kopt < Kmax)
                { // Reform the dynamic programming matrix S and J

                    Array.Resize(ref J, Kopt);
                }

                int[] cluster_sorted = new int[N];

                // Backtrack to find the clusters beginning and ending indices
                if (is_equally_weighted && criterion == DissimilarityType.L1)
                {
                    DynamicProgramming.BacktrackL1(x_sorted, J, cluster_sorted, centers, withinss, size);
                }
                else if (is_equally_weighted && criterion == DissimilarityType.L2)
                {
                    DynamicProgramming.Backtrack(x_sorted, J, cluster_sorted, centers, withinss, size);
                }
                else if (criterion == DissimilarityType.L2Y)
                {
                    DynamicProgramming.BacktrackL2Y(x_sorted, y_sorted, J, cluster_sorted, centers, withinss, size);
                }
                else
                {
                    DynamicProgramming.BacktrackWeighted(x_sorted, y_sorted, J, cluster_sorted, centers, withinss, size);
                }

                /*#if DEBUG
                                std::cout << "backtrack done." << std::endl;
                #endif*/

                for (int i = 0; i < N; ++i)
                {
                    // Obtain clustering on data in the original order
                    clusters[order[i]] = cluster_sorted[i];
                }
            }
            else
            {


                // A single cluster that contains all elements
                for (int i = 0; i < N; ++i)
                {
                    clusters[i] = 0;
                }


                centers = new number[1];
                withinss = new number[1];
                size = new number[1];

                centers[0] = x[0];
                withinss[0] = 0;
                size[0] = N * (is_equally_weighted ? 1 : y[0]);
            }
        }
    }
}
