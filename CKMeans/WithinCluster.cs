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

    public static class WithinCluster
    {
        public static number SSQ(int j, int i, number[] sum_x, number[] sum_x_sq, number[] sum_w = null)
        {
            number sji = 0;

            if (sum_w == null || sum_w.Length == 0)
            { // equally weighted version
                if (j >= i)
                {
                    sji = 0;
                }
                else if (j > 0)
                {
                    number muji = (sum_x[i] - sum_x[j - 1]) / (i - j + 1);
                    sji = sum_x_sq[i] - sum_x_sq[j - 1] - (i - j + 1) * muji * muji;
                }
                else
                {
                    sji = sum_x_sq[i] - sum_x[i] * sum_x[i] / (i + 1);
                }
            }
            else
            { // unequally weighted version
                if (sum_w[j] >= sum_w[i])
                {
                    sji = 0;
                }
                else if (j > 0)
                {
                    number muji = (sum_x[i] - sum_x[j - 1]) / (sum_w[i] - sum_w[j - 1]);
                    sji = sum_x_sq[i] - sum_x_sq[j - 1] - (sum_w[i] - sum_w[j - 1]) * muji * muji;
                }
                else
                {
                    sji = sum_x_sq[i] - sum_x[i] * sum_x[i] / sum_w[i];
                }
            }

            sji = (sji < 0) ? 0 : sji;
            return sji;
        }

        public static number SABS(int j, int i, number[] sum_x, number[] sum_w = null)
        {
            number sji = 0;

            if (sum_w != null || sum_w.Length == 0)
            { // equally weighted version
                if (j >= i)
                {
                    sji = 0;
                }
                else if (j > 0)
                {
                    int l = (i + j) >> 1; // l is the index to the median of the cluster

                    if (((i - j + 1) % 2) == 1)
                    {
                        // If i-j+1 is odd, we have
                        //   sum (x_l - x_m) over m = j .. l-1
                        //   sum (x_m - x_l) over m = l+1 .. i
                        sji = -sum_x[l - 1] + sum_x[j - 1] + sum_x[i] - sum_x[l];
                    }
                    else
                    {
                        // If i-j+1 is even, we have
                        //   sum (x_l - x_m) over m = j .. l
                        //   sum (x_m - x_l) over m = l+1 .. i
                        sji = -sum_x[l] + sum_x[j - 1] + sum_x[i] - sum_x[l];
                    }
                }
                else
                { // j==0
                    int l = i >> 1; // l is the index to the median of the cluster

                    if (((i + 1) % 2) == 1)
                    {
                        // If i-j+1 is odd, we have
                        //   sum (x_m - x_l) over m = 0 .. l-1
                        //   sum (x_l - x_m) over m = l+1 .. i
                        sji = -sum_x[l - 1] + sum_x[i] - sum_x[l];
                    }
                    else
                    {
                        // If i-j+1 is even, we have
                        //   sum (x_m - x_l) over m = 0 .. l
                        //   sum (x_l - x_m) over m = l+1 .. i
                        sji = -sum_x[l] + sum_x[i] - sum_x[l];
                    }
                }
            }
            else
            { // unequally weighted version
              // no exact solutions are known.
            }

            sji = (sji < 0) ? 0 : sji;
            return sji;
        }

        public static number Dissimilarity(DissimilarityType disType, int j, int i,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq = null)
        {
            number d = 0;

            switch (disType)
            {
                case DissimilarityType.L1:
                    d = SABS(j, i, sum_x, sum_w);
                    break;
                case DissimilarityType.L2:
                    d = SSQ(j, i, sum_x, sum_x_sq, sum_w);
                    break;
                case DissimilarityType.L2Y:
                    d = SSQ(j, i, sum_w, sum_w_sq);
                    break;
            }
            return d;
        }
    }
}
