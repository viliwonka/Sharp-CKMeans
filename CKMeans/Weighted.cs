/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */
using System;
using System.Linq;
using System.Collections.Generic;
using number = System.Decimal;

namespace Sharp.CKMeans
{
    public class Weighted
    {
        static number PI = (number)Math.PI;

        public static void ShiftedDataVariance(number[] x, number[] y, number totalWeight, int left, int right, out number mean, out number variance)
        {

            number sum = 0;
            number sumsq = 0;

            mean = 0;
            variance = 0;

            int n = right - left + 1;

            if (right >= left)
            {
                number median = x[(left + right) / 2];

                for (int i = left; i <= right; ++i)
                {
                    sum += (x[i] - median) * y[i];
                    sumsq += (x[i] - median) * (x[i] - median) * y[i];
                }

                mean = (sum / totalWeight) + median;

                if (n > 1)
                {
                    variance = (sumsq - ((sum * sum) / totalWeight)) / (totalWeight - 1);
                }
            }
        }

        public static int SelectLevels(number[] x, number[] y, int[][] J, int Kmin, int Kmax, double[] BIC)
        {
            int N = x.Length;

            /*if (Kmin == Kmax)
            {
                return Kmin;
            }*/


            if (Kmin > Kmax || N < 2)
            {
                return Math.Min(Kmin, Kmax);
            }


            if (BIC.Length != Kmax - Kmin + 1)
            {
                Array.Resize(ref BIC, Kmax - Kmin + 1);
            }

            // double variance_min, variance_max;
            // range_of_variance(x, variance_min, variance_max);

            int Kopt = Kmin;

            double maxBIC = 0;

            number[] lambda = new number[Kmax];
            number[] mu = new number[Kmax];
            number[] sigma2 = new number[Kmax];
            double[] coeff = new double[Kmax];
            int[] counts = new int[Kmax];
            number[] weights = new number[Kmax];

            for (int K = Kmin; K <= Kmax; ++K)
            {
                // std::vector< std::vector< size_t > > JK(J.begin(), J.begin()+K);

                // Backtrack the matrix to determine boundaries between the bins.
                DynamicProgramming.BacktrackWeighted(x, y, J, counts, weights, K);

                // double totalweight = std::accumulate(weights.begin(), weights.begin() + K, 0, std::plus<double>());

                number totalweight;

                totalweight = 0;
                for (int k = 0; k < K; k++)
                {
                    totalweight += weights[k];
                }

                int indexLeft = 0;
                int indexRight;

                for (int k = 0; k < K; ++k)
                { // Estimate GMM parameters first

                    lambda[k] = weights[k] / totalweight;

                    indexRight = indexLeft + counts[k] - 1;

                    ShiftedDataVariance(x, y, weights[k], indexLeft, indexRight, out mu[k], out sigma2[k]);

                    if (sigma2[k] == 0 || counts[k] == 1)
                    {
                        number dmin;

                        if (indexLeft > 0 && indexRight < N - 1)
                        {
                            dmin = Math.Min(x[indexLeft] - x[indexLeft - 1], x[indexRight + 1] - x[indexRight]);
                        }
                        else if (indexLeft > 0)
                        {
                            dmin = x[indexLeft] - x[indexLeft - 1];
                        }
                        else
                        {
                            dmin = x[indexRight + 1] - x[indexRight];
                        }

                        // std::cout << "sigma2[k]=" << sigma2[k] << "==>";
                        if (sigma2[k] == 0) sigma2[k] = (dmin * dmin) / (number)4 / (number)9;
                        if (counts[k] == 1) sigma2[k] = (dmin * dmin);
                        // std::cout << sigma2[k] << std::endl;
                    }

                    /*
                     if(sigma2[k] == 0) sigma2[k] = variance_min;
                     if(size[k] == 1) sigma2[k] = variance_max;
                     */

                    coeff[k] = (double)lambda[k] / Math.Sqrt((double)(2 * PI * sigma2[k]));

                    indexLeft = indexRight + 1;
                }

                double loglikelihood = 0;

                for (int i = 0; i < N; ++i)
                {
                    double L = 0;
                    for (int k = 0; k < K; ++k)
                    {
                        L += (coeff[k]) * Math.Exp((double)(-(x[i] - mu[k]) * (x[i] - mu[k]) / (2 * sigma2[k])));
                    }

                    loglikelihood += ((double)y[i]) * Math.Log(L);
                }

                // double & bic = BIC[K-Kmin];

                // Compute the Bayesian information criterion

                double bic = 2 * loglikelihood - (3 * K - 1) * Math.Log((double)totalweight);  //(K*3-1)

                // std::cout << "k=" << K << ": Loglh=" << loglikelihood << ", BIC=" << BIC << std::endl;

                if (K == Kmin)
                {
                    maxBIC = bic;
                    Kopt = Kmin;
                }
                else
                {
                    if (bic > maxBIC)
                    {
                        maxBIC = bic;
                        Kopt = K;
                    }
                }

                BIC[K - Kmin] = bic;
            }
            return Kopt;
        }

    }
}