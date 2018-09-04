/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */
using System;
using System.Collections.Generic;
using number = System.Decimal;

namespace Sharp.CKMeans
{
    public static class NonWeighted
    {

        static number PI = (number)Math.PI;


        public static void ShiftedDataVariance(number[] x, int left, int right, out number mean, out number variance)
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
                    sum += x[i] - median;
                    sumsq += (x[i] - median) * (x[i] - median);
                }

                mean = sum / n + median;

                if (n > 1)
                {
                    variance = (sumsq - sum * sum / n) / (n - 1);
                }
            }
        }

        public static void RangeOfVariance(number[] x, out number variance_min, out number variance_max)
        {
            number dposmin = x[x.Length - 1] - x[0];
            number dposmax = 0;

            for (int n = 1; n < x.Length; ++n)
            {
                number d = x[n] - x[n - 1];
                if (d > 0 && dposmin > d)
                {
                    dposmin = d;
                }
                if (d > dposmax)
                {
                    dposmax = d;
                }
            }

            variance_min = dposmin * dposmin / 3;
            variance_max = dposmax * dposmax;
        }

        public static int SelectLevels(number[] x, int[][] J, int Kmin, int Kmax, double[] BIC)
        {
            int N = x.Length;

            if (Kmin >= Kmax || N < 2)
            {
                return Math.Min(Kmin, Kmax);
            }

            if(BIC.Length != Kmax - Kmin + 1)
            {
                Array.Resize(ref BIC, Kmax - Kmin + 1);

            }

            // double variance_min, variance_max;
            // range_of_variance(x, variance_min, variance_max);

            int Kopt = Kmin;

            double maxBIC = 0;

            double[] lambda = new double[Kmax];
            number[] mu = new number[Kmax];
            number[] sigma2 = new number[Kmax];
            double[] coeff = new double[Kmax];

            for (int K = Kmin; K <= Kmax; ++K)
            {
                int[] size = new int[K];

                // Backtrack the matrix to determine boundaries between the bins.
                DynamicProgramming.Backtrack(x, J, size, K);

                int indexLeft = 0;
                int indexRight;

                for (int k = 0; k < K; ++k)
                { // Estimate GMM parameters first
                    lambda[k] = size[k] / (double)N;

                    indexRight = indexLeft + size[k] - 1;

                    ShiftedDataVariance(x, indexLeft, indexRight, out mu[k], out sigma2[k]);

                    if (sigma2[k] == 0 || size[k] == 1)
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
                        if (sigma2[k] == 0) sigma2[k] = dmin * dmin / ((number)4) / ((number)9);
                        if (size[k] == 1) sigma2[k] = dmin * dmin;
                        // std::cout << sigma2[k] << std::endl;
                    }

                    /*
                     if(sigma2[k] == 0) sigma2[k] = variance_min;
                    if(size[k] == 1) sigma2[k] = variance_max;
                    */

                    coeff[k] = lambda[k] / Math.Sqrt((double)(2 * PI * sigma2[k]));

                    indexLeft = indexRight + 1;
                }

                double loglikelihood = 0;

                for (int i = 0; i < N; ++i)
                {
                    double L = 0;
                    for (int k = 0; k < K; ++k)
                    {
                        L += coeff[k] * Math.Exp((double)( -(x[i] - mu[k]) * (x[i] - mu[k]) / (2 * sigma2[k]) ));
                    }

                    loglikelihood += Math.Log(L);
                }

                // Compute the Bayesian information criterion
                BIC[K - Kmin] = 2d * loglikelihood - (3d * K - 1d) * Math.Log(N);  //(K*3-1)

                double bic = BIC[K - Kmin];

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
            }
            return Kopt;
        }

    }
}