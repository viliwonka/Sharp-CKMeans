/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */

using System;
using number = System.Decimal;

namespace Sharp.CKMeans
{
    public static class Fill
    {
        public static void Quadratic(
            int imin, int imax, int q,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {
            // Assumption: each cluster must have at least one point.
            for (int i = imin; i <= imax; ++i)
            {
                S[q][i] = S[q - 1][i - 1];
                J[q][i] = i;

                int jmin = Math.Max(q, (int)J[q - 1][i]);

                for (int j = i - 1; j >= jmin; --j)
                {
                    number Sj = S[q - 1][j - 1] + WithinCluster.Dissimilarity(criterion, j, i, sum_x, sum_x_sq, sum_w, sum_w_sq);

                    // ssq(j, i, sum_x, sum_x_sq, sum_w)

                    if (Sj < S[q][i])
                    {
                        S[q][i] = Sj;
                        J[q][i] = j;
                    }
                }
            }
        }


        public static void LogLinear(
            int imin, int imax, int q,
            int jmin, int jmax,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {

            if (imin > imax)
            {
                return;
            }

            int N = S[0].Length;
            int i = (imin + imax) / 2;
#if DEBUG
            // std::cout << "  i=" << i << ": ";
#endif
            // Initialization of S[q][i]:
            S[q][i] = S[q - 1][i - 1];
            J[q][i] = i;

            int jlow = q; // the lower end for j

            if (imin > q)
            {
                // jlow = std::max(jlow, (int)J[q][imin-1]);
                jlow = Math.Max(jlow, jmin);
            }
            jlow = Math.Max(jlow, J[q - 1][i]);

            int jhigh = i - 1; // the upper end for j
            if (imax < N - 1)
            {
                // jhigh = std::min(jhigh, (int)J[q][imax+1]);
                jhigh = Math.Min(jhigh, jmax);
            }

#if DEBUG
            // std::cout << "    j-=" << jlow << ", j+=" << jhigh << ": ";
#endif

            for (int j = jhigh; j >= jlow; --j)
            {

                // compute s(j,i)
                number sji = WithinCluster.SSQ(j, i, sum_x, sum_x_sq, sum_w);

                // MS May 11, 2016 Added:
                if (sji + S[q - 1][jlow - 1] >= S[q][i]) break;

                // Examine the lower bound of the cluster border
                // compute s(jlow, i)
                number sjlowi = WithinCluster.Dissimilarity(criterion, jlow, i, sum_x, sum_x_sq, sum_w, sum_w_sq);
                // ssq(jlow, i, sum_x, sum_x_sq, sum_w);

                number SSQ_jlow = sjlowi + S[q - 1][jlow - 1];

                if (SSQ_jlow < S[q][i])
                {
                    // shrink the lower bound
                    S[q][i] = SSQ_jlow;
                    J[q][i] = jlow;
                }

                jlow++;

                number SSQ_j = sji + S[q - 1][j - 1];
                if (SSQ_j < S[q][i])
                {
                    S[q][i] = SSQ_j;
                    J[q][i] = j;
                }
            }

#if DEBUG
            //std::cout << // " q=" << q << ": " <<
            //  "\t" << S[q][i] << "\t" << J[q][i];
            //std::cout << std::endl;
#endif

            jmin = (imin > q) ? (int)J[q][imin - 1] : q;
            jmax = (int)J[q][i];

            LogLinear(imin, i - 1, q, jmin, jmax,
                                  S, J, sum_x, sum_x_sq, sum_w,
                                  sum_w_sq, criterion);

            jmin = (int)J[q][i];
            jmax = (imax < N - 1) ? (int)J[q][imax + 1] : imax;
            LogLinear(i + 1, imax, q, jmin, jmax,
                                  S, J, sum_x, sum_x_sq, sum_w,
                                  sum_w_sq, criterion);
        }

        //SMAWK

        private static void ReduceInPlace(
            int imin, int imax, int istep, int q,
            int[] js, out int[] js_red,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {
            int N = (imax - imin) / istep + 1;

            js_red = js;

            if (N >= js.Length)
            {
                return;
            }

            // Two positions to move candidate j's back and forth
            int left = -1; // points to last favorable position / column
            int right = 0; // points to current position / column

            int m = js_red.Length;

            while (m > N)
            { // js_reduced has more than N positions / columns

                int p = (left + 1);

                int i = (imin + p * istep);
                int j = (js_red[right]);
                number Sl = (S[q - 1][j - 1] + WithinCluster.Dissimilarity(criterion, j, i, sum_x, sum_x_sq, sum_w, sum_w_sq));
                // ssq(j, i, sum_x, sum_x_sq, sum_w));

                int jplus1 = (js_red[right + 1]);
                number Slplus1 = (S[q - 1][jplus1 - 1] + WithinCluster.Dissimilarity(criterion, jplus1, i, sum_x, sum_x_sq, sum_w, sum_w_sq));
                // ssq(jplus1, i, sum_x, sum_x_sq, sum_w));

                if (Sl < Slplus1 && p < N - 1)
                {
                    js_red[++left] = j; // i += istep;
                    right++; // move on to next position / column p+1
                }
                else if (Sl < Slplus1 && p == N - 1)
                {
                    js_red[++right] = j; // delete position / column p+1
                    m--;
                }
                else
                { // (Sl >= Slplus1)
                    if (p > 0)
                    { // i > imin
                      // delete position / column p and
                      //   move back to previous position / column p-1:
                        js_red[right] = js_red[left--];
                        // p --; // i -= istep;
                    }
                    else
                    {
                        right++; // delete position / column 0
                    }
                    m--;
                }
            }

            for (int r = (left + 1); r < m; ++r)
            {
                js_red[r] = js_red[right++];
            }

            Array.Resize(ref js_red, m);

            return;
        }

        private static void FillEvenPositions(
            int imin, int imax, int istep, int q,
            int[] js,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {
            // Derive j for even rows (0-based)
            int n = js.Length;
            int istepx2 = (istep << 1);
            int jl = js[0];
            for (int i = imin, r = 0; i <= imax; i += istepx2)
            {
                // auto jmin = (i == imin) ? js[0] : J[q][i - istep];
                while (js[r] < jl)
                {
                    // Increase r until it points to a value of at least jmin
                    r++;
                }

                // Initialize S[q][i] and J[q][i]
                S[q][i] = S[q - 1][js[r] - 1] +
                  WithinCluster.Dissimilarity(criterion, js[r], i, sum_x, sum_x_sq, sum_w, sum_w_sq);
                // ssq(js[r], i, sum_x, sum_x_sq, sum_w);
                J[q][i] = js[r]; // rmin

                // Look for minimum S upto jmax within js
                int jh = (i + istep <= imax)
                  ? J[q][i + istep] : js[n - 1];

                int jmax = Math.Min(jh, i);

                number sjimin = WithinCluster.Dissimilarity(criterion, jmax, i, sum_x, sum_x_sq, sum_w, sum_w_sq);
                // ssq(jmax, i, sum_x, sum_x_sq, sum_w)


                for (++r; r < n && js[r] <= jmax; r++)
                {

                    int jabs = js[r];

                    if (jabs > i) break;

                    if (jabs < J[q - 1][i]) continue;

                    number s = WithinCluster.Dissimilarity(criterion, jabs, i, sum_x, sum_x_sq, sum_w, sum_w_sq);
                    // (ssq(jabs, i, sum_x, sum_x_sq, sum_w));
                    number Sj = (S[q - 1][jabs - 1] + s);

                    if (Sj <= S[q][i])
                    {
                        S[q][i] = Sj;
                        J[q][i] = js[r];
                    }
                    else if (S[q - 1][jabs - 1] + sjimin > S[q][i])
                    {
                        break;
                    }
                    /*else if(S[q-1][js[rmin]-1] + s > S[q][i]) {
                        break;
                    }*/
                }

                r--;
                jl = jh;
            }
        }

        private static void FindMinFromCandidates(
            int imin, int imax, int istep, int q,
            int[] js,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {
            int rmin_prev = 0;

            for (int i = (imin); i <= imax; i += istep)
            {
                int rmin = rmin_prev;

                // Initialization of S[q][i] and J[q][i]
                S[q][i] = S[q - 1][js[rmin] - 1] + WithinCluster.Dissimilarity(criterion, js[rmin], i, sum_x, sum_x_sq, sum_w, sum_w_sq);
                // ssq(js[rmin], i, sum_x, sum_x_sq, sum_w);
                J[q][i] = js[rmin];

                for (int r = (rmin + 1); r < js.Length; ++r)
                {
                    int j_abs = js[r];

                    if (j_abs < J[q - 1][i]) continue;
                    if (j_abs > i) break;

                    number Sj = (S[q - 1][j_abs - 1] + WithinCluster.Dissimilarity(criterion, j_abs, i, sum_x, sum_x_sq, sum_w, sum_w_sq));
                    // ssq(j_abs, i, sum_x, sum_x_sq, sum_w));
                    if (Sj <= S[q][i])
                    {
                        S[q][i] = Sj;
                        J[q][i] = js[r];
                        rmin_prev = r;
                    }
                }
            }
        }

        private static void SMAWK(
            int imin, int imax, int istep, int q,
            int[] js,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {
#if DEBUG_REDUCE
            std::cout << "i:" << '[' << imin << ',' << imax << ']' << '+' << istep
                        << std::endl;
#endif

            if (imax - imin <= 0 * istep)
            { // base case only one element left

                FindMinFromCandidates(
                  imin, imax, istep, q, js, S, J, sum_x, sum_x_sq, sum_w,
                  sum_w_sq, criterion
                );

            }
            else
            {
                // REDUCE

#if DEBUG_REDUCE
                std::cout << "js:";
                for (size_t l = 0; l < js.size(); ++l)
                {
                    std::cout << js[l] << ",";
                }
                std::cout << std::endl;
                std::cout << std::endl;
#endif
                int[] js_odd;

                ReduceInPlace(imin, imax, istep, q, js, out js_odd,
                                S, J, sum_x, sum_x_sq, sum_w,
                                sum_w_sq, criterion);

                int istepx2 = (istep << 1);
                int imin_odd = (imin + istep);
                int imax_odd = (imin_odd + (imax - imin_odd) / istepx2 * istepx2);

                // Recursion on odd rows (0-based):
                SMAWK(imin_odd, imax_odd, istepx2,
                      q, js_odd, S, J, sum_x, sum_x_sq, sum_w,
                      sum_w_sq, criterion);

#if DEBUG_REDUCE
                std::cout << "js_odd (reduced):";
                for (size_t l = 0; l < js_odd.size(); ++l)
                {
                    std::cout << js_odd[l] << ",";
                }
                std::cout << std::endl << std::endl;

                std::cout << "even pos:";
                for (int i = imin; i < imax; i += istepx2)
                {
                    std::cout << i << ",";
                }
                std::cout << std::endl << std::endl;
#endif

                FillEvenPositions(imin, imax, istep, q, js,
                                    S, J, sum_x, sum_x_sq, sum_w,
                                    sum_w_sq, criterion);
            }
        }

        public static void SMAWK(
            int imin, int imax, int q,
            number[][] S, int[][] J,
            number[] sum_x, number[] sum_x_sq, number[] sum_w, number[] sum_w_sq,
            DissimilarityType criterion)
        {

            int[] js = new int[imax - q + 1];

            int abs = (q);

            for (int iter = 0; iter < js.Length; iter++)
            {
                js[iter] = abs++;
            }

            SMAWK(imin, imax, 1, q, js, S, J, sum_x, sum_x_sq, sum_w, sum_w_sq, criterion);
        }
    }
}
