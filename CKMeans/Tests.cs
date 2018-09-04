/*
 * Original library is https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html
 * Original terms (credits & licence & authors..) apply.
 * Ported by Vili Volčini.
 *
 */
using NUnit.Framework;
using System.Linq;
using System;

using number = System.Decimal;

namespace Sharp.CKMeans
{
    [TestFixture]
    class KMeansTest
    {
        // all data centered around 1
        [TestCase(new double[] { 1, 1, 1 })]
        [TestCase(new double[] { -1, 0, 1, 2, 3 })]
        [TestCase(new double[] { -2, -1, +1, +2, +5 })]
        public void SingleClusterTest(double[] input)
        {
            number[] data = input.Select(x => (number)x).ToArray();

            var result = Main.CKMeans1D(data, null, 1, 1, Method.Linear);

            Assert.AreEqual(data.Length, result.Clusters.Length);
            Assert.AreEqual(data.Length, result.Sizes[0]);
            Assert.AreEqual(1, result.Centers.Length);
            Assert.AreEqual(1, result.Centers[0]);
        }

        [TestCase(new double[] { -1, 0, +1 })]
        [TestCase(new double[] { -2, -1, 0, +1, +2 })]
        [TestCase(new double[] { 1, 1, 2, 2, 3, 3 })]
        [TestCase(new double[] { 1, 1.1, 2, 2.1, 3, 3.1 })]
        public void ThreeClustersTest(double[] input)
        {
            number[] data = input.Select(x => (number)x).ToArray();

            var result = Main.CKMeans1D(data, null, 3, 3, Method.Linear);

            Assert.AreEqual(data.Length, result.Clusters.Length);
            Assert.AreEqual(3, result.Centers.Length);
            Assert.AreEqual(data.Length, result.Sizes.Sum());
        }

        [TestCase(new double[] { 1, 1.1, 2, 2.1, 3, 3.1, 4, 5, 6, 6.1, 7.1, 7.5, 7.7, 7.8, 7.9, 8, 8.1, 9, 9.1, 10.1, 10.2, 10.3, 10.4 })]
        [TestCase(new double[] { 0, 0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.34, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9, 1.11, 1.23, 2, 3, 4, 10, 11, 12, 13, 14, 15, 20, 21, 21.1, 21.3, 21.4, 21.5, 21.7, 21.10, 30, 31, 31, 32, 40, 41, 41, 42 })]
        [TestCase(new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 9, 9, 9, 9 })]
        [TestCase(new double[] { 1, 1, 0, 1, 0, 2, 3, 6, 0, 3, 7, 2, 0, 6, 9, 5, 1, 0, -1, -2, -3 })]
        public void VariableClustersTest(double[] input)
        {
            number[] data = input.Select(x => (number)x).ToArray();

            var resultQuad = Main.CKMeans1D(data, null, 1, 10, Method.Quadratic);
            var resultLinear = Main.CKMeans1D(data, null, 1, 10, Method.Linear);
            var resultLogLin = Main.CKMeans1D(data, null, 1, 10, Method.LogLinear);

            Assert.AreEqual(resultQuad.Centers.Length, resultLogLin.Centers.Length);
            Assert.AreEqual(resultQuad.Centers.Length, resultLinear.Centers.Length);

            Assert.AreEqual(resultQuad.BIC.Length, resultLogLin.BIC.Length);
            Assert.AreEqual(resultQuad.BIC.Length, resultLinear.BIC.Length);

            // test cluster indexes
            for (int i = 0; i < data.Length; i++)
            {
                Assert.AreEqual(resultQuad.Clusters[i], resultLogLin.Clusters[i]);
                Assert.AreEqual(resultQuad.Clusters[i], resultLinear.Clusters[i]);
            }

            // test cluster centers
            for (int k = 0; k < resultQuad.Centers.Length; k++)
            {
                Assert.AreEqual(resultQuad.Centers[k], resultLogLin.Centers[k]);
                Assert.AreEqual(resultQuad.Centers[k], resultLinear.Centers[k]);
            }

            Assert.AreEqual(data.Length, resultLogLin.Sizes.Sum());
            Assert.AreEqual(data.Length, resultLinear.Sizes.Sum());
        }


        [TestCase(new double[] { 0, 1, 0, 1, 1, 1 })]
        [TestCase(new double[] { 1, 0.1, 0, 2, 0.01, 3, 0.0001, 0.000001 })]
        [TestCase(new double[] { 1, 1, 2, 1, 3, 1, 4, 1, 6, 1, 7.1, 1, 7.7, 1, 7.9, 1, 8.1, 1, 9.1, 1, 10.2, 1, 10.4, 1 })]
        [TestCase(new double[] { 1, 0.1, 1.1, 0.1, 2, 0.1, 2.1, 0.05, 3.1, 0.2, 3, 0.1, 3.05, 0.1, 6, 0.1, 7, 0.2 })]
        public void WeightedSingleClusterTest(double[] input)
        {
            number[] data = input.Where((x, i) => i % 2 == 0).Select(x => (number)x).ToArray();
            number[] weight = input.Where((x, i) => i % 2 == 1).Select(x => (number)x).ToArray();

            var result = Main.CKMeans1D(data, weight, 1, 1, Method.Linear);

            number w_sum = weight.Sum();
            number mean = data.Select((x, i) => (x * weight[i])).Sum();

            mean /= w_sum;

            Assert.AreEqual(data.Length, result.Clusters.Length);
            Assert.AreEqual(w_sum, result.Sizes[0]);

            Assert.AreEqual(1, result.Centers.Length);
            Assert.AreEqual(mean, result.Centers[0]);
        }

        [TestCase(new double[] { -1, 1, +0, 2, +1, 1 })]
        [TestCase(new double[] { -2, 1, -1, 1, +0, 1, +1, 2, +2, 2})]
        [TestCase(new double[] { 1, 0.01, 1, 0.02, 2, 0.01, 2, 0.02, 3, 0.01, 3, 0.02})]
        [TestCase(new double[] { 1, 0.1, 1.1, 0.2, 2, 0.3, 2.1, 0.4, 3, 0.5, 3.1, 0.6 })]
        public void WeightedThreeClustersTest(double[] input)
        {
            number[] data = input.Where((x, i) => i % 2 == 0).Select(x => (number)x).ToArray();
            number[] weight = input.Where((x, i) => i % 2 == 1).Select(x => (number)x).ToArray();

            var result = Main.CKMeans1D(data, weight, 3, 3, Method.Linear);

            Assert.AreEqual(data.Length, result.Clusters.Length);
            Assert.AreEqual(3, result.Centers.Length);
            Assert.AreEqual(weight.Sum(), result.Sizes.Sum());
        }

        [TestCase(new double[] { 1, 0.1, 2, 0.1, 1.5, 0.1, 2.1, 0.12, 2.4, 0.5, 2.6, 0.1, 2.7, 0.5, 0.6, 0.7})]
        [TestCase(new double[] {-1, 0.1, 2, 0.2, -3, 0.12, 4, 0.22, -5, 0.27, 6, 0.12, -7, 0.17, 8, 0.28, -9, 0.34, 10, 0.93, -11, 0.643, 12, 0.93, -13, 0.1 })]
        public void WeightedVariableClustersTest(double[] input)
        {
            number[] data = input.Where((x, i) => i % 2 == 0).Select(x => (number)x).ToArray();
            number[] weight = input.Where((x, i) => i % 2 == 1).Select(x => (number)x).ToArray();

            number w_sum = weight.Sum();

            var resultQuad = Main.CKMeans1D(data, weight, 1, 10, Method.Quadratic);
            var resultLinear = Main.CKMeans1D(data, weight, 1, 10, Method.Linear);
            var resultLogLin = Main.CKMeans1D(data, weight, 1, 10, Method.LogLinear);

            Assert.AreEqual(resultQuad.Centers.Length, resultLogLin.Centers.Length);
            Assert.AreEqual(resultQuad.Centers.Length, resultLinear.Centers.Length);

            Assert.AreEqual(resultQuad.BIC.Length, resultLogLin.BIC.Length);
            Assert.AreEqual(resultQuad.BIC.Length, resultLinear.BIC.Length);

            // test cluster indexes
            for (int i = 0; i < data.Length; i++)
            {
                Assert.AreEqual(resultQuad.Clusters[i], resultLogLin.Clusters[i]);
                Assert.AreEqual(resultQuad.Clusters[i], resultLinear.Clusters[i]);
            }

            // test cluster centers
            for (int k = 0; k < resultQuad.Centers.Length; k++)
            {
                Assert.AreEqual(resultQuad.Centers[k], resultLogLin.Centers[k]);
                Assert.AreEqual(resultQuad.Centers[k], resultLinear.Centers[k]);
            }

            Assert.AreEqual(w_sum, resultLogLin.Sizes.Sum());
            Assert.AreEqual(w_sum, resultLinear.Sizes.Sum());
        }

    }
}
