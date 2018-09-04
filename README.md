# Sharp-CKMeans
CKMeans ported to C# (from C++). Used in R language.

## What is this?

This is library for 1-dimensional/univariate optimal clustering.

It's special because it's deterministic - it doesn't use random numbers & iteration for calculation, which means it always returns same result for same input, it's precise (always most optimal result) and very fast (uses dynamic programming).

## Why use this?

Works natively in C#. Made so that it works with `decimal` data type.

You can use this instead of Gaussian Mixture Model (depends on what you wish to do though).

It's slow with `decimal` though (takes around 3ms per clustering).

As an alternative, you can change `using number = System.Decimal;` to `using number = System.Double;` on top of every file, 
this way it will work with doubles (or floats if you wish) and will be faster, although less precise.


### How to use

All needed functions are present at `Main.cs`.

What you have available is `Main.CKMeans1D(...)`, `Main.CKMedian1D(...)`, `Main.CKSegs1D(...)`.

For all functions, arguments are `(number[] x, number[] y, int Kmin, int Kmax, Method method = Method.Linear)`.

* `x` are points / positions
* `y` are weights
* `Kmin` is minimal number of clusters
* `Kmax` is maximal number of clusters
* `method` is what method to use for calculation (`Linear`, `LogLinear`, `Quadratic`). `LogLinear` is fastest.

All functions return `CKResult` object, which contains:

* int[] Clusters is an array of cluster IDs for each point in x,
* number[] Centers is an array of centers for each cluster
* number[] Withinss is an array of within-cluster sum of squares for each cluster
* number[] Sizes is an array of (weighted) sizes of each cluster
* double[] BIC is Bayesian Information Criterion

[Tutorial "Optimal univariate clustering" (for R)](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/vignettes/Ckmeans.1d.dp.html)

### Log-Likelihood

Logarithm/Exponent (log likelihood) calculation is done with doubles (64-bit in C#), originaly it was done with long-double (80-bit in C++).

I added tests for CKMeans (not CKMedian and others!) and it works as expected, but if you experience irregularities, you need to convert log-likelihood calculation part of code to higher precision. I tried with `decimal` but it threw `out of range` errors! 

It's short part of code, it's present in `NonWeighted.cs` at line 150 and `Weighted.cs` at line 145.

### Original sources

[Original page for this library](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html)

[Original Github mirror](https://github.com/cran/Ckmeans.1d.dp)