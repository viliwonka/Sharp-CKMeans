# Sharp-CKMeans
CKMeans ported to C# (from C++)

## Why use this?
Works natively in C#. Made so that it work with `decimal` data type.
It's slow with `decimal` data type though (takes ~3ms per clustering).

You can change `using number = System.Decimal;` to `using number = System.Double;` on top of every file, 
this way it will work with doubles (or floats if you wish) and will be faster, although less precise. 

Logarithm/Exponent (log likelihood) calculation is done with doubles (64-bit in C#), originaly it was done with long-double (80-bit in C++).

I added tests for CKMeans and it works as expected, but if you experience irregularities, you need to convert log-likelihood calculation part of code to higher precision. I tried with `decimal` but it threw `out of range` errors!

