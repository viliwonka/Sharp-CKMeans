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

namespace Sharp.CKMeans
{
    public enum Method
    {
        Linear,
        LogLinear,
        Quadratic
    }

    public enum DissimilarityType
    {
        L1, L2, L2Y
    }

}
