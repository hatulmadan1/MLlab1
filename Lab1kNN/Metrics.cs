using System;
using System.Collections.Generic;
using System.Text;

namespace Lab1kNN
{
    public static class Metrics
    {
        public static Dictionary<string, Func<IReadOnlyList<double>, IReadOnlyList<double>, double>> MetricFuncs = 
            new Dictionary<string, Func<IReadOnlyList<double>, IReadOnlyList<double>, double>>()
            {
                {"Manhattan", Manhattan},
                {"Euclidean", Euclidean},
                {"Chebyshev", Chebyshev}
            };

        static double Manhattan(IReadOnlyList<double> x, IReadOnlyList<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count; i++)
            {
                ans += Math.Abs(x[i] - y[i]);
            }
            return ans;
        }

        static double Euclidean(IReadOnlyList<double> x, IReadOnlyList<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count; i++)
            {
                ans += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return Math.Sqrt(ans);
        }

        static double Chebyshev(IReadOnlyList<double> x, IReadOnlyList<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count; i++)
            {
                ans = Math.Max(ans, Math.Abs(x[i] - y[i]));
            }
            return Math.Sqrt(ans);
        }
    }
}
