using System;
using System.Collections.Generic;
using System.Text;

namespace Lab1kNN
{
    public static class Metrics
    {
        public static Dictionary<string, Func<List<double>, List<double>, double>> MetricFuncs = 
            new Dictionary<string, Func<List<double>, List<double>, double>>()
            {
                {"Manhattan", Manhattan},
                {"Euclidean", Euclidean},
                {"Chebyshev", Chebyshev}
            };

        static double Manhattan(List<double> x, List<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count - 1; i++)
            {
                ans += Math.Abs(x[i] - y[i]);
            }
            return ans;
        }

        static double Euclidean(List<double> x, List<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count - 1; i++)
            {
                ans += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return Math.Sqrt(ans);
        }

        static double Chebyshev(List<double> x, List<double> y)
        {
            double ans = 0.0;
            for (int i = 0; i < x.Count - 1; i++)
            {
                ans = Math.Max(ans, Math.Abs(x[i] - y[i]));
            }
            return Math.Sqrt(ans);
        }
    }
}
