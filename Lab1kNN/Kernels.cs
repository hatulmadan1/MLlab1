using System;
using System.Collections.Generic;
using System.Text;

namespace Lab1kNN
{
    public static class Kernels
    {
        public static Dictionary<string, Func<double, double>> KernelFuncs =
            new Dictionary<string, Func<double, double>>()
            {
                {"Epanechnikov", Epanechnikov},
                {"Uniform", Uniform},
                {"Triangular", Triangular},
                
                {"Quartic", Quartic},
                {"Triweight", Triweight},
                {"Tricube", Tricube},
                {"Gaussian", Gaussian},
                {"Cosine", Cosine},
                {"Logistic", Logistic},
                {"Sigmoid", Sigmoid}
            };

        static double Uniform(double u)
        {
            return Math.Abs(u) < 1 ? 1.0 / 2.0 : 0.0;
        }

        static double Triangular(double u)
        {
            return Math.Abs(u) < 1 ? 1 - Math.Abs(u) : 0.0;
        }

        static double Epanechnikov(double u)
        {
            return Math.Abs(u) < 1 ? (1 - Math.Pow(u, 2)) * 3.0 / 4.0 : 0.0;
        }

        static double Quartic(double u)
        {
            return Math.Abs(u) < 1 ? Math.Pow((1 - Math.Pow(u, 2)), 2) * 15.0 / 16.0 : 0.0;
        }

        static double Triweight(double u)
        {
            return Math.Abs(u) < 1 ? Math.Pow((1 - Math.Pow(u, 2)), 3) * 35.0 / 32.0 : 0.0;
        }

        static double Tricube(double u)
        {
            return Math.Abs(u) < 1 ? Math.Pow((1 - Math.Pow(Math.Abs(u), 3)), 3) * 70.0 / 81.0 : 0.0;
        }

        static double Gaussian(double u)
        {
            return Math.Exp(-Math.Pow(u, 2) / 2.0) / Math.Sqrt(2 * Math.PI);
        }

        static double Cosine(double u)
        {
            return Math.Abs(u) < 1 ? Math.Cos(u * (Math.PI / 2.0)) * (Math.PI / 4.0) : 0.0;
        }

        static double Logistic(double u)
        {
            return 1.0 / (Math.Exp(u) + 2.0 + Math.Exp(-u));
        }

        static double Sigmoid(double u)
        {
            return 2.0 / (Math.PI * (Math.Exp(u) + Math.Exp(-u)));
        }
	}
}
