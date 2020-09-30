using System;
using System.Collections.Generic;
using System.Text;

namespace Lab1kNN
{
    public static class NaiveConverting
    {
        public static int RegressionToClassification(double x)
        {
            return (int)Math.Round(x);
        }
    }
}
