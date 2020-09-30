using System;
using System.Collections.Generic;
using System.Linq;

namespace Lab1kNN
{
    public class Executor
    {
        private const int dataSetSize = 488;
        private double[,] ConfusionMatrix;
        private double confusionMatrixSum;
        private string metric;
        private string kernel;
        private bool windowTypeFixed;
        private double windowSize;
        private readonly int cmsize = 0;

        public Executor(string metric, string kernel, bool windowTypeFixed, double windowSize)
        {
            cmsize = (int) Program.classes.Max() + 1;
            ConfusionMatrix = new double[cmsize, cmsize];
            confusionMatrixSum = 0;
            this.windowSize = windowSize;
            this.metric = metric;
            this.kernel = kernel;
            this.windowTypeFixed = windowTypeFixed;
        }
        void FillConfusionMatrixNaive()
        {
            List<List<double>> dataWithout = new List<List<double>>(Program.data);
            for (int i = 0; i < Program.data.Count; i++)
            {
                int realClass = (int)Program.data[i].Last();
                dataWithout.Remove(Program.data[i]);
                int predictedClass = NaiveConverting.RegressionToClassification(Execute(dataWithout, Program.data[i]));
                dataWithout.Add(Program.data[i]);

                ConfusionMatrix[realClass, predictedClass]++;
                confusionMatrixSum++;
            }
        }

        void FillConfusionMatrixOneHot()
        {
            List<List<double>> dataWithout = new List<List<double>>(Program.data.AsReadOnly());
            foreach (var dataString in Program.data)
            {
                dataWithout.Remove(dataString);
                int realClass = (int)dataString.Last();
                //Console.WriteLine(realClass);
                int classInd = 0;
                double predictData = 0.0;
                int predictedClass = 0;
                foreach (var binaryClassification in OneHotConverting.GetBinaryClassificator(dataWithout.AsReadOnly(), out List<double> classList))
                {
                    List<List<double>> dataToBeSent = new List<List<double>>();
                    for (int i = 0; i < dataWithout.Count; i++)
                    {
                        dataToBeSent.Add(new List<double>());
                        for (int j = 0; j < dataWithout[i].Count - 1; j++)
                        {
                            dataToBeSent.Last().Add(dataWithout[i][j]);
                        }
                        dataToBeSent.Last().Add(binaryClassification[i]);
                    }

                    double predict = Execute(dataToBeSent, dataString);
                    if (predict >= predictData)
                    {
                        predictData = predict;
                        predictedClass = (int)classList[classInd];
                    }

                    classInd++;
                }

                confusionMatrixSum++;
                ConfusionMatrix[realClass, predictedClass]++;
                dataWithout.Add(dataString);
            }
        }

        private double Execute(IReadOnlyList<List<double>> data, List<double> query)
        {
            List<double> dists = new List<double>();
            List<double> distsToBeOrdered = new List<double>();

            foreach (var dataString in data)
            {
                dists.Add(Metrics.MetricFuncs[metric].Invoke(dataString, query));
                if (!windowTypeFixed)
                {
                    distsToBeOrdered.Add(dists.Last());
                }
            }

            double _windowSize = 0;
            if (!windowTypeFixed)
            {
                distsToBeOrdered.Sort();
                _windowSize = distsToBeOrdered[distsToBeOrdered.Count - (int)windowSize];
            }

            double resultky = 0, resultk = 0;
            bool sameWithQuery = false;

            for (int i = 0; i < data.Count; i++)
            {
                double ker;

                if (_windowSize == 0 && dists[i] != 0)
                {
                    continue;
                }
                sameWithQuery = _windowSize == 0 && dists[i] == 0;
                double u = sameWithQuery ? 0 : dists[i] / windowSize;

                ker = Kernels.KernelFuncs[kernel].Invoke(u);

                resultky += ker * data[i].Last();
                resultk += ker;
            }

            if ((windowSize == 0 && !sameWithQuery) || resultk == 0)
            {
                double res = 0.0;
                foreach (var dataString in data)
                {
                    res += dataString.Last();
                }
                return res / data.Count;
            }
            return resultky / resultk;
        }

        public void ComputeFMeasure(bool isNaive, out double micro, out double macro)
        {
            if (isNaive) FillConfusionMatrixNaive();
            else FillConfusionMatrixOneHot();
            double[] c = new double[cmsize], p = new double[cmsize], t = new double[cmsize];

            for (int i = 0; i < cmsize; i++)
            {
                for (int j = 0; j < cmsize; j++)
                {
                    c[i] += ConfusionMatrix[i, j];
                    p[j] += ConfusionMatrix[i, j];
                    if (i == j)
                    {
                        t[i] = ConfusionMatrix[i, j];
                    }
                }
            }

            double rec = confusionMatrixSum != 0 ? t.Aggregate((x, y) => x + y) / confusionMatrixSum : 0;
            double prec = 0;
            for (int i = 0; i < cmsize; i++)
            {
                prec += p[i] == 0 ? 0 : t[i] * c[i] / p[i];
            }
            prec = confusionMatrixSum == 0 ? 0 : prec / confusionMatrixSum;

            macro = (rec + prec) == 0 ? 0 : 2.0 * rec * prec / (rec + prec);

            micro = 0.0;

            for (int i = 0; i < cmsize; i++)
            {
                prec = p[i] == 0 ? 0 : t[i] / p[i];
                rec = c[i] == 0 ? 0 : t[i] / c[i];
                micro += (rec + prec) == 0 ? 0 : (2 * rec * prec / (prec + rec)) * c[i];
            }
            micro = confusionMatrixSum == 0 ? 0 : micro / confusionMatrixSum;
        }
    }
}