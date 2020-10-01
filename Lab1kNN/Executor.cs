using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Lab1kNN
{
    public class Executor
    {
        private double[,] ConfusionMatrix;
        private string metric;
        private string kernel;
        private bool windowTypeFixed;
        private double windowSize;
        private readonly int cmsize = 0;
        private bool isNaive;
        public Mutex mutex = new Mutex();

        public Executor(string metric, string kernel, bool windowTypeFixed, double windowSize, bool isNaive)
        {
            cmsize = (int) Program.classes.Max() + 1;
            ConfusionMatrix = new double[cmsize, cmsize];
            this.windowSize = windowSize;
            this.metric = metric;
            this.kernel = kernel;
            this.windowTypeFixed = windowTypeFixed;
            this.isNaive = isNaive;
        }
        public void ComputeFMeasure(out double micro, out double macro)
        {
            FillConfusionMatrix();
            double confusionMatrixSum = Program.data.Count;
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
        private void FillConfusionMatrix()
        {
            Parallel.For(0, Program.data.Count, ComputePredict);
        }

        private void ComputePredict(int i)
        {
                int realClass = (int)Program.data[i].RealClass;
                int predictedClass = 0;
                if (isNaive)
                {
                    predictedClass = NaiveConverting.RegressionToClassification(Execute(i, -1));
                }
                else
                {
                    double predictData = 0.0;
                    for (int j = 0; j < DataItem.OneHotClasses.Count; j++)
                    {
                        double predict = Execute(i, j);
                        if (predict >= predictData)
                        {
                            predictData = predict;
                            predictedClass = (int)DataItem.OneHotClasses[j];
                        }
                    }
                }

                mutex.WaitOne();
                ConfusionMatrix[realClass, predictedClass]++;
                mutex.ReleaseMutex();
        }

        private double Execute(int queryStringId, int classId)
        {
            IReadOnlyList<double> query = Program.data[queryStringId].Features.AsReadOnly();
            List<double> dists = new List<double>();
            List<double> distsToBeOrdered = new List<double>();
            double classValue;

            for (int i = 0; i < Program.data.Count; i++)
            {
                dists.Add(Metrics.MetricFuncs[metric].Invoke(Program.data[i].Features, query));
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
            else
            {
                _windowSize = windowSize;
            }

            double resultky = 0, resultk = 0;
            bool sameWithQuery = false;

            for (int i = 0; i < Program.data.Count; i++)
            {
                if (i == queryStringId)
                {
                    continue;
                }
                
                if (_windowSize == 0 && dists[i] != 0)
                {
                    continue;
                }

                double ker;
                sameWithQuery = _windowSize == 0 && dists[i] == 0;
                double u = sameWithQuery ? 0 : dists[i] / windowSize;

                ker = Kernels.KernelFuncs[kernel].Invoke(u);
                classValue = classId == -1 ? Program.data[i].RealClass : Program.data[i].OneHotClassification[classId];

                resultky += ker * classValue;
                resultk += ker;
            }

            if ((windowSize == 0 && !sameWithQuery) || resultk == 0)
            {
                double res = 0.0;
                for (int i = 0; i < Program.data.Count; i++)
                {
                    if (i == queryStringId)
                    {
                        continue;
                    }
                    classValue = classId == -1 ? Program.data[i].RealClass : Program.data[i].OneHotClassification[classId];
                    res += classValue;
                }
                return res / (Program.data.Count - 1);
            }
            return resultky / resultk;
        }
    }
}