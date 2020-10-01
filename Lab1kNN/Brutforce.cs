using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Lab1kNN
{
    public static class Brutforce
    {
        private static double maxMicro = 0;
        private static double maxMacro = 0;
        public static Dictionary<string, double> maxMetricValue;
        public static Mutex mutex = new Mutex();

        public static void BrutforceBoth()
        {
            maxMetricValue = new Dictionary<string, double>()
            {
                {"Manhattan", 0.0},
                {"Euclidean", 0.0},
                {"Chebyshev", 0.0}
            };

            foreach (var metric in Metrics.MetricFuncs.Keys)
            {
                for (int i = 0; i < Program.data.Count; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        maxMetricValue[metric] = Math.Max(maxMetricValue[metric],
                            Metrics.MetricFuncs[metric].Invoke(Program.data[i].Features, Program.data[j].Features));
                    }
                }
            }

            Parallel.ForEach(Kernels.KernelFuncs.Keys, KernelBrutforceInnerLogic);
        }

        private static void KernelBrutforceInnerLogic(string kernel)
        {
            foreach (var metric in Metrics.MetricFuncs.Keys)
            {
                List<Task> tasks = new List<Task>();
                for (int i = 1; i <= Program.data.Count - 1; i++)
                {
                    var i1 = i;
                    tasks.Add(new Task(() => DoLogic(kernel, metric, i1, false, true)));
                    var i2 = i;
                    tasks.Add(new Task(() => DoLogic(kernel, metric, i2, false, false)));
                }

                for (double i = maxMetricValue[metric] + 0.5; i > 0; i -= 0.5)
                {
                    var i1 = i;
                    tasks.Add(new Task(() => DoLogic(kernel, metric, i1, true, true)));
                    var i2 = i;
                    tasks.Add(new Task(() => DoLogic(kernel, metric, i2, true, false)));
                }

                Parallel.ForEach(tasks, StartTasksAsync);
            }
        }

        private static void StartTasksAsync(Task task)
        {
            task.Start();
        }

        private static void DoLogic(string kernel, string metric, double i, bool winTypeFixed, bool isNaive)
        {
            var executor = new Executor(metric, kernel, winTypeFixed, i, isNaive);
            executor.ComputeFMeasure(out double micro, out double macro);

            mutex.WaitOne();
            if (kernel.Equals("Triweight") && isNaive && winTypeFixed && metric.Equals("Manhattan"))
            {
                Program.Naive.Add(i, macro);
            }
            if (kernel.Equals("Triweight") && !isNaive && winTypeFixed && metric.Equals("Manhattan"))
            {
                Program.OneHot.Add(i, macro);
            }
            if (macro > maxMacro || micro > maxMicro)
            {
                Console.WriteLine("{0} {1} {5} {2} {6}\nMicro: {3}\nMacro: {4}", metric, kernel, i, micro, macro, (winTypeFixed ? "fixed" : "variable"), (isNaive ? "naive" : "onehot"));
            }
            if (macro > maxMacro)
            {
                maxMacro = macro;
            } 
            if (micro > maxMicro)
            {
                maxMicro = micro;
            }
            mutex.ReleaseMutex();
        }
    }
}
