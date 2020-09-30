using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Lab1kNN
{
    public static class Brutforce
    {
        private static double maxMicro = 0;
        private static double maxMacro = 0;

        public static void NaiveConvertingBrutforce()
        {
            double maxDist = 0.0;
            foreach (var metric in Metrics.MetricFuncs.Keys)
            {
                for (int i = 0; i < Program.data.Count; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        maxDist = Math.Max(maxDist, Metrics.MetricFuncs[metric].Invoke(Program.data[i], Program.data[j]));
                    }
                }
                Console.WriteLine(maxDist);
                foreach (var kernel in Kernels.KernelFuncs.Keys)
                {
                    for (int i = 1; i <= Program.data.Count - 1; i++)
                    {
                        doLogic(kernel, metric, i, false, true);
                    }
                    for (double i = maxDist + 0.5; i > 0; i -= 0.5)
                    {
                        doLogic(kernel, metric, i, true, true);
                    }
                }
            }

            Console.WriteLine("Naive completed");
        }

        public static void OneHotConvertingBrutforce()
        {
            double maxDist = 0.0;
            foreach (var metric in Metrics.MetricFuncs.Keys)
            {
                for (int i = 0; i < Program.data.Count; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        maxDist = Math.Max(maxDist, Metrics.MetricFuncs[metric].Invoke(Program.data[i], Program.data[j]));
                    }
                }
                Console.WriteLine(maxDist);
                foreach (var kernel in Kernels.KernelFuncs.Keys)
                {
                    for (int i = 1; i <= Program.data.Count - 1; i++)
                    {
                        doLogic(kernel, metric, i, false, false);
                    }
                    for (double i = maxDist + 0.5; i > 0; i -= 0.5)
                    {
                        doLogic(kernel, metric, i, true, false);
                    }
                }
            }

            Console.WriteLine("OneHots completed");
        }

        private static void doLogic(string kernel, string metric, double i, bool winTypeFixed, bool isNaive)
        {
            var executor = new Executor(metric, kernel, winTypeFixed, i);
            executor.ComputeFMeasure(isNaive, out double micro, out double macro);
            if (macro > maxMacro || micro > maxMicro)
            {
                Console.WriteLine("{0} {1} is fixed: {5} {2}\nMicro: {3}\n Macro: {4}", metric, kernel, i, micro, macro, winTypeFixed);
            }
            if (macro > maxMacro)
            {
                maxMacro = macro;
            } 
            if (micro > maxMicro)
            {
                maxMicro = micro;
            }
        }
    }
}
