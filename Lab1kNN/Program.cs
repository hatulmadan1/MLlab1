using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using Microsoft.VisualBasic.CompilerServices;

namespace Lab1kNN
{
    class Program
    {
        public static List<DataItem> data;
        public static HashSet<double> classes;
        public static Dictionary<double, double> Naive;
        public static Dictionary<double, double> OneHot;

        static void Main(string[] args)
        {
            DateTime start = DateTime.Now;
            data = new List<DataItem>();
            classes = new HashSet<double>();
            Naive = new Dictionary<double, double>();
            OneHot = new Dictionary<double, double>();

            ReadData(out var min, out var max);
            DataItem.OneHotClasses = classes.ToList();
            NormalizeData(min, max);
            OneHotConverting.FillBinaryClassification(); 
            Brutforce.BrutforceBoth();
            Console.WriteLine(DateTime.Now - start);

            Console.WriteLine("Naive");
            foreach (var v in Naive)
            {
                Console.WriteLine("({0}; {1})", v.Key, v.Value);
            }

            Console.WriteLine("OneHot");
            foreach (var v in OneHot)
            {
                Console.WriteLine("({0}; {1})", v.Key, v.Value);
            }

            ExportData();
        }

        private static void ReadData(out double min, out double max)
        {
            max = 0.0;
            min = Double.MaxValue;
            using (var reader = new StreamReader(@"..\..\..\ESL.csv"))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(',');
                    if (Double.TryParse(values?[0], out _))
                    {
                        data.Add(new DataItem());
                    }
                    else
                    {
                        continue;
                    }

                    for (int i = 0; i < values.Length; i++)
                    {
                        Double.TryParse(values[i], out var val);
                        if (i != values.Length - 1)
                        {
                            data.Last().Features.Add(val);
                            min = Math.Min(min, val);
                            max = Math.Max(max, val);
                        }
                        else
                        {
                            data.Last().RealClass = (val);
                            classes.Add(val);
                        }
                    }
                    
                }
            }
        }

        private static void NormalizeData(double min, double max)
        {
            foreach (var dataString in data)
            {
                for (int i = 0; i < dataString.Features.Count - 1; i++)
                {
                    dataString.Features[i] -= min;
                }
            }
        }

        private static void ExportData()
        {
            using (StreamWriter w = File.AppendText("LabResult.csv"))
            {
                foreach (var dist in OneHot.Keys)
                {
                    w.Write("{0},{1},{2}\n", dist, Naive[dist], OneHot[dist]);
                }
            }
        }
    }
}
