using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;

namespace Lab1kNN
{
    class Program
    {
        public static List<DataItem> data;
        public static HashSet<double> classes;

        static void Main(string[] args)
        {
            data = new List<DataItem>();
            classes = new HashSet<double>();
            ReadData(out var min, out var max);
            DataItem.OneHotClasses = classes.ToList();
            NormalizeData(min, max);
            OneHotConverting.FillBinaryClassification(); 
            Brutforce.BrutforceBoth();
        }

        private static void ReadData(out double min, out double max)
        {
            max = 0.0;
            min = Double.MaxValue;
            using (var reader = new StreamReader(@"..\..\..\..\..\ESL.csv"))
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
    }
}
