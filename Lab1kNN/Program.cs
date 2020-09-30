using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;

namespace Lab1kNN
{
    class Program
    {
        public static List<List<Double>> data;
        public static HashSet<double> classes;

        static void Main(string[] args)
        {
            data = new List<List<double>>();
            classes = new HashSet<double>();
            ReadData(out var min, out var max);
            NormalizeData(min, max);
            Brutforce.OneHotConvertingBrutforce();
            Brutforce.NaiveConvertingBrutforce();
            
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
                        data.Add(new List<double>());
                    }
                    else
                    {
                        continue;
                    }

                    foreach (var value in values)
                    {
                        Double.TryParse(value, out var val);
                        data.Last().Add(val);
                        min = Math.Min(min, val);
                        max = Math.Max(max, val);
                        //classes.Add(val);
                    }
                    classes.Add(data.Last().Last());
                }
            }
        }

        private static void NormalizeData(double min, double max)
        {
            foreach (var dataString in data)
            {
                for (int i = 0; i < dataString.Count - 1; i++)
                {
                    dataString[i] -= min;
                }
            }
        }
    }
}
