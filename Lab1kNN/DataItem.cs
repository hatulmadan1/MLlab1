using System;
using System.Collections.Generic;
using System.Text;

namespace Lab1kNN
{
    public class DataItem
    {
        public static List<double> OneHotClasses;
        public List<double> Features;
        public double RealClass;
        public List<double> OneHotClassification;

        public DataItem()
        {
            Features = new List<double>();
            OneHotClassification = new List<double>();
        }
    }
}
