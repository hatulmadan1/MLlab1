using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace Lab1kNN
{
    public static class OneHotConverting
    {
        public static void FillBinaryClassification()
        {
            Parallel.ForEach(Program.data, FillDataItem);
        }

        private static void FillDataItem(DataItem dataItem)
        {
            foreach (var classValue in DataItem.OneHotClasses)
            {
                dataItem.OneHotClassification.Add(classValue == dataItem.RealClass ? 1 : 0);
            }
        }
    }
}
