using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;

namespace Lab1kNN
{
    public static class OneHotConverting
    {
        public static IReadOnlyList<List<double>> GetBinaryClassificator(IReadOnlyList<List<double>> data, out List<double> classList)
        {
            List<List<double>> BinaryClassificator = new List<List<double>>();
            for (int i = 0; i < Program.classes.Count; i++)
            {
                BinaryClassificator.Add(new List<double>());
            }
            int classNumber = 0;
            classList = new List<double>();
            foreach (var currentClass in Program.classes)
            {
                foreach (var dataString in data)
                {
                    BinaryClassificator[classNumber].Add(dataString.Last() == currentClass ? 1 : 0);
                }
                classNumber++;
                classList.Add(currentClass);
            }

            return BinaryClassificator;
        }
    }
}
