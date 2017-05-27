using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BiasAndVarianceOfID3
{
    class Program
    {
        private const string DataCsv = @"c:\Data\dataset_default.csv";

        // Calculated with external tool
        private Dictionary<int, int> _indexElbowMap = new Dictionary<int, int>
        {
            { 1, 3 },
            { 5, 3 },
            { 12, 3 },
            { 13, 3 },
            { 14, 4 },
            { 15, 3 },
            { 16, 5 }, // 3, 4 or 5 seem to be good elbows
            { 17, 7 }, // 3 and 7 are elbows
            { 18, 4 }, // This is definetly 4
            { 19, 2 }, // This is definetly 2
            { 20, 3 }, 
            { 21, 3 }, // This is definetly 4
            { 22, 4 }, // Too curve between 2 and 5 :/
            { 23, 4 }, // 4 is sharp, but there is a straight line between 2 and 4, so 2 could be also good.
        };

        static void Main(string[] args)
        {
            DataWrangler dataWrangler = DataWrangler.Load(DataCsv);

            // Print columns for calculating elbows in external tool
            // This is used to check the elbows in an external tool.
            List<int> continuesIndexes = new List<int>
            {
                1, // Amount of the given credit (NT dollar)
                5, // Age
                12, // 12 - 17 Amount of bill statement (NT dollar)
                13,
                14,
                15,
                16,
                17,
                18, // 18 - 23 Amount of previous payment (NT dollar)
                19,
                20,
                21,
                22,
                23
            };
            foreach (int i in continuesIndexes)
            {
                using (StreamWriter sw = new StreamWriter($"{i}.txt"))
                {
                    sw.WriteLine(string.Join(",", dataWrangler.Data.Select(array => array[i])));
                }
            }



            Console.WriteLine("Press ENTER to exit...");
            Console.ReadLine();
        }
    }
}
