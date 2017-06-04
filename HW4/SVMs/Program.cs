using Accord.MachineLearning;
using BiasAndVarianceOfID3;
using EnsembleMethods;
using libsvm;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMs
{
    class Program
    {
        private const string TrainingCsv = @"c:\Data\training.csv";

        private const string TestingCsv = @"c:\Data\testing.csv";

        private const int BiasVarianceNumOfSamples = 30;

        private const int ClassIndex = 24;

        private static List<int> _continuesIndexes = new List<int>
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

        // Calculated with external tool
        private static Dictionary<int, int> _indexElbowMap = new Dictionary<int, int>
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
            List<double[]> continuousTrainData = DataWrangler.LoadContinuousDataAsync(TrainingCsv).Result;
            List<double[]> continuousTestData = DataWrangler.LoadContinuousDataAsync(TestingCsv).Result;

            // Print continuous columns for calculating elbows in external tool(https://bl.ocks.org/rpgove/0060ff3b656618e9136b)
            foreach (int i in _continuesIndexes)
            {
                using (StreamWriter sw = new StreamWriter($"{i}.txt"))
                {
                    sw.WriteLine(string.Join(",", continuousTrainData.Select(array => array[i])));
                }
            }

            // Convert continuous to discrete
            Dictionary<int, GaussianClusterCollection> indexClusterMapping = DataWrangler.GetIndexClustersMap(continuousTrainData, _indexElbowMap);
            List<int[]> discreteTrainData = DataWrangler.ConvertContinuesToDiscrete(continuousTrainData, indexClusterMapping);
            List<int[]> discreteTestData = DataWrangler.ConvertContinuesToDiscrete(continuousTestData, indexClusterMapping);

            List<List<int[]>> samples = Sampler.SampleData(discreteTrainData, BiasVarianceNumOfSamples);

            var problem = ProblemHelper.ReadProblem(discreteTrainData.Select(arr =>
            {
                // Move class to front as it is expected by libsvm.
                int temp = arr[0];
                arr[0] = arr[ClassIndex];
                arr[ClassIndex] = temp;
                return arr.Select(i => (double)i).ToList();
            }).ToList());

            var test = ProblemHelper.ReadProblem(discreteTestData.Select(arr =>
            {
                // Move class to front as it is expected by libsvm.
                int temp = arr[0];
                arr[0] = arr[ClassIndex];
                arr[ClassIndex] = temp;
                return arr.Select(i => (double)i).ToList();
            }).ToList());

            double accuracy = 0;
            var svm = new C_SVC(problem, KernelHelper.SigmoidKernel(1, 1), 1);
            for (int i = 0; i < test.l; i++)
            {
                var x = test.x[i];
                var y = test.y[i];
                var predictedY = svm.Predict(x);
                if (y == predictedY)
                {
                    accuracy++;
                }
            }

            accuracy /= test.l;
            Console.WriteLine($"{accuracy}");

            Console.WriteLine("Press ENTER to continue...");
            Console.WriteLine();
        }
    }
}
