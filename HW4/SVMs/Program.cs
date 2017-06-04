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

        private const int ClassIndex = 23;

        private static List<int> _continuousIndexes = new List<int>
            {
                0, // Amount of the given credit (NT dollar)
                4, // Age
                11, // 12 - 17 Amount of bill statement (NT dollar)
                12,
                13,
                14,
                15,
                16,
                17, // 18 - 23 Amount of previous payment (NT dollar)
                18,
                19,
                20,
                21,
                22
            };

        // Calculated with external tool
        private static Dictionary<int, int> _indexElbowMap = new Dictionary<int, int>
        {
            { 0, 3 },
            { 4, 3 },
            { 11, 3 },
            { 12, 3 },
            { 13, 4 },
            { 14, 3 },
            { 15, 5 }, // 3, 4 or 5 seem to be good elbows
            { 16, 7 }, // 3 and 7 are elbows
            { 17, 4 }, // This is definetly 4
            { 18, 2 }, // This is definetly 2
            { 19, 3 },
            { 20, 3 }, // This is definetly 4
            { 21, 4 }, // Too curve between 2 and 5 :/
            { 22, 4 }, // 4 is sharp, but there is a straight line between 2 and 4, so 2 could be also good.
        };

        private static HashSet<int> _indexToIgnore = new HashSet<int>
        {
            0 // ignore the id column
        };

        static void Main(string[] args)
        {
            List<double[]> continuousTrainData = DataWrangler.LoadContinuousDataAsync(TrainingCsv, _indexToIgnore).Result;
            List<double[]> continuousTestData = DataWrangler.LoadContinuousDataAsync(TestingCsv, _indexToIgnore).Result;

            // Print continuous columns for calculating elbows in external tool(https://bl.ocks.org/rpgove/0060ff3b656618e9136b)
            foreach (int i in _continuousIndexes)
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

            // defaults taken from documentation http://weka.sourceforge.net/doc.stable/weka/classifiers/functions/LibSVM.html
            double c = 1; // default C is 1
            double gamma = 1.0 / problem.l; // default gamma is 1/k
            double r = 0; // default coef0 is 0
            int degree = 3; // default degree is 3

            Dictionary<string, Kernel> nameKernelMap = new Dictionary<string, Kernel>(StringComparer.OrdinalIgnoreCase)
            {
                { "Linear", KernelHelper.LinearKernel() },
                { "Polynomial", KernelHelper.PolynomialKernel(degree, gamma, r) },
                { "Radial", KernelHelper.RadialBasisFunctionKernel(gamma) },
                { "Sigmoid", KernelHelper.SigmoidKernel(gamma, r) },
            };

            //Parallel.ForEach(nameKernelMap.Keys, (kernelName) =>
            foreach(string kernelName in nameKernelMap.Keys)
            {
                Console.WriteLine($"{kernelName}: {GetAccuracy(problem, test, nameKernelMap[kernelName], c)}");
            };

            Console.WriteLine("Press ENTER to continue...");
            Console.ReadLine();
        }

        private static double GetAccuracy(svm_problem trainData, svm_problem testData, Kernel kernel, double c)
        {
            double accuracy = 0;
            var svm = new C_SVC(trainData, kernel, c);
            for (int i = 0; i < testData.l; i++)
            {
                var x = testData.x[i];
                var y = testData.y[i];
                var predictedY = svm.Predict(x);
                if (y == predictedY)
                {
                    accuracy++;
                }
            }

            return accuracy / testData.l;
        }
    }
}
