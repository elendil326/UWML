using Accord.MachineLearning;
using EnsembleMethods;
using HW1;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BiasAndVarianceOfID3
{
    class Program
    {
        private const string TrainingCsv = @"c:\Data\training.csv";

        private const string TestingCsv = @"c:\Data\testing.csv";

        private const int BiasVarianceNumOfSamples = 30;

        private const int BaggerSamples = 30;

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
            //foreach (int i in _continuesIndexes)
            //{
            //    using (StreamWriter sw = new StreamWriter($"{i}.txt"))
            //    {
            //        sw.WriteLine(string.Join(",", continuousTrainData.Select(array => array[i])));
            //    }
            //}

            // Convert continuous to discrete
            Dictionary<int, GaussianClusterCollection> indexClusterMapping = DataWrangler.GetIndexClustersMap(continuousTrainData, _indexElbowMap);
            List<int[]> discreteTrainData = DataWrangler.ConvertContinuesToDiscrete(continuousTrainData, indexClusterMapping);
            List<int[]> discreteTestData = DataWrangler.ConvertContinuesToDiscrete(continuousTestData, indexClusterMapping);

            // Verify clustering
            //foreach (int continuousIndex in _continuousIndexes)
            //{
            //    HashSet<int> clustersTrain = new HashSet<int>();
            //    HashSet<int> clustersTest = new HashSet<int>();

            //    foreach (int[] instance in discreteTrainData)
            //    {
            //        clustersTrain.Add(instance[continuousIndex]);
            //    }
            //    foreach (int[] instance in discreteTestData)
            //    {
            //        clustersTest.Add(instance[continuousIndex]);
            //    }

            //    Console.WriteLine($"Clustering in index {continuousIndex} generated {clustersTrain.Count} unique clusters in train and {clustersTest.Count} unique clusters in test");
            //    Console.WriteLine($"Train: {string.Join(",", clustersTrain.OrderBy(i => i))}");
            //    Console.WriteLine($"Test: {string.Join(",", clustersTest.OrderBy(i => i))}");
            //    Console.WriteLine();
            //}

            List<List<int[]>> samples = Sampler.SampleData(discreteTrainData, BiasVarianceNumOfSamples);

            ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentDictionary<int, int>>> id3ClassifierDepthInstanceClassifierPredictionMappings = new ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentDictionary<int, int>>>();
            ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentDictionary<int, int>>> id3BaggerDepthInstanceClassifierPredictionMappings = new ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentDictionary<int, int>>>();

            ConcurrentDictionary<int, Tuple<double, double, double>> id3ClassifierResult = new ConcurrentDictionary<int, Tuple<double, double, double>>();
            ConcurrentDictionary<int, Tuple<double, double, double>> id3BaggerClassifierResult = new ConcurrentDictionary<int, Tuple<double, double, double>>();
            // From 0 to 10 maxDepth, calculate the bias and variance of the classifiers in parallel.
            Parallel.For(0, 11, (maxDepth) =>
            //for (int maxDepth = 0; maxDepth < 2; maxDepth++)
            {
                ConcurrentDictionary<int, ConcurrentDictionary<int, int>> id3BaggerInstanceClassifierPredictionMappings = id3BaggerDepthInstanceClassifierPredictionMappings.GetOrAdd(maxDepth, new ConcurrentDictionary<int, ConcurrentDictionary<int, int>>());
                ConcurrentDictionary<int, ConcurrentDictionary<int, int>> id3ClassifierInstanceClassifierPredictionMappings = id3ClassifierDepthInstanceClassifierPredictionMappings.GetOrAdd(maxDepth, new ConcurrentDictionary<int, ConcurrentDictionary<int, int>>());

                for (int i = 0; i < BiasVarianceNumOfSamples; i++)
                //Parallel.For(0, BiasVarianceNumOfSamples, i =>
                {
                    // Initialize and train the classifiers in parallel
                    Id3Bagger bagger = new Id3Bagger(BaggerSamples);
                    bagger.Train(samples[i], ClassIndex, 0, maxDepth);

                    // Store predictions on test data
                    //Parallel.For(0, discreteTestData.Count, (j) =>
                    for (int j = 0; j < discreteTestData.Count; j++)
                    {
                        ConcurrentDictionary<int, int> classifierPredictionMapping = id3BaggerInstanceClassifierPredictionMappings.GetOrAdd(j, new ConcurrentDictionary<int, int>());
                        classifierPredictionMapping.GetOrAdd(i, bagger.GetClass(discreteTestData[j]));
                    };

                    // Free the memory
                    bagger = null;

                    // Initialize and train the classifiers in parallel
                    Id3Classifier tree = new Id3Classifier(samples[i], ClassIndex, 0, maxDepth);

                    // Store predictions on test data
                    //Parallel.For(0, discreteTestData.Count, (j) =>
                    for (int j = 0; j < discreteTestData.Count; j++)
                    {
                        ConcurrentDictionary<int, int> classifierPredictionMapping = id3ClassifierInstanceClassifierPredictionMappings.GetOrAdd(j, new ConcurrentDictionary<int, int>());
                        classifierPredictionMapping.GetOrAdd(i, tree.GetClass(discreteTestData[j]));
                    };

                    // Free the memory
                    tree = null;
                };

                // Calculate bias and variance for each classifier and store it.
                id3BaggerClassifierResult.GetOrAdd(maxDepth, BiasVarianceHelper.GetBiasVarianceAccuracy(discreteTestData, ClassIndex, id3BaggerInstanceClassifierPredictionMappings));
                id3ClassifierResult.GetOrAdd(maxDepth, BiasVarianceHelper.GetBiasVarianceAccuracy(discreteTestData, ClassIndex, id3ClassifierInstanceClassifierPredictionMappings));
            });

            Console.WriteLine("Id3 Classifier");
            Console.WriteLine("Max Depth, Bias, Variance, Accuracy");
            foreach(int maxDepth in id3ClassifierResult.Keys)
            {
                Console.WriteLine($"{maxDepth}, {id3ClassifierResult[maxDepth].Item1}, {id3ClassifierResult[maxDepth].Item2}, {id3ClassifierResult[maxDepth].Item3}");
            }

            Console.WriteLine();

            Console.WriteLine("Bagger");
            Console.WriteLine("Max Depth, Bias, Variance, Accuracy");
            foreach (int maxDepth in id3BaggerClassifierResult.Keys)
            {
                Console.WriteLine($"{maxDepth}, {id3BaggerClassifierResult[maxDepth].Item1}, {id3BaggerClassifierResult[maxDepth].Item2}, {id3ClassifierResult[maxDepth].Item3}");
            }

            Console.WriteLine();

            Console.WriteLine("Press ENTER to exit...");
            Console.ReadLine();
        }
    }
}
