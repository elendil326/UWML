﻿using Accord.MachineLearning;
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

            ConcurrentDictionary<int, ConcurrentDictionary<int, int>> id3ClassifierInstanceClassifierPredictionMappings = new ConcurrentDictionary<int, ConcurrentDictionary<int, int>>();
            ConcurrentDictionary<int, ConcurrentDictionary<int, int>> id3BaggerInstanceClassifierPredictionMappings = new ConcurrentDictionary<int, ConcurrentDictionary<int, int>>();

            ConcurrentDictionary<int, Tuple<double, double, double>> id3ClassifierResult = new ConcurrentDictionary<int, Tuple<double, double, double>>();
            ConcurrentDictionary<int, Tuple<double, double, double>> id3BaggerClassifierResult = new ConcurrentDictionary<int, Tuple<double, double, double>>();
            // From 0 to 5 maxDepth, calculate the bias and variance of the classifiers in parallel.
            Parallel.For(0, 4, (maxDepth) =>
            //for (int maxDepth = 0; maxDepth < 6; maxDepth++)
            {
                //for (int i = 0; i < BiasVarianceNumOfSamples; i++)
                Parallel.For(0, BiasVarianceNumOfSamples, i =>
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
                });

                // Calculate bias and variance for each classifier and store it.
                id3BaggerClassifierResult.GetOrAdd(maxDepth, GetBiasVarianceAccuracy(discreteTestData, id3BaggerInstanceClassifierPredictionMappings));
                id3ClassifierResult.GetOrAdd(maxDepth, GetBiasVarianceAccuracy(discreteTestData, id3ClassifierInstanceClassifierPredictionMappings));
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

        public static Tuple<double, double, double> GetBiasVarianceAccuracy(List<int[]> instances, IDictionary<int, ConcurrentDictionary<int, int>> instanceClassifierPredictionMappings)
        {
            double loss = 0;
            double bias = 0;
            double accuracy = 0;

            for (int i = 0; i < instances.Count; i++)
            {
                Tuple<double, double, double> lossBiasAccuracy = GetLossBiasAccuracy(instances[i], instanceClassifierPredictionMappings[i]);
                loss += lossBiasAccuracy.Item1;
                bias += lossBiasAccuracy.Item2;
                accuracy += lossBiasAccuracy.Item3;
            }

            loss /= instances.Count;
            bias /= instances.Count;
            accuracy /= instances.Count;

            return new Tuple<double, double, double>(bias, loss - bias, accuracy);
        }

        public static Tuple<double, double, double> GetLossBiasAccuracy(int[] instance, IDictionary<int, int> classifierPredictionMapping)
        {
            double loss = 0;
            double bias = 0;
            double accuracy = 0;
            int[] classCounter = { 0, 0 };

            // Count 
            for (int i = 0; i < BiasVarianceNumOfSamples; i++)
            {
                classCounter[classifierPredictionMapping[i]]++;
                if (instance[ClassIndex] == classifierPredictionMapping[i])
                {
                    accuracy++;
                }
            }

            int predictedModeClass = classCounter[0] > classCounter[1] ? 0 : 1 ;
            int realClassCount = classCounter[instance[ClassIndex]];

            loss = 1.0 - (realClassCount / BiasVarianceNumOfSamples);
            bias = predictedModeClass != instance[ClassIndex] ? 1 : 0;
            accuracy /= BiasVarianceNumOfSamples;

            return new Tuple<double, double, double>(loss, bias, accuracy);
        }
    }
}