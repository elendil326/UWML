using ArffTools;
using HW1;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

namespace EnsembleMethods
{
    class Program
    {
        // Modify these paths to the correct path in your computer
        private const string TrainingArffFile = @"c:\Data\train.arff";
        private const string TestArffFile = @"c:\Data\test.arff";

        // Index in the Arff file that specifies the class of the instance.
        private const int ClassIndex = 0;

        // No pruning
        private const int Confidence = 0;

        private static object _lockConsole = new object();

        static void Main(string[] args)
        {
            // Training
            ArffHeader header = null;
            List<object[]> instances = new List<object[]>();
            using (ArffReader arffReader = new ArffReader(TrainingArffFile))
            {
                header = arffReader.ReadHeader();
                object[] instance;
                while ((instance = arffReader.ReadInstance()) != null)
                {
                    instances.Add(instance);
                }
            }

            List<int[]> trainingData = new List<int[]>(instances.Select(objectArray => objectArray.Select(o => o == null ? -1 : (int)o).ToArray()));

            // Test
            instances = new List<object[]>();
            using (ArffReader arffReader = new ArffReader(TestArffFile))
            {
                header = arffReader.ReadHeader();
                object[] instance;
                while ((instance = arffReader.ReadInstance()) != null)
                {
                    instances.Add(instance);
                }
            }

            List<int[]> testData = new List<int[]>(instances.Select(objectArray => objectArray.Select(o => o == null ? -1 : (int)o).ToArray()));

            Console.WriteLine("Number of Samples, training accuracy, test accuracy");
            // Do this excercise multiple times as sampling is random.
            Parallel.For(0, 100, (k) =>
            {
                // Dictionaries to store results for different accuracies. KEEP IN SYNC :)
                ConcurrentDictionary<int, double> sampleTrainingAccuraciesMap = new ConcurrentDictionary<int, double>(new Dictionary<int, double>
                {
                    { 1, 0 },
                    { 3, 0 },
                    { 5, 0 },
                    { 10, 0 },
                    { 20, 0 },
                    { 25, 0 },
                    { 50, 0 },
                    { 75, 0 },
                    { 100, 0 }
                });
                ConcurrentDictionary<int, double> sampleTestAccuraciesMap = new ConcurrentDictionary<int, double>(new Dictionary<int, double>
                {
                    { 1, 0 },
                    { 3, 0 },
                    { 5, 0 },
                    { 10, 0 },
                    { 20, 0 },
                    { 25, 0 },
                    { 50, 0 },
                    { 75, 0 },
                    { 100, 0 }
                });

                // Calculate different sample accuracies in parallel.
                Parallel.ForEach(sampleTrainingAccuraciesMap.Keys, numOfSamples =>
                {
                    Id3Bagger bagger = new Id3Bagger(numOfSamples);
                    bagger.Train(trainingData, ClassIndex, Confidence);

                    // Evaluate training and test to look out for overfitting.
                    sampleTrainingAccuraciesMap[numOfSamples] = Evaluate(trainingData, bagger);
                    sampleTestAccuraciesMap[numOfSamples] = Evaluate(testData, bagger);
                });

                lock(_lockConsole)
                {
                    foreach (int numOfSamples in sampleTestAccuraciesMap.Keys.OrderBy(n => n))
                    {
                        Console.WriteLine($"{numOfSamples},{sampleTrainingAccuraciesMap[numOfSamples]},{sampleTestAccuraciesMap[numOfSamples]}");
                    }
                }
            });

            Console.WriteLine("Press ENTER to exit...");
            Console.ReadLine();
        }

        private static double Evaluate(List<int[]> instances, Id3Bagger bagger)
        {
            // Accuracy helpers
            double totalExamples = instances.Count;
            double correctAnswers = 0;

            foreach (int[] instance in instances)
            {
                if (bagger.GetClass(instance) == instance[ClassIndex])
                {
                    correctAnswers++;
                }
            }

            return correctAnswers / totalExamples;
        }
    }
}
