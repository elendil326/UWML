using ArffTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW1
{
    class Program
    {
        private static string _arffFile = @".\DataSetFiles\training_subsetD.arff";

        private static string _testArffFile = @".\DataSetFiles\testingD.arff";

        static void Main(string[] args)
        {
            // Training
            ArffHeader header = null;
            List<object[]> instances = new List<object[]>();
            using (ArffReader arffReader = new ArffReader(_arffFile))
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
            using (ArffReader arffReader = new ArffReader(_testArffFile))
            {
                header = arffReader.ReadHeader();
                object[] instance;
                while ((instance = arffReader.ReadInstance()) != null)
                {
                    instances.Add(instance);
                }
            }

            List<int[]> testData = new List<int[]>(instances.Select(objectArray => objectArray.Select(o => o == null ? -1 : (int)o).ToArray()));

            double[] confidences = new double[]
            {
                0.0,
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                0.9,
                0.95,
                0.99,
                0.9999
            };

            Parallel.ForEach(confidences, confidence =>
            {
                Id3Node tree = Id3Node.BuildTree(trainingData, trainingData[0].Length - 1, confidence);
                Console.WriteLine($"Confidence {confidence}: Num of nodes {GetCount(tree)}");
                Console.WriteLine($"Confidence {confidence}: Accuracy on train = { trainingData.Where(instance => GetClass(instance, tree) == instance[trainingData[0].Length - 1]).Count() / (double)trainingData.Count}");
                Console.WriteLine($"Confidence {confidence}: Accuracy on test = { testData.Where(instance => GetClass(instance, tree) == instance[testData[0].Length - 1]).Count() / (double)testData.Count}");
            });
        }

        private static int GetClass(int[] instance, Id3Node tree)
        {
            if (tree.IsLeaf) return tree.Class;

            int valueIndex = instance[tree.AttributeIndex];
            if (!tree.Children.ContainsKey(valueIndex))
            {
                Id3Node maxChild = null;
                int maxCount = int.MinValue;
                foreach (Id3Node child in tree.Children.Values)
                {
                    int count = child.ValueClassCounts.Values.SelectMany(kvp => kvp.Values).Sum();
                    if (count > maxCount)
                    {
                        maxCount = count;
                        maxChild = child;
                    }
                }

                return GetClass(instance, maxChild);
            }

            return GetClass(instance, tree.Children[valueIndex]);
        }

        private static void PrintTree(Id3Node tree, ArffHeader header, string suffix, int classAttributeIndex)
        {
            if (tree.IsLeaf)
            {
                Console.WriteLine($"{suffix}{((ArffNominalAttribute)header.Attributes.ElementAt(classAttributeIndex).Type).Values[tree.Class]}");
                return;
            }

            Console.WriteLine($"{suffix}{header.Attributes.ElementAt(tree.AttributeIndex).Name}");
            foreach (KeyValuePair<int, Id3Node> kvp in tree.Children)
            {
                if (kvp.Key == -1)
                {
                    Console.WriteLine($"{suffix}-->?");
                }
                Console.WriteLine($"{suffix}--> {((ArffNominalAttribute)header.Attributes.ElementAt(tree.AttributeIndex).Type).Values[kvp.Key]}");
                PrintTree(kvp.Value, header, $"{suffix}{suffix}", classAttributeIndex);
            }
        }

        private static int GetCount(Id3Node node)
        {
            if (node == null) return 0;
            if (node.IsLeaf) return 1;
            return 1 + node.Children.Values.Select(n => GetCount(n)).Sum();
        }
    }
}
