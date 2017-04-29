using ArffTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW1
{
    class Program
    {
        private static string _arffFile = @".\DataSetFiles\training_subsetD.arff";

        private static string _testArffFile = @".\DataSetFiles\testingD.arff";

        private static string _outputFolder = @"c:\Test";

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

            PrintAsCsv(header, trainingData, @"c:\users\andresz\desktop\data.csv");

            Parallel.ForEach(confidences, confidence =>
            {
                Id3Node tree = Id3Node.BuildTree(trainingData, trainingData[0].Length - 1, confidence);

                Console.WriteLine($"Confidence {confidence}: Num of nodes {GetCount(tree)}");
                // Test accuracy on training
                Console.WriteLine($"Confidence {confidence}: Accuracy on train = { trainingData.Where(instance => GetClass(instance, tree) == instance[trainingData[0].Length - 1]).Count() / (double)trainingData.Count}");

                // Test accuracy on test
                Console.WriteLine($"Confidence {confidence}: Accuracy on test = { testData.Where(instance => GetClass(instance, tree) == instance[testData[0].Length - 1]).Count() / (double)testData.Count}");

                StringBuilder sb = new StringBuilder();
                StringBuilder sbMaxPositive = new StringBuilder();
                StringBuilder sbMaxNegative = new StringBuilder();
                int maxPositive = int.MinValue;
                int maxNegative = int.MinValue;
                // Only print small trees.
                if (confidence > 0.5)
                {
                    PrintTreeAsRules(sb, ref sbMaxPositive, ref sbMaxNegative, ref maxPositive, ref maxNegative, tree, header);
                    sb.AppendLine("The most max positive rule is:");
                    sb.AppendLine(sbMaxPositive.ToString());
                    sb.AppendLine();
                    sb.AppendLine("The most max negative rule is:");
                    sb.AppendLine(sbMaxNegative.ToString());
                    Directory.CreateDirectory(_outputFolder);
                    File.WriteAllText(Path.Combine(_outputFolder, $"Tree{confidence}.txt"), sb.ToString());
                }
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

        private static void PrintTreeAsRules(StringBuilder sb, ref StringBuilder sbMaxPositive, ref StringBuilder sbMaxNegative, ref int maxPositive, ref int maxNegative, Id3Node tree, ArffHeader header)
        {
            if (tree.IsLeaf)
            {
                Id3Node leaf = tree;
                StringBuilder localSB = new StringBuilder();
                localSB.AppendLine("Rule is:");
                localSB.AppendLine();

                int count = tree.Parent.ValueClassCounts[tree.ParentValue].Values.Sum();
                while (tree.Parent != null)
                {
                    string value = tree.ParentValue == -1 ? "?" : ((ArffNominalAttribute)header.Attributes.ElementAt(tree.Parent.AttributeIndex).Type).Values[tree.ParentValue];
                    localSB.Append($"<{header.Attributes.ElementAt(tree.Parent.AttributeIndex).Name}> equals to <{value}>");
                    localSB.Append(" and ");
                    tree = tree.Parent;
                }
                localSB.AppendLine();
                localSB.AppendLine("---------------------------------------");

                if (leaf.Class == 0)
                {
                    sb.Append(localSB.ToString());
                    if (count > maxPositive)
                    {
                        maxPositive = count;
                        sbMaxPositive = localSB;
                        sbMaxPositive.AppendLine($"COUNT: {count}");
                    }
                }
                else
                {
                    if (count > maxNegative)
                    {
                        maxNegative = count;
                        sbMaxNegative = localSB;
                        sbMaxNegative.AppendLine($"COUNT: {count}");
                    }
                }
            }
            else
            {
                foreach (KeyValuePair<int, Id3Node> kvp in tree.Children)
                {

                    PrintTreeAsRules(sb, ref sbMaxPositive, ref sbMaxNegative, ref maxPositive, ref maxNegative, kvp.Value, header);
                }
            }
        }

        private static int GetCount(Id3Node node)
        {
            if (node == null) return 0;
            if (node.IsLeaf) return 1;
            return 1 + node.Children.Values.Select(n => GetCount(n)).Sum();
        }

        private static void PrintAsCsv(ArffHeader header, List<int[]> instances, string filePath)
        {
            using (StreamWriter sw = new StreamWriter(filePath))
            {
                sw.WriteLine(string.Join(",", header.Attributes.Select(a => a.Name)));
                foreach (int[] instance in instances)
                {
                    sw.WriteLine(string.Join(",", instance.Select((valueIndex, attributeIndex) =>
                    {
                        return valueIndex >= 0
                            ? ((ArffNominalAttribute)header.Attributes.ElementAt(attributeIndex).Type).Values[valueIndex]
                            : "?";

                    })));
                }
            }
        }
    }
}
