using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace HW1
{
    public class Id3Node
    {
        public int AttributeIndex { get; private set; } = -1;

        public Dictionary<int, Dictionary<int, int>> ValueClassCounts { get; } = new Dictionary<int, Dictionary<int, int>>();

        public Id3Node Parent { get; private set; }

        public int ParentValue { get; private set; } = -1;

        public Dictionary<int, Id3Node> Children { get; } = new Dictionary<int, Id3Node>();

        public bool IsLeaf { get { return !Children.Any(); } }

        public int Class { get; set; }

        public int Count { get { return GetCount(this); } }

        private Id3Node() { }

        public static Id3Node BuildTree(List<int[]> instances, int classAttributeIndex)
        {
            return BuildTree(instances, classAttributeIndex, 0);
        }

        public static Id3Node BuildTree(List<int[]> instances, int classAttributeIndex, double confidence)
        {
            return BuildTree(instances, classAttributeIndex, confidence, int.MaxValue);
        }

        public static Id3Node BuildTree(List<int[]> instances, int classAttributeIndex, double confidence, int maxDepth)
        {
            bool[] visitedAttributes = new bool[instances.First().Length];
            return BuildTree(instances, classAttributeIndex, confidence, visitedAttributes, maxDepth);
        }

        public static Id3Node BuildTree(List<int[]> instances, int classAttributeIndex, double confidence, bool[] visitedAttributes)
        {
            return BuildTree(instances, classAttributeIndex, confidence, visitedAttributes, int.MaxValue);
        }

        /// <summary>
        /// Builds a tree based on the instances received.
        /// </summary>
        /// <param name="instances">The list of instances to train on.</param>
        /// <param name="classAttributeIndex">The index of the class attribute.</param>
        /// <param name="confidence">The confidence threshold for split stop.</param>
        /// <param name="visitedAttributes">Flag array to keep track of which attributes have already been visited so far.</param>
        /// <param name="maxDepth">The maximum depth of the tree.</param>
        /// <returns>The trained tree for the given list of instances and confidence threshold</returns>
        public static Id3Node BuildTree(List<int[]> instances, int classAttributeIndex, double confidence, bool[] visitedAttributes, int maxDepth)
        {
            if (maxDepth == 0) { return null; }
            if (maxDepth == 1) { return GetClassNodeForMax(instances, classAttributeIndex); }

            // If all instances are of the same class.
            int classType = instances[0][classAttributeIndex];
            if (instances.All(i => i[classAttributeIndex] == classType))
            {
                return new Id3Node
                {
                    Class = classType
                };
            }

            // If all attributes except the class attribute has been visited already
            bool allAttributesVisited = true;
            for (int i = 0; i < visitedAttributes.Length; i++)
            {
                if (i == classAttributeIndex)
                    continue;

                if (!visitedAttributes[i])
                {
                    allAttributesVisited = false;
                    break;
                }
            }

            // If all attributes have been visited, return the class with higher occurrance.
            if (allAttributesVisited)
            {
                return GetClassNodeForMax(instances, classAttributeIndex);
            }

            Id3Node bestNode = BestAttributeNode(instances, visitedAttributes, classAttributeIndex);

            if (!IsValidChiSquared(bestNode, confidence))
            {
                return GetClassNodeForMax(instances, classAttributeIndex);
            }

            bool[] localVisitedAttributes = new bool[visitedAttributes.Length];
            visitedAttributes.CopyTo(localVisitedAttributes, 0);
            localVisitedAttributes[bestNode.AttributeIndex] = true;

            foreach (int attributeValue in bestNode.ValueClassCounts.Keys)
            {
                bestNode.Children[attributeValue] = BuildTree(instances.Where(i => i[bestNode.AttributeIndex] == attributeValue).ToList(), classAttributeIndex, confidence, localVisitedAttributes, maxDepth - 1);
                bestNode.Children[attributeValue].Parent = bestNode;
                bestNode.Children[attributeValue].ParentValue = attributeValue;
            }

            return bestNode;
        }

        private static Id3Node BestAttributeNode(IEnumerable<int[]> instances, bool[] visitedAttributes, int classAttributeIndex)
        {
            // Each node will have the information on how many times a given value is present for a given class.
            Dictionary<int, Id3Node> attributeNodes = GetAttributeNodes(instances, visitedAttributes, classAttributeIndex);

            // Calculate the conditional entropy and pick the min, as this will maximize mutual information.
            int minEntropyAttributeIndex = -1;
            double minEntropy = double.MaxValue;
            foreach (KeyValuePair<int, Id3Node> kvp in attributeNodes)
            {
                int attributeIndex = kvp.Key;
                Id3Node currentNode = kvp.Value;

                double entropy = GetEntropy(currentNode.ValueClassCounts);
                if (entropy < minEntropy)
                {
                    minEntropy = entropy;
                    minEntropyAttributeIndex = attributeIndex;
                }
            }

            return attributeNodes[minEntropyAttributeIndex];
        }

        private static Dictionary<int, Id3Node> GetAttributeNodes(IEnumerable<int[]> instances, bool[] visitedAttributes, int classAttributeIndex)
        {
            Dictionary<int, Id3Node> attributeNodes = new Dictionary<int, Id3Node>();

            // Initialize all unvisited attribute nodes. Ignore the class attribute
            for (int i = 0; i < instances.First().Length - 1; i++)
            {
                if (visitedAttributes[i] || classAttributeIndex == i)
                    continue;

                attributeNodes.Add(i, new Id3Node
                {
                    AttributeIndex = i
                });
            }

            // Traverse instances and increase the counts of each attribute value for each class of the instance.
            foreach (int[] instance in instances)
            {
                int instanceClass = instance[classAttributeIndex];
                foreach (int attributeIndex in attributeNodes.Keys)
                {
                    // Get the index of the value of the current attribute
                    int valueIndex = instance[attributeIndex];
                    if (!attributeNodes[attributeIndex].ValueClassCounts.ContainsKey(valueIndex))
                    {
                        attributeNodes[attributeIndex].ValueClassCounts[valueIndex] = new Dictionary<int, int>();
                    }

                    if (!attributeNodes[attributeIndex].ValueClassCounts[valueIndex].ContainsKey(instanceClass))
                    {
                        attributeNodes[attributeIndex].ValueClassCounts[valueIndex][instanceClass] = 0;
                    }
                    attributeNodes[attributeIndex].ValueClassCounts[valueIndex][instanceClass]++;
                }
            }

            return attributeNodes;
        }

        /// <summary>
        /// Gets the conditional entropy for a map of values -> classes -> counts
        /// </summary>
        /// <param name="valueClassCounts">The map of values, the classes that appear and the counts in which they appear.</param>
        /// <returns></returns>
        private static double GetEntropy(Dictionary<int, Dictionary<int, int>> valueClassCounts)
        {
            // The sum of all counts of all classes of all possible values of the attribtes is the total.
            double total = valueClassCounts.SelectMany(kvp => kvp.Value).Select(kvp => kvp.Value).Sum();

            if (total == 0)
                return 0;

            // conditional entropy is the sum of the weighted entropy for each possible value of the attribute
            double entropy = valueClassCounts.Select(kvp => 
                    ( (kvp.Value.Select(classCounts => classCounts.Value).Sum()) / total // Calculate weight by adding all counts for one value and divinding by the total
                    ) * GetI(kvp.Value) // Multiply by the information measure
                ).Sum();

            return entropy;
        }

        /// <summary>
        /// Gets the information measure for a map between class index and class counts.
        /// </summary>
        /// <param name="classCounts">A map between class indexes and the count for that class.</param>
        /// <returns>Returns the information measure.</returns>
        private static double GetI(Dictionary<int, int> classCounts)
        {
            // The total is the sum of all classes counts.
            double total = classCounts.Values.Sum();
            if (total == 0)
                return 0;

            double i = classCounts.Values.Select(count =>
                    -1 * (count / total) * (Math.Log(count / total))
                ).Sum();

            return i;
        }

        private static bool IsValidChiSquared(Id3Node node, double confidence)
        {
            // Get total counts for all classes
            double total = 0;
            Dictionary<int, double> allClassCounts = new Dictionary<int, double>();
            foreach (KeyValuePair<int, Dictionary<int, int>> kvp in node.ValueClassCounts)
            {
                Dictionary<int, int> classCounts = kvp.Value;
                foreach (KeyValuePair<int, int> kvpClassCount in classCounts)
                {
                    int classIndex = kvpClassCount.Key;
                    int classCount = kvpClassCount.Value;

                    total += classCount;
                    if (!allClassCounts.ContainsKey(classIndex))
                    {
                        allClassCounts[classIndex] = 0;
                    }

                    allClassCounts[classIndex] += classCount;
                }
            }

            // Get expected counts for all possible values of the attribute
            Dictionary<int, Dictionary<int, double>> expectedCountPerValuePerClass = new Dictionary<int, Dictionary<int, double>>();
            foreach (KeyValuePair<int, Dictionary<int, int>> kvpValueClassCounts in node.ValueClassCounts)
            {
                int valueIndex = kvpValueClassCounts.Key;
                Dictionary<int, int> classCounts = kvpValueClassCounts.Value;

                if (!expectedCountPerValuePerClass.ContainsKey(valueIndex))
                {
                    expectedCountPerValuePerClass[valueIndex] = new Dictionary<int, double>();
                }

                double totalCountForValue = classCounts.Select(kvp => kvp.Value).Sum();

                foreach (KeyValuePair<int, int> kvpClassCount in classCounts)
                {
                    int classIndex = kvpClassCount.Key;
                    int classCount = kvpClassCount.Value;

                    expectedCountPerValuePerClass[valueIndex][classIndex] = allClassCounts[classIndex] * (totalCountForValue / total);
                }
            }

            double distribution = 0;
            foreach (KeyValuePair<int, Dictionary<int, int>> kvpValueClassCounts in node.ValueClassCounts)
            {
                int valueIndex = kvpValueClassCounts.Key;
                Dictionary<int, int> classCounts = kvpValueClassCounts.Value;

                foreach (KeyValuePair<int, int> kvpClassCount in classCounts)
                {
                    int classIndex = kvpClassCount.Key;
                    int classCount = kvpClassCount.Value;
                    double expectedCount = expectedCountPerValuePerClass[valueIndex][classIndex];

                    distribution += Math.Pow((classCount - expectedCount), 2) / expectedCount;
                }
            }

            int possibleValues = node.ValueClassCounts.Keys.Count;
            if (possibleValues <= 0)
            {
                possibleValues = 1;
            }
            ChiSquared chiSquared = new ChiSquared(possibleValues);

            return confidence <= chiSquared.CumulativeDistribution(distribution);
        }

        private static Id3Node GetClassNodeForMax(IEnumerable<int[]> instances, int classAttributeIndex)
        {
            IEnumerable<IGrouping<int, int[]>> classCounts = instances.GroupBy(i => i[classAttributeIndex]);
            int maxClassCount = -1;
            int maxCount = int.MinValue;
            foreach (IGrouping<int, int[]> classCount in classCounts)
            {
                int count = classCount.Count();
                if (count > maxCount)
                {
                    maxCount = count;
                    maxClassCount = classCount.Key;
                }
            }
            return new Id3Node
            {
                Class = maxClassCount
            };
        }

        private static int GetCount(Id3Node node)
        {
            if (node == null) return 0;
            if (node.IsLeaf) return 1;
            return 1 + node.Children.Values.Select(n => GetCount(n)).Sum();
        }
    }
}
