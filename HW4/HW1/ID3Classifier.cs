using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW1
{
    public class Id3Classifier
    {
        public double Confidence { get; }

        public Id3Node Tree { get; }

        public int MaxDepth { get; }

        public Id3Classifier(List<int[]> instances, int classIndex)
            : this(instances, classIndex, 0, int.MaxValue)
        { }

        public Id3Classifier(List<int[]> instances, int classIndex, double confidence)
            :this(instances, classIndex, confidence, int.MaxValue)
        { }

        public Id3Classifier(List<int[]> instances, int classIndex, double confidence, int maxDepth)
        {
            Confidence = confidence;

            Tree = Id3Node.BuildTree(instances, classIndex, confidence, maxDepth);
        }

        public int GetClass(int[] instance)
        {
            return GetClass(instance, Tree);
        }

        private int GetClass(int[] instance, Id3Node tree)
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
    }
}
