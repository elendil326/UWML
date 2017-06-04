using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HW1
{
    public class Id3Classifier
    {
        public List<int[]> Instances { get; } = new List<int[]>();

        public double Confidence { get; }

        public Id3Node Tree { get; }

        public Id3Classifier(List<int[]> instances, int classIndex, double confidence)
        {
            Instances = instances;
            Confidence = confidence;

            Tree = Id3Node.BuildTree(instances, classIndex, confidence);
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
