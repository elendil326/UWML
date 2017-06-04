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
            Id3Node child = tree.Children.ContainsKey(valueIndex)
                ? tree.Children[valueIndex]
                : tree.MaxChild;

            return GetClass(instance, child);
        }
    }
}
