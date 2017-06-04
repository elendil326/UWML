using HW1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EnsembleMethods
{
    public class Id3Bagger
    {
        public int NumberOfSamples { get; }

        private List<Id3Classifier> Classifiers { get; set; } = new List<Id3Classifier>();

        public Id3Bagger(int numberOfSamples)
        {
            NumberOfSamples = numberOfSamples;
        }

        public void Train(List<int[]> trainingData, int classIndex, double confidence)
        {
            Train(trainingData, classIndex, confidence, int.MaxValue);
        }

        public void Train(List<int[]> trainingData, int classIndex, double confidence, int maxDepth)
        {
            Classifiers = new List<Id3Classifier>();
            List<List<int[]>> samples = Sampler.SampleData(trainingData, NumberOfSamples);

            for (int i = 0; i < NumberOfSamples; i++)
            {
                Id3Classifier classifier = new Id3Classifier(samples[i], classIndex, confidence, maxDepth);
                Classifiers.Add(classifier);
            }
        }

        public int GetClass(int[] instance)
        {
            var groups = Classifiers.Select(c => c.GetClass(instance)).GroupBy(c => c);
            int maxCount = groups.Max(g => g.Count());
            return groups.First(g => g.Count() == maxCount).Key;
        }
    }
}
