using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveBayes
{
    public class NaiveBayesClassifier
    {
        // Counters for calculating probability based on Max Likelihood.
        private Dictionary<int, Dictionary<int, double>>[] _classFeatureValueCountMaps;
        private Dictionary<int, Dictionary<int, double>>[] _classFeatureValueProbabilityMaps;

        // Global probability of being class x
        private double[] _classCount;
        private double[] _globalClassProbability;
        private double _totalExamples = 0;

        // Prior probability helpers.
        public double[] ClassWeightToPrior { get; }

        public double[] ClassPriorProbability { get; }

        private NaiveBayesClassifier(double[] classWeightToPrior, double[] classPriorProbability)
        {
            if (classWeightToPrior == null || classPriorProbability == null || classWeightToPrior.Length != classPriorProbability.Length)
            {
                throw new ArgumentException("Invalid arrays passed");
            }

            ClassWeightToPrior = new double[classWeightToPrior.Length];
            ClassPriorProbability = new double[classPriorProbability.Length];
            Array.Copy(classWeightToPrior, ClassWeightToPrior, classWeightToPrior.Length);
            Array.Copy(classPriorProbability, ClassPriorProbability, classPriorProbability.Length);

            _classFeatureValueCountMaps = new Dictionary<int, Dictionary<int, double>>[classWeightToPrior.Length];
            _classFeatureValueProbabilityMaps = new Dictionary<int, Dictionary<int, double>>[classWeightToPrior.Length];
            _classCount = new double[classWeightToPrior.Length];
            _globalClassProbability = new double[classWeightToPrior.Length];
        }

        public static NaiveBayesClassifier Load(List<int[]> instances, int classIndex, double[] classWeightToPrior, double[] classPriorProbability)
        {
            NaiveBayesClassifier classifier = new NaiveBayesClassifier(classWeightToPrior, classPriorProbability);
            classifier.Train(instances, classIndex);
            return classifier;
        }

        public void Train(List<int[]> instances, int classIndex)
        {
            foreach (int[] instance in instances)
            {
                int classX = instance[classIndex];
                Dictionary<int, Dictionary<int, double>> featureValueCountMaps = _classFeatureValueCountMaps[classX];
                if (featureValueCountMaps == null)
                {
                    _classFeatureValueCountMaps[classX] = featureValueCountMaps = new Dictionary<int, Dictionary<int, double>>();
                }

                for (int featureIndex = 0; featureIndex < instance.Length; featureIndex++)
                {
                    if (featureIndex == classIndex) continue;

                    Dictionary<int, double> valueCountMap;
                    if (!featureValueCountMaps.TryGetValue(featureIndex, out valueCountMap))
                    {
                        featureValueCountMaps[featureIndex] = valueCountMap = new Dictionary<int, double>();
                    }

                    if (!valueCountMap.ContainsKey(instance[featureIndex]))
                    {
                        valueCountMap[instance[featureIndex]] = 0;
                    }
                    valueCountMap[instance[featureIndex]]++;
                }

                _classCount[classX]++;
                _totalExamples++;
            }

            for (int classI = 0; classI < _classFeatureValueProbabilityMaps.Length; classI++)
            {
                _classFeatureValueProbabilityMaps[classI] = new Dictionary<int, Dictionary<int, double>>();
                double classCount = _classCount[classI];
                double weight = ClassWeightToPrior[classI];
                double prior = ClassPriorProbability[classI];

                foreach (int featureIndex in _classFeatureValueCountMaps[classI].Keys)
                {
                    _classFeatureValueProbabilityMaps[classI][featureIndex] = new Dictionary<int, double>();
                    foreach (int value in _classFeatureValueCountMaps[classI][featureIndex].Keys)
                    {
                        double featureCount = _classFeatureValueCountMaps[classI][featureIndex][value];
                        _classFeatureValueProbabilityMaps[classI][featureIndex][value] = (featureCount + (weight * prior)) / (classCount + weight);
                    }
                }

                _globalClassProbability[classI] = classCount / _totalExamples;
            }
        }

        public double GetPredictionAccuracy(List<int[]> instances, int classIndex)
        {
            double accuracy = 0;
            foreach (int[] instance in instances)
            {
                if (GetClass(instance, classIndex) == instance[classIndex])
                {
                    accuracy++;
                }
            }

            return accuracy / instances.Count;
        }

        public int GetClass(int[] instance, int classIndex)
        {
            double[] logProbabilityPerClass = new double[_classCount.Length];

            for (int classI = 0; classI < logProbabilityPerClass.Length; classI++)
            {
                double sumOfProbabilities = 0;
                for (int featureIndex = 0; featureIndex < instance.Length; featureIndex++)
                {
                    if (featureIndex == classIndex) continue;

                    double featureProbability = 0;
                    int featureValue = instance[featureIndex];
                    if (!_classFeatureValueProbabilityMaps[classI][featureIndex].ContainsKey(featureValue))
                    {
                        featureProbability = ClassPriorProbability[classI];
                    }
                    else
                    {
                        featureProbability = _classFeatureValueProbabilityMaps[classI][featureIndex][featureValue];
                    }

                    if (featureProbability <= 0 || featureProbability >= 1) { throw new InvalidOperationException($"Feature {featureIndex} with value {featureValue} has a probability {featureProbability} which is invalid"); }

                    sumOfProbabilities += Math.Log(featureProbability);
                }

                logProbabilityPerClass[classI] = Math.Log(_globalClassProbability[classI]) + sumOfProbabilities;
            }

            int argMax = -1;
            double maxProb = double.MinValue;
            for (int classI = 0; classI < logProbabilityPerClass.Length; classI++)
            {
                if (logProbabilityPerClass[classI] > maxProb)
                {
                    maxProb = logProbabilityPerClass[classI];
                    argMax = classI;
                }
            }

            return argMax;
        }
    }
}
