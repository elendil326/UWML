using System;
using System.Collections.Generic;
using System.Collections.Concurrent;

namespace BiasAndVarianceOfID3
{
    public static class BiasVarianceHelper
    {
        public static Tuple<double, double, double> GetBiasVarianceAccuracy(List<int[]> instances, int classIndex, IDictionary<int, ConcurrentDictionary<int, int>> instanceClassifierPredictionMappings)
        {
            double loss = 0;
            double bias = 0;
            double accuracy = 0;

            for (int i = 0; i < instances.Count; i++)
            {
                Tuple<double, double, double> lossBiasAccuracy = GetLossBiasAccuracy(instances[i], classIndex, instanceClassifierPredictionMappings[i]);
                loss += lossBiasAccuracy.Item1;
                bias += lossBiasAccuracy.Item2;
                accuracy += lossBiasAccuracy.Item3;
            }

            loss /= instances.Count;
            bias /= instances.Count;
            accuracy /= instances.Count;

            return new Tuple<double, double, double>(bias, loss - bias, accuracy);
        }

        public static Tuple<double, double, double> GetLossBiasAccuracy(int[] instance, int classIndex, IDictionary<int, int> classifierPredictionMapping)
        {
            double loss = 0;
            double bias = 0;
            double accuracy = 0;
            int[] classCounter = { 0, 0 };

            // Count 
            for (int i = 0; i < classifierPredictionMapping.Count; i++)
            {
                classCounter[classifierPredictionMapping[i]]++;
                if (instance[classIndex] == classifierPredictionMapping[i])
                {
                    accuracy++;
                }
            }

            int predictedModeClass = classCounter[0] > classCounter[1] ? 0 : 1;
            int realClassCount = classCounter[instance[classIndex]];

            loss = 1.0 - (realClassCount / classifierPredictionMapping.Count);
            bias = predictedModeClass != instance[classIndex] ? 1 : 0;
            accuracy /= classifierPredictionMapping.Count;

            return new Tuple<double, double, double>(loss, bias, accuracy);
        }
    }
}
