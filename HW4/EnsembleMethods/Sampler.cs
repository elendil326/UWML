using System;
using System.Collections.Generic;

namespace EnsembleMethods
{
    public static class Sampler
    {
        public static List<List<int[]>>  SampleData(List<int[]> instances, int numOfSamples)
        {
            Random r = new Random();
            List<List<int[]>> samples = new List<List<int[]>>();
            for (int i = 0; i < numOfSamples; i++)
            {
                List<int[]> sample = new List<int[]>();

                // Sample without replacement the same number of instances.
                for (int j = 0; j < instances.Count; j++)
                {
                    sample.Add(instances[r.Next(instances.Count)]);
                }

                // Store newly created sample.
                samples.Add(sample);
            }

            return samples;
        }
    }
}
