using System;
using System.Collections.Generic;

namespace EnsembleMethods
{
    public class Sampler
    {
        public List<List<int[]>> Samples { get; } = new List<List<int[]>>();

        public Sampler(List<int[]> instances, int numOfSamples)
        {
            Random r = new Random();

            for (int i = 0; i < numOfSamples; i++)
            {
                List<int[]> sample = new List<int[]>();

                // Sample without replacement the same number of instances.
                for (int j = 0; j < instances.Count; j++)
                {
                    sample.Add(instances[r.Next(instances.Count)]);
                }

                // Store newly created sample.
                Samples.Add(sample);
            }
        }
    }
}
