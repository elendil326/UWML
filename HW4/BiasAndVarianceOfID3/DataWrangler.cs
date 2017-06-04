using Accord.MachineLearning;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BiasAndVarianceOfID3
{
    public static class DataWrangler
    { 
        public static async Task<List<double[]>> LoadContinuousDataAsync(string filePath)
        {
            // Read file
            int lineCounter = 0;
            List<double[]> data = new List<double[]>();
            using (StreamReader sr = new StreamReader(filePath))
            {
                do
                {
                    string line = await sr.ReadLineAsync();
                    if (line == null) break;

                    // First two lines are column names.
                    lineCounter++;
                    if (lineCounter < 3) continue;

                    string[] parts = line.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    data.Add(parts.Select(s => double.Parse(s)).ToArray());
                } while (true);
            }

            return data;
        }

        public static Dictionary<int, GaussianClusterCollection> GetIndexClustersMap(List<double[]> continuousData, IDictionary<int, int> continuesIndexKMap)
        {
            Dictionary<int, GaussianClusterCollection> indexClusterMap = new Dictionary<int, GaussianClusterCollection>();

            Accord.Math.Random.Generator.Seed = 0;

            foreach (int i in continuesIndexKMap.Keys)
            {
                // Prepare data to be analyzed in the KMeans algorithm
                double[][] data = new double[continuousData.Count][];
                for (int j = 0; j < continuousData.Count; j++)
                {
                    data[j] = new[] { continuousData[j][i] };
                }

                // Grab k from optimum map.
                int k = continuesIndexKMap[i];
                KMeans kMeans = new KMeans(k);

                var cluster = kMeans.Learn(data);

                // Learn gaussian  mixture model based on k-model
                GaussianMixtureModel gaussianMixtureModel = new GaussianMixtureModel(kMeans);
                indexClusterMap[i] = gaussianMixtureModel.Learn(data);
            }

            return indexClusterMap;
        }

        public static List<int[]> ConvertContinuesToDiscrete(List<double[]> continuousData, Dictionary<int, GaussianClusterCollection> indexClusterMap)
        {
            List<int[]> discreteData = new List<int[]>(continuousData.Select(dArray => dArray.Select(d => (int)d).ToArray()));

            for(int i = 0; i < continuousData.Count; i++)
            {
                foreach (int j in indexClusterMap.Keys)
                {
                    discreteData[i][j] = indexClusterMap[j].Decide(new[] { continuousData[i][j] });
                }
            }

            return discreteData;
        }
    }
}
