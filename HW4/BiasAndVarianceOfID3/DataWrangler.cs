using Accord.MachineLearning;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BiasAndVarianceOfID3
{
    public class DataWrangler
    {
        private List<double[]> _data = new List<double[]>();

        private List<double[]> _testData = new List<double[]>();

        private List<double[]> _trainData = new List<double[]>();

        public IReadOnlyCollection<double[]> Data { get { return _data; } }

        public IReadOnlyCollection<double[]> TestData { get { return _testData; } }

        public IReadOnlyCollection<double[]> TrainData { get { return _trainData; } }

        private DataWrangler() { }

        public static DataWrangler Load(string filePath)
        {
            return Load(filePath, null);
        }

        public static DataWrangler Load(string filePath, IDictionary<int, int> continuesIndexKMap)
        {
            DataWrangler dataWrangler = new DataWrangler();

            // Read file
            int lineCounter = 0;
            using (StreamReader sr = new StreamReader(filePath))
            {
                do
                {
                    string line = sr.ReadLine();
                    if (line == null) break;

                    // First two lines are column names.
                    lineCounter++;
                    if (lineCounter < 3) continue;

                    string[] parts = line.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    dataWrangler._data.Add(parts.Select(s => double.Parse(s)).ToArray());
                } while (true);
            }

            if (continuesIndexKMap != null)
            {
                dataWrangler.ConvertContinuesToDiscrete(continuesIndexKMap);
            }

            SplitData(10, dataWrangler._data, out dataWrangler._trainData, out dataWrangler._testData);

            return dataWrangler;
        }

        private void ConvertContinuesToDiscrete(IDictionary<int, int> continuesIndexKMap)
        {
            foreach (int i in continuesIndexKMap.Keys)
            {
                // Prepare data to be analyzed in the KMeans algorithm
                double[][] data = new double[_data.Count][];
                for (int j = 0; j < _data.Count; j++)
                {
                    data[j] = new[] { _data[j][i] };
                }

                // Grab k from optimum map.
                int k = continuesIndexKMap[i];
                KMeans kMeans = new KMeans(k);

                var cluster = kMeans.Learn(data);

                for (int j = 0; j < _data.Count; j++)
                {
                    _data[j][i] = cluster.Decide(new[] { _data[j][i] });
                }
            }
        }

        private static void SplitData(int testDataPercentage, List<double[]> data, out List<double[]> testData, out List<double[]> trainData)
        {
            testData = new List<double[]>();
            trainData = new List<double[]>();

            int testDataCount = (data.Count * testDataPercentage) / 100;
            Random r = new Random();
            int limit = data.Count - 1;
            while (testDataCount > 0)
            {
                int randomIndex = r.Next(0, limit);

                double[] temp = data[limit];
                data[limit] = data[randomIndex];
                data[randomIndex] = temp;

                testDataCount--;
                limit--;
            }

            for (int i = 0; i < limit; i ++)
            {
                trainData.Add(data[i]);
            }
            for (int i = limit; i < data.Count; i++)
            {
                testData.Add(data[i]);
            }
        }
    }
}
