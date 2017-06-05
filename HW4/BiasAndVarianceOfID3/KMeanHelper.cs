using Accord;
using Accord.Math;
using Accord.MachineLearning;

namespace BiasAndVarianceOfID3
{
    public class KMeanHelper
    {


        public static double[] GenerateKElbowGraphPoints(int maxK, double[][] data)
        {
            double[] errors = new double[maxK + 1];

            for (int k = 1; k <= maxK; k++)
            {
                KMeans kMeans = new KMeans(k); // Use Square Euclidean distance
                var clusters = kMeans.Learn(data);
                errors[k] = kMeans.Error;
            }

            return errors;
        }
    }
}
