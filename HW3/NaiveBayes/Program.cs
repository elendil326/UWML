using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveBayes
{
    class Program
    {
        // Change these paths to point to the correct files in your drive.
        private const string TainingDataFilePath = @"c:\Data\train";
        private const string TestDataFilePath = @"C:\Data\test";

        // Counters for calculating probability based on Max Likelihood.
        private static Dictionary<string, double> wordTotalCountMap = new Dictionary<string, double>(StringComparer.Ordinal);
        private static Dictionary<string, double> wordHamCountMap = new Dictionary<string, double>(StringComparer.Ordinal);
        private static Dictionary<string, double> wordHamProbabilityMap = new Dictionary<string, double>(StringComparer.Ordinal);

        // Prior probability helpers.
        private const double WeightToPrior = 1;
        private const double PriorHamProbability = 0.00001;

        // Ham or Spam constant
        private const string Ham = "ham";

        static void Main(string[] args)
        {
            Train();
            double accuracy = Test(false);
            Console.WriteLine($"Total accuracy is {accuracy}");

            // Compare accuracy with that of predicting all as spam.
            double dummyAccuracy = Test(true);
            Console.WriteLine($"Total accuracy of dummy prediction is {dummyAccuracy}");
        }

        private static void Train()
        {
            using (StreamReader sr = new StreamReader(TainingDataFilePath))
            {
                do
                {
                    string line = sr.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) { break; }

                    string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2) { throw new InvalidDataException($"The line read from the file does not conform to the format <ID Type word count word count ...>{Environment.NewLine}{line}"); }

                    bool isHam = string.Equals(parts[1], Ham, StringComparison.OrdinalIgnoreCase);
                    for (int i = 2; i < parts.Length; i++)
                    {
                        string word = parts[i];
                        i++;
                        int count = int.Parse(parts[i]);
                        if (isHam)
                        {
                            if (!wordHamCountMap.ContainsKey(word))
                            {
                                wordHamCountMap[word] = 0;
                            }
                            wordHamCountMap[word] += count;
                        }

                        if (!wordTotalCountMap.ContainsKey(word))
                        {
                            wordTotalCountMap[word] = 0;
                        }
                        wordTotalCountMap[word] += count;
                    }
                } while (true);
            }

            foreach (string word in wordTotalCountMap.Keys)
            {
                // Take how many times this word was in a ham mail. If never, default to cero.
                double hamCount = wordHamCountMap.ContainsKey(word) ? wordHamCountMap[word] : 0;

                // Calculate probability with prior
                wordHamProbabilityMap[word] = (hamCount + (WeightToPrior*PriorHamProbability)) / (wordTotalCountMap[word] + WeightToPrior);
            }
        }

        /// <summary>
        /// Test the data.
        /// </summary>
        /// <param name="predictAllAsSpam">If true, it ignores the training data and assumes all mails are spams. Otherwise, uses the training data to calculate probabilities as true NaiveBayes.</param>
        /// <returns>Returns the accuracy of the predicted values against the tests.</returns>
        private static double Test(bool predictAllAsSpam)
        {
            // Accuracy counters
            double correctTests = 0;
            double totalTests = 0;

            using (StreamReader sr = new StreamReader(TestDataFilePath))
            {
                do
                {
                    string line = sr.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) { break; }

                    string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2) { throw new InvalidDataException($"The line read from the file does not conform to the format <ID Type word count word count ...>{Environment.NewLine}{line}"); }

                    bool isTrueHam = string.Equals(parts[1], Ham, StringComparison.OrdinalIgnoreCase);

                    // The Naive Bayes is a multiplication of probabilities. Let's change it to sum of logs to avoid underflow. Since we are looking for the max, it doesn't matter.
                    double sumOfLogsForHam = 0;
                    double sumOfLogsForSpam = 0;
                    for (int i = 2; i < parts.Length; i++)
                    {
                        string word = parts[i];
                        i++;
                        int count = int.Parse(parts[i]);

                        double hamProbability = 0;

                        // If we haven't seen a word before, default to the prior probability.
                        if (!wordHamProbabilityMap.ContainsKey(word))
                        {
                            hamProbability = PriorHamProbability;
                        }
                        else
                        {
                            hamProbability = wordHamProbabilityMap[word];
                        }

                        if (hamProbability <= 0 || hamProbability >= 1)
                        {
                            throw new InvalidOperationException($"Word '{word}' has a ham probability of {hamProbability} which is invalid.");
                        }

                        sumOfLogsForHam += Math.Log(hamProbability);
                        sumOfLogsForSpam += Math.Log(1 - hamProbability);
                    }

                    bool isHam = sumOfLogsForHam > sumOfLogsForSpam;

                    if (isHam == isTrueHam)
                    {
                        correctTests++;
                    }

                    totalTests++;
                } while (true);
            }

            return correctTests / totalTests;
        }
    }
}
