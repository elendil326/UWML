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
        private static Dictionary<string, double> _wordTotalCountMap = new Dictionary<string, double>(StringComparer.Ordinal);
        private static Dictionary<string, double> _wordHamCountMap = new Dictionary<string, double>(StringComparer.Ordinal);
        private static Dictionary<string, double> _wordHamProbabilityMap = new Dictionary<string, double>(StringComparer.Ordinal);

        // Prior probability helpers.
        private const double WeightToPrior = 99;
        private const double PriorHamProbability = 0.9999;

        // Ham or Spam constant
        private const string Ham = "ham";

        // Global probability of being ham or spam
        private static double _hamCount = 0;
        private static double _totalExamples = 0;
        private static double _globalHamProbability = 0;

        static void Main(string[] args)
        {
            Train();
            double accuracy = Test(false);
            Console.WriteLine($"Total accuracy is {accuracy}");

            // Compare accuracy with that of predicting all as spam.
            double dummyAccuracy = Test(true);
            Console.WriteLine($"Total accuracy of dummy prediction is {dummyAccuracy}");

            Console.WriteLine("Press ENTER to exit...");
            Console.ReadLine();
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
                            _hamCount++;

                            if (!_wordHamCountMap.ContainsKey(word))
                            {
                                _wordHamCountMap[word] = 0;
                            }
                            _wordHamCountMap[word] += count;
                        }

                        if (!_wordTotalCountMap.ContainsKey(word))
                        {
                            _wordTotalCountMap[word] = 0;
                        }
                        _wordTotalCountMap[word] += count;

                        _totalExamples++;
                    }
                } while (true);
            }

            foreach (string word in _wordTotalCountMap.Keys)
            {
                // Take how many times this word was in a ham mail. If never, default to cero.
                double hamCount = _wordHamCountMap.ContainsKey(word) ? _wordHamCountMap[word] : 0;

                // Calculate probability with prior
                _wordHamProbabilityMap[word] = (hamCount + (WeightToPrior*PriorHamProbability)) / (_wordTotalCountMap[word] + WeightToPrior);
            }

            // Probability of a doc being ham or not;
            _globalHamProbability = _hamCount / _totalExamples;
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
            double spamCounter = 0;

            using (StreamReader sr = new StreamReader(TestDataFilePath))
            {
                do
                {
                    string line = sr.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) { break; }

                    string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length < 2) { throw new InvalidDataException($"The line read from the file does not conform to the format <ID Type word count word count ...>{Environment.NewLine}{line}"); }

                    bool isTrueHam = string.Equals(parts[1], Ham, StringComparison.OrdinalIgnoreCase);
                    if (!isTrueHam)
                    {
                        spamCounter++;
                    }

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
                        if (!_wordHamProbabilityMap.ContainsKey(word))
                        {
                            hamProbability = PriorHamProbability;
                        }
                        else
                        {
                            hamProbability = _wordHamProbabilityMap[word];
                        }

                        if (hamProbability <= 0 || hamProbability >= 1)
                        {
                            throw new InvalidOperationException($"Word '{word}' has a ham probability of {hamProbability} which is invalid.");
                        }

                        sumOfLogsForHam += Math.Log(hamProbability);
                        sumOfLogsForSpam += Math.Log(1 - hamProbability);
                    }

                    bool isHam = Math.Log(_globalHamProbability) + sumOfLogsForHam > Math.Log((1 - _globalHamProbability)) + sumOfLogsForSpam;

                    if (isHam == isTrueHam)
                    {
                        correctTests++;
                    }

                    totalTests++;
                } while (true);
            }

            // If we should predict all as spams, we will only have correctly guessed the true spams.
            return predictAllAsSpam 
                ? spamCounter / totalTests 
                : correctTests / totalTests;
        }
    }
}
