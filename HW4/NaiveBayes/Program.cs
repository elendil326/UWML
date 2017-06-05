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
        private static Dictionary<string, double> _wordSpamCountMap = new Dictionary<string, double>(StringComparer.Ordinal);
        private static Dictionary<string, double> _wordSpamProbabilityMap = new Dictionary<string, double>(StringComparer.Ordinal);

        // Prior probability helpers.
        private const double HamWeightToPrior = 1;
        private const double PriorHamProbability = 0.15;
        private const double SpamWeightToPrior = 1;
        private const double PriorSpamProbability = 0.85;

        // Ham or Spam constant
        private const string Ham = "ham";

        // Global probability of being ham or spam
        private static double _hamCount = 0;
        private static double _spamCount = 0;
        private static double _totalExamples = 0;
        private static double _globalHamProbability = 0;
        private static double _globalSpamProbability = 0;

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
                        else
                        {
                            _spamCount++;

                            if (!_wordSpamCountMap.ContainsKey(word))
                            {
                                _wordSpamCountMap[word] = 0;
                            }
                            _wordSpamCountMap[word] += count;
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
                // Take how many times this word was in a ham mail. If never, default to zero.
                double hamCount = _wordHamCountMap.ContainsKey(word) ? _wordHamCountMap[word] : 0;

                // Take how many times this word was in spam mail. If never, default to zero.
                double spamCount = _wordSpamCountMap.ContainsKey(word) ? _wordSpamCountMap[word] : 0;

                // Calculate probability with prior
                _wordHamProbabilityMap[word] = (hamCount + (HamWeightToPrior*PriorHamProbability)) / (_wordTotalCountMap[word] + HamWeightToPrior);
                _wordSpamProbabilityMap[word] = (spamCount + (SpamWeightToPrior * PriorSpamProbability)) / (_wordTotalCountMap[word] + SpamWeightToPrior);
            }

            // Probability of a doc being ham or not;
            _globalHamProbability = _hamCount / _totalExamples;
            // Probability of a doc being spam or not;
            _globalSpamProbability = _spamCount / _totalExamples;
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
                        double spamProbability = 0;

                        // If we haven't seen a word before, default to the prior probability.
                        if (!_wordHamProbabilityMap.ContainsKey(word))
                        {
                            hamProbability = PriorHamProbability;
                        }
                        else
                        {
                            hamProbability = _wordHamProbabilityMap[word];
                        }

                        if (!_wordSpamProbabilityMap.ContainsKey(word))
                        {
                            spamProbability = PriorSpamProbability;
                        }
                        else
                        {
                            spamProbability = _wordSpamProbabilityMap[word];
                        }

                        if (hamProbability <= 0 || hamProbability >= 1)
                        {
                            throw new InvalidOperationException($"Word '{word}' has a ham probability of {hamProbability} which is invalid.");
                        }
                        if (spamProbability <= 0 || spamProbability >= 1)
                        {
                            throw new InvalidOperationException($"Word '{word}' has a ham probability of {spamProbability} which is invalid.");
                        }

                        sumOfLogsForHam += Math.Log(hamProbability);
                        sumOfLogsForSpam += Math.Log(spamProbability);
                    }

                    bool isHam = Math.Log(_globalHamProbability) + sumOfLogsForHam > Math.Log((_globalSpamProbability)) + sumOfLogsForSpam;

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
