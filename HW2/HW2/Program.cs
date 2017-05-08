using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace HW2
{
    class Program
    {
        // Modify these paths for valid local paths
        private static string _pathToTrainingFile = @"C:\Data\TrainingRatings.txt";
        private static string _pathToTestFile = @"C:\Data\TestingRatings.txt";
        private static string _pathToMoviesFile = @"C:\Data\Movies.txt";


        // Data structures for Mean votes per user
        private static Dictionary<int, int> userVoteMap = new Dictionary<int, int>();
        private static Dictionary<int, int> userVoteCountMap = new Dictionary<int, int>();
        private static Dictionary<int, double> userMeanMap = new Dictionary<int, double>();

        // Data structures for weight
        private static Dictionary<int, HashSet<int>> userMoviesRatedMap = new Dictionary<int, HashSet<int>>();

        // Data structures for prediction
        private static Dictionary<int, Dictionary<int, int>> movieUserVoteMaps = new Dictionary<int, Dictionary<int, int>>();

        static void Main(string[] args)
        {
            // Load Training data
            using (StreamReader sr = new StreamReader(_pathToTrainingFile))
            {
                while (true)
                {
                    string line = sr.ReadLine();
                    if (line == null) break;

                    // Read training line
                    int[] ids = line.Split(',').Select(s => (int)double.Parse(s)).ToArray();
                    int movieId = ids[0];
                    int userId = ids[1];
                    int rating = ids[2];

                    UpdateData(movieId, userId, rating);
                }
            }

            // Calculate the mean of each user
            CalculateMean();

            Console.WriteLine("Finished saving training data.");

            // Extra credit
            Console.WriteLine(string.Join(Environment.NewLine, GetMyMovies().Select(kvp => $"{kvp.Key} --- {kvp.Value}")));

            // Evaluation metrics counters
            int predictedCount = 0;
            double sumAbsoulteDifference = 0;
            double sumSquaredDifference = 0;

            using (StreamReader sr = new StreamReader(_pathToTestFile))
            {
                while(true)
                {
                    string line = sr.ReadLine();
                    if (line == null) break;

                    // Read test line
                    int[] ids = line.Split(',').Select(s => (int)double.Parse(s)).ToArray();
                    int movieId = ids[0];
                    int userA = ids[1];
                    int rating = ids[2];

                    //if (userA == 2170930)
                    //{
                    //    System.Diagnostics.Debugger.Break();
                    //}

                    // Calculate the sum of the weights times the difference from the mean for all none-zero weighted users
                    double sum = 0;
                    object lockObj = new object();
                    ConcurrentDictionary<int, double> userWeightMap = new ConcurrentDictionary<int, double>();
                    Parallel.ForEach(movieUserVoteMaps[movieId].Keys, userI =>
                    {
                        if (userA == userI) return;

                        double weight = GetWeight(userA, userI);
                        if (weight == 0) return;

                        // When userI hasn't evaluated the movie, we won't gain any value from assuming the mean in the sum, but the weight
                        // will contribute in the normalizer K so other users do not change the prediction as much as they should.
                        double userIValue = movieUserVoteMaps[movieId].ContainsKey(userI) ? movieUserVoteMaps[movieId][userI] : userMeanMap[userI];

                        double result = weight * (userIValue - userMeanMap[userI]);
                        lock (lockObj)
                        {
                            sum += result;
                        }

                        userWeightMap.AddOrUpdate(userI, (i) => weight, (i, d) => weight);
                    });

                    double prediction = 0;

                    if (userWeightMap.Count == 0)
                    {
                        // This user does not correlate to anybody. It could be because it rates all movies in the same way. Predict the mean for this.
                        prediction = userMeanMap[userA];
                    }
                    else
                    {
                        // Normalize with k as the inverse of the sums of the abosulte weights
                        double k = 0;
                        foreach (int userId in userWeightMap.Keys)
                        {
                            k += Math.Abs(userWeightMap[userId]);
                        }
                        k = 1 / k;

                        // Get prediction
                        prediction = userMeanMap[userA] + (k * sum);
                    }

                    if (double.IsNaN(prediction))
                    {
                        System.Diagnostics.Debugger.Break();
                    }

                    // Accumulate differences for error calculation
                    double diff = prediction - rating;
                    sumAbsoulteDifference += Math.Abs(diff);
                    sumSquaredDifference += diff * diff;
                    predictedCount++;

                    // Keep learning from test data
                    UpdateData(movieId, userA, rating);
                    CalculateMean();
                }
            }

            Console.WriteLine($"Mean absolute error: {sumAbsoulteDifference / predictedCount}");
            Console.WriteLine($"Root mean squared error: {Math.Sqrt(sumSquaredDifference / predictedCount)}");

            
            

        }

        private static List<KeyValuePair<string, double>> GetMyMovies()
        {
            Dictionary<int, int> myRatings = new Dictionary<int, int>()
            {
                { 10820, 5},
                { 4956, 5},
                { 851, 5},
                { 6206, 3},
                { 11914, 3},
                { 3962, 4 },
                { 1905, 4 },
                { 5991, 5},
                { 2782, 5},
                { 3624, 4},
                { 1549, 4},
                { 17328, 2 },
                { 2342, 3},
                { 2682, 4 },
                { 828, 3 },
                { 12870, 4 },
                { 14660, 3 },
                { 10421, 4 },
                { 3197, 3 },
                { 5020, 2 }
            };
            int userA = 999999;
            foreach (int movieId in myRatings.Keys)
            {
                UpdateData(movieId, userA, myRatings[movieId]);
            }

            CalculateMean();

            ConcurrentDictionary<int, double> myUserWeightMap = new ConcurrentDictionary<int, double>();
            SortedDictionary<double, List<int>> rankMovieMap = new SortedDictionary<double, List<int>>();
            foreach (int movieId in movieUserVoteMaps.Keys)
            {
                Dictionary<int, int> userVoteMap = movieUserVoteMaps[movieId];

                double sum = 0;
                object lockObj = new object();
                ConcurrentDictionary<int, double> userWeightMap = new ConcurrentDictionary<int, double>();
                Parallel.ForEach(movieUserVoteMaps[movieId].Keys, userI =>
                {
                    if (userA == userI) return;

                    double weight = 0;
                    if (!myUserWeightMap.ContainsKey(userI))
                    {
                        myUserWeightMap[userI] = GetWeight(userA, userI);
                    }
                    weight = myUserWeightMap[userI];
                    if (weight == 0) return;

                    // When userI hasn't evaluated the movie, we won't gain any value from assuming the mean in the sum, but the weight
                    // will contribute in the normalizer K so other users do not change the prediction as much as they should.
                    double userIValue = movieUserVoteMaps[movieId].ContainsKey(userI) ? movieUserVoteMaps[movieId][userI] : userMeanMap[userI];

                    double result = weight * (userIValue - userMeanMap[userI]);
                    lock (lockObj)
                    {
                        sum += result;
                    }

                    userWeightMap.AddOrUpdate(userI, (i) => weight, (i, d) => weight);
                });

                double prediction = 0;

                if (userWeightMap.Count == 0)
                {
                    // This user does not correlate to anybody. It could be because it rates all movies in the same way. Predict the mean for this.
                    prediction = userMeanMap[userA];
                }
                else
                {
                    // Normalize with k as the inverse of the sums of the abosulte weights
                    double k = 0;
                    foreach (int userId in userWeightMap.Keys)
                    {
                        k += Math.Abs(userWeightMap[userId]);
                    }
                    k = 1 / k;

                    // Get prediction
                    prediction = userMeanMap[userA] + (k * sum);
                }

                if (!rankMovieMap.ContainsKey(prediction))
                {
                    rankMovieMap[prediction] = new List<int>();
                }
                rankMovieMap[prediction].Add(movieId);
                if (double.IsNaN(prediction))
                {
                    System.Diagnostics.Debugger.Break();
                }
            }

            Dictionary<int, string> movieNameMap = new Dictionary<int, string>();
            using (StreamReader sr = new StreamReader(_pathToMoviesFile))
            {
                while (true)
                {
                    string line = sr.ReadLine();
                    if (line == null) break;
                    string[] parts = line.Split(',');
                    if (parts.Length < 3) continue;
                    int movieId = int.Parse(parts[0]);
                    string movieName = parts[2];

                    movieNameMap[movieId] = movieName;
                }
            }

            List<KeyValuePair<double, int>> rankMovies = new List<KeyValuePair<double, int>>();
            foreach (double rank in rankMovieMap.Keys)
            {
                foreach (int movieId in rankMovieMap[rank])
                {
                    rankMovies.Add(new KeyValuePair<double, int>(rank, movieId));
                }
            }
            return rankMovies.Select(kvp => new KeyValuePair<string, double>(movieNameMap[kvp.Value], kvp.Key)).ToList();
        }

        private static double GetWeight(int userA, int userI)
        {
            double sumDiffATimesDiffI = 0;
            double sumSquaredDiffA = 0;
            double sumSquaredDiffI = 0;
            foreach (int movieId in userMoviesRatedMap[userA].Intersect(userMoviesRatedMap[userI]))
            {
                double userADiff = movieUserVoteMaps[movieId][userA] - userMeanMap[userA];
                double userIDiff = movieUserVoteMaps[movieId][userI] - userMeanMap[userI];

                sumDiffATimesDiffI += userADiff * userIDiff;
                sumSquaredDiffA += userADiff * userADiff;
                sumSquaredDiffI += userIDiff * userIDiff;
            }

            return sumSquaredDiffI == 0 || sumSquaredDiffA == 0
                ? 0
                : sumDiffATimesDiffI / Math.Sqrt(sumSquaredDiffA * sumSquaredDiffI);
        }

        private static void UpdateData(int movieId, int userId, int rating)
        {
            if (!userVoteMap.ContainsKey(userId))
            {
                userVoteMap[userId] = 0;
            }
            userVoteMap[userId] += rating;

            if (!userVoteCountMap.ContainsKey(userId))
            {
                userVoteCountMap[userId] = 0;
            }
            userVoteCountMap[userId]++;

            if (!userMoviesRatedMap.ContainsKey(userId))
            {
                userMoviesRatedMap[userId] = new HashSet<int>();
            }
            userMoviesRatedMap[userId].Add(movieId);

            if (!movieUserVoteMaps.ContainsKey(movieId))
            {
                movieUserVoteMaps[movieId] = new Dictionary<int, int>();
            }
            if (!movieUserVoteMaps[movieId].ContainsKey(userId))
            {
                movieUserVoteMaps[movieId][userId] = rating;
            }
        }

        private static void CalculateMean()
        {
            foreach (int userId in userVoteMap.Keys)
            {
                userMeanMap[userId] = ((double)userVoteMap[userId]) / ((double)userVoteCountMap[userId]);
            }
        }
    }
}
