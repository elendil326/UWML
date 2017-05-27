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
        public IReadOnlyCollection<string[]> Data { get; private set; } = new List<string[]>();

        private DataWrangler() { }

        public static DataWrangler Load(string filePath)
        {
            DataWrangler dataWrangler = new DataWrangler();

            // Read file
            List<string[]> lines = new List<string[]>();
            using (StreamReader sr = new StreamReader(filePath))
            {
                do
                {
                    string line = sr.ReadLine();
                    if (line == null) break;

                    string[] parts = line.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    lines.Add(parts);
                } while (true);
            }

            dataWrangler.Data = lines.AsReadOnly();
            return dataWrangler;
        }
    }
}
