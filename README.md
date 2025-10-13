**Dataset**: [UCI El Niño dataset](https://archive.ics.uci.edu/ml/datasets/El+Nino)

I choose this dataset, becasue it was really useful to study the el-nino/la-nino. It has a lot of data with a lot of different colums for different variables like humidity, wind and temperatures.

**Progress summary:**
1. IDA/EDA: I learned that there is a huge imbalance in time. In the first 10 years, there is very little data compared to the next year. The number of observations per year and month varies in particular. So I tried to counter that by defining a different weight for each year. Furthermoremore, there are also a lot of data missing. Like for instance, the humidity wasn't measured in the first couple of years. This means that the dataset has extensive missing data in the early years, while later years are more complete I don't know yet how to deal with that, but I will see.  
