import re
import pandas as pd
import matplotlib.pyplot as plt
sample = pd.read_csv("test.csv")
print(sample.columns)
all_stars = sample['stars']
ratings = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
star_occurrences = {key: 0 for key in ratings}
print(star_occurrences.keys())
for stars in all_stars:
    star_occurrences[stars] += 1
plt.plot(star_occurrences.keys(), star_occurrences.values(), '-ok')
plt.show()

all_reviewcount = sample['review_count']
review_occurrences = {}
plt.scatter(all_stars, all_reviewcount)
plt.xlabel('Star rating')
plt.ylabel('# of Reviews')
plt.show()
