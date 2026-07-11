import numpy as np
import pandas as pd

samples = 4000

traces = []
labels = []

for i in range(samples):

    key_class = np.random.randint(0, 4)

    trace = np.random.normal(
        loc=key_class,
        scale=0.3,
        size=100
    )

    traces.append(trace)
    labels.append(key_class)

df = pd.DataFrame(traces)
df["label"] = labels

df.to_csv("dataset/traces.csv", index=False)

print("Dataset Created")