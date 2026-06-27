import time
import pandas as pd

start = time.time()
df = pd.read_csv("/data/pokemon.csv")
end = time.time()

print(f"걸린 시간: {end - start:.4f}초")


