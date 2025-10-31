import GA
import time

## Diabetes

print("Performance on diabetes dataset. This set of best covariates is based on an exhaustive all subsets search using AIC as the fitness score (rather than CV-MSPE), with the MSPE simply the in-sample MSPE (and therefore likely a bit too low) based on that set of predictors. Your predictor set might differ a bit and your CV MSPE might be a bit higher. For this task, if your time reaches into the 10s or 100s of seconds, that would be concerning.")

print(f"AIC-based best predictors: ")
print(f"In-sample RMSPE: ")


t0 = time.time()
results = GA.select(X, y)
fulltime = time.time() - t0

print(f"Selected predictors {results.selected}.")
print(f"CV RMSPE: {results.rmspe}.")
print(f"Time taken: {fulltime}.")
