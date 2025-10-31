from setdata import X_diab, y_diab
import GA
import time

## Diabetes example

print("Performance on diabetes dataset. This set of best covariates is based on an exhaustive all subsets search using AIC as the fitness score (rather than CV-based prediction error), with the R^2 simply the in-sample R^2 (and therefore likely a bit too high) based on that set of predictors. Your predictor set might differ a bit and your CV R^2 might be a bit lower. For this task, if your time reaches into the 10s or 100s of seconds, that would be concerning.\n")

print(f"AIC-based best predictors (0-based indexing): 1,2,3,4,5,8 (sex+bmi+bp+s1+s2+s5)") 
print(f"In-sample R^2: 0.515")


t0 = time.time()
results = GA.select(X_diab, y_diab)
fulltime = time.time() - t0

print(f"Selected predictors {results['selected']}.")
print(f"CV R2: {results['R2']}.")
print(f"Time taken: {fulltime}.")
