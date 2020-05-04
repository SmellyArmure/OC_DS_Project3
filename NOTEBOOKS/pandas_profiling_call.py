import pandas_profiling
# Generating a Data Profiling html report via pandas_profiling
profile = pandas_profiling.ProfileReport(dfcopy) #, check_correlation = False
profile.to_file("OFFacts_report2_3.html")
profile
# extract rejected columns (based on correlation coeff greater than 0.9)
rejected_variables_90 = profile.get_rejected_variables(threshold=0.9)
len(rejected_variables_90)