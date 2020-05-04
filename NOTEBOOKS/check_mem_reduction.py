
''' This cell is a demo showing the performance of a good choice of data type
while reading the csv to dataframe - not necessary to run each time
'''
def mem_usage(pandas_obj):
  if isinstance(pandas_obj,pd.DataFrame):
    usage_b = pandas_obj.memory_usage(deep=True).sum()
  else:
    usage_b = pandas_obj.memory_usage(deep=True)    
  usage_mb = usage_b / 1024 ** 2
  return "{:03.2f} MB".format(usage_mb)

# Checking memory usage for the sample df before
print(df.info(memory_usage='deep'), '\n'+'.'*20)

# columns 'category'
df_obj = df.select_dtypes(include=['object']).copy()
converted_obj = pd.DataFrame()
for col in df_obj.columns:
  if len(df_obj[col].unique()) / len(df_obj[col]) < 0.5:
    converted_obj.loc[:,col] = df_obj[col].astype('category')
  else:
    converted_obj.loc[:,col] = df_obj[col]
print('For {} object columns: {} before, {} after conversion to category.'\
      .format(df_obj.shape[1], mem_usage(df_obj), mem_usage(converted_obj)))
# columns 'int'
df_int = df.select_dtypes(include=['int']).copy()
converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')
print('For {} int columns: {} before, {} after downcasting.'\
      .format(df_int.shape[1], mem_usage(df_int), mem_usage(converted_int)))
# columns 'float'
df_float = df.select_dtypes(include=['float']).copy()
converted_float = df_float.apply(pd.to_numeric, downcast='float')
print('For {} float columns: {} before, {} after downcasting.'\
      .format(df_float.shape[1], mem_usage(df_float), mem_usage(converted_float),))

df[converted_obj.columns] = converted_obj
df[converted_int.columns] = converted_int
df[converted_float.columns] = converted_float
print('.'*20+'\n')

# Re-checking memory usage for the sample df after
print('.'*20+'\n', df.info(memory_usage='deep'))
# (60.9MB->26.0MB: less than 43% !!!)

del df_obj, converted_obj, df_int, converted_int, df_float, converted_float