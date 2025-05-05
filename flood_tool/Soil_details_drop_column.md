
# The labelled postcode data
df2 = pd.read_csv(os.path.join(ft._data_dir, 'postcodes_labelled.csv'))
df2.head()

# The soil data
import pandas as pd
df = pd.read_csv('soil_details.csv', encoding='ISO-8859-1')  
df

# Merge the two dataframes
df_merged = pd.merge(df,df2, on='soilType', how='left')
df_merged.to_csv('merged_dataframe.csv', index=False)

# High correlation that should drop
df_merged = df_merged.drop(columns=['Porosity Min(%)','Porosity Mean(%)','PSD max(mm)','Bulk Density Min (g/cm³)','Bulk Density Max (g/cm³)','Permeability mean(m²)'])

![alt text](image.png)
