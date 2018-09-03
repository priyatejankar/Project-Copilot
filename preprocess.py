import pandas as pd
from sklearn.model_selection import train_test_split

# List of the columns on the .csv
column_names = ['Accident_Index', 'Location_Easting_OSGR',	'Location_Northing_OSGR', 'Longitude',
                'Latitude', 'Police_Force',	'Accident_Severity', 'Number_of_Vehicles',
                'Number_of_Casualties',	'Date',	'Day_of_Week', 'Time', 'Local_Authority_(District)',
                'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number',
                'Road_Type', 'Speed_limit', 'Junction_Detail', 'Junction_Control',
                '2nd_Road_Class', '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',
                'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions', 'Weather_Conditions',
                'Road_Surface_Conditions', 'Special_Conditions_at_Site', 'Carriageway_Hazards',
                'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location']

# Panda's object that contains the dataset

df = pd.read_csv("Accidents0515.csv", names=column_names)
print(df.shape)
# print(df)

# name of the columns we are willing to drop
column_todrop = ['Accident_Index', 'Location_Easting_OSGR',	'Location_Northing_OSGR',
                'Police_Force',	'Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties',
                'Local_Authority_(District)', 'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number',
                'Date', 'Day_of_Week', 'Road_Type', 'Junction_Detail', 'Junction_Control',
                '2nd_Road_Class', '2nd_Road_Number', 'Pedestrian_Crossing-Human_Control',
                'Pedestrian_Crossing-Physical_Facilities',
                'Special_Conditions_at_Site', 'Carriageway_Hazards',
                'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location']

df.drop(columns = column_todrop, inplace=True)

print(df.shape)
# print(df)

# print(df.loc[2])
# print(df.loc[2, 'Time'])
#
# a = df.loc[2, 'Time']
# print(type(a))
# print(df.iloc[[2]]['latitude'])

# Redefine the time column in the dataset with the following conventions
# 1 is for morning 7 - 12 am
# 2 is for afternoon 12 - 5pm
# 3 is for evening 5 -8 pm
# 4 is for night


classification_time = -1
category = -1
rows_to_drop = []
for i in range(1, df.shape[0]):
    time_str = df.loc[i, 'Time']
    try:
        classification_time = int(time_str[:2])

    except:
        classification_time = -1
        print(i)
        print("care bad time : ", time_str)
        # df.drop(i,  inplace= True)
        rows_to_drop.append(i)
        continue
    if (7 <= classification_time) and (classification_time < 12):
        category = 1
    if (12 <= classification_time) and (classification_time < 17):
        category = 2
    if (17 <= classification_time) and (classification_time < 20):
        category = 3
    else:
        category = 4
    df.set_value(i, 'Time', category)


print(df.shape)
# Drop invalid time rows
df.drop(rows_to_drop,  inplace= True)
# drop empty rows
df=df.dropna()
print(df)
# Save it again in a new csv containing only the features we want
df.to_csv("Reduced_accidents0515.csv", index=False)