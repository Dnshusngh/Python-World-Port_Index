# Python-World-Port_Index
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns

# Load the dataset
dataset_path = '/kaggle/input/world-port-index/UpdatedPub150.csv'
df = pd.read_csv(dataset_path)


# Show the first few rows to get an initial understanding of the data
df.head()

df.info()

# Function to delete columns with more than 60% values equal to zero, space, and 'Unknown'
def drop_columns(df):
    num_rows = len(df)
    threshold = 0.6 * num_rows
    cols_to_drop = []
    
    for col in df.columns:
        zero_count = (df[col] == 0).sum()
        space_count = (df[col] == ' ').sum()
        unknown_count = (df[col] == 'Unknown').sum()
        
        if zero_count + space_count + unknown_count > threshold:
            cols_to_drop.append(col)
            
    return df.drop(columns=cols_to_drop)

# Apply the function to the sample DataFrame
df = drop_columns(df)
df.info()

df.describe()

# Count number of cells in each column which have just one space as their field
# This is a common way of representing missing values in a dataset
df[df == ' '].count()

#print of each column and the number of space values it has sorted in descending order top twenty records
df[df == ' '].count().sort_values(ascending=False).head(20)

#print of each column and the number of zero values it has sorted in descending order top twenty records
df[df == 0].count().sort_values(ascending=False).head(20)

#print of each column and the number of Unknown values it has sorted in descending order top twenty records
df[df == 'Unknown'].count().sort_values(ascending=False).head(50)

# print all column names in the dataset in a grid format
print(df.columns.tolist())

# Drop the specified columns
df = df.drop(columns=['UN/LOCODE', 'Digital Nautical Chart', 'Publication Link', 'Standard Nautical Chart'])

# Replace fields with just a space with 'unknown'
df = df.fillna('Unknown')

# Replace space with 'unknown'
df = df.replace(' ', 'unknown')

# Identifying numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate basic descriptive statistics for numerical columns
numerical_stats = df[numerical_columns].describe().transpose()
numerical_stats['variance'] = df[numerical_columns].var()
numerical_stats = numerical_stats[['mean', '50%', 'variance', 'min', 'max']]
numerical_stats.rename(columns={'50%': 'median'}, inplace=True)

numerical_stats


# Identifying categorical columns
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

# Calculate basic descriptive statistics for categorical columns
categorical_stats = df[categorical_columns].describe().transpose()

categorical_stats.head(10)  # Displaying first 10 to keep the output manageable


# Importing libraries for data visualization
import matplotlib.pyplot as plt
import geopandas as gpd

# Create a GeoDataFrame with the latitude and longitude of the ports
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Load the world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plotting the world map and ports
fig, ax = plt.subplots(1, 1, figsize=(30, 30))
world.boundary.plot(ax=ax, linewidth=1, color='black')
gdf.plot(ax=ax, markersize=5, color='red', alpha=0.6)
plt.title('Geographical Distribution of Ports')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Analyzing the distribution of ports by country and region
country_distribution = df['Country Code'].value_counts().reset_index()
country_distribution.columns = ['Country Code', 'Number of Ports']

region_distribution = df['Region Name'].value_counts().reset_index()
region_distribution.columns = ['Region Name', 'Number of Ports']

# Plotting the distribution by country and region
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# By Country
axes[0].barh(country_distribution['Country Code'][:10], country_distribution['Number of Ports'][:10], color='blue')
axes[0].set_title('Top 10 Countries by Number of Ports')
axes[0].set_xlabel('Number of Ports')
axes[0].set_ylabel('Country Code')

# By Region
axes[1].barh(region_distribution['Region Name'][:10], region_distribution['Number of Ports'][:10], color='green')
axes[1].set_title('Top 10 Regions by Number of Ports')
axes[1].set_xlabel('Number of Ports')
axes[1].set_ylabel('Region Name')

plt.tight_layout()
plt.show()


# Analyzing the distribution of ports by world water bodies
water_body_distribution = df['World Water Body'].value_counts().reset_index()
water_body_distribution.columns = ['World Water Body', 'Number of Ports']

# Plotting the distribution by world water bodies
plt.figure(figsize=(20, 5))
plt.barh(water_body_distribution['World Water Body'][:10], water_body_distribution['Number of Ports'][:10], color='purple')
plt.title('Top 10 World Water Bodies by Number of Ports')
plt.xlabel('Number of Ports')
plt.ylabel('World Water Body')
plt.tight_layout()
plt.show()


# Extracting columns related to facilities and services
facilities_columns = [col for col in df.columns if 'Facility -' in col or 'Supplies -' in col or 'Services -' in col]

# Count the frequency of each facility and service
facilities_count = df[facilities_columns].apply(pd.Series.value_counts).fillna(0).transpose()
facilities_count.columns = ['No', 'Unknown', 'Yes']
facilities_count['Total Ports'] = facilities_count.sum(axis=1)

# Calculate the percentage availability, unavailability, and unknown status
facilities_count['Percentage Available'] = (facilities_count['Yes'] / facilities_count['Total Ports']) * 100
facilities_count['Percentage Unavailable'] = (facilities_count['No'] / facilities_count['Total Ports']) * 100
facilities_count['Percentage Unknown'] = (facilities_count['Unknown'] / facilities_count['Total Ports']) * 100

# Sort by percentage availability
facilities_count = facilities_count.sort_values(by='Percentage Available', ascending=False)

# Combine the most common and least common facilities into a single DataFrame
facilities = facilities_count[['Percentage Available', 'Percentage Unavailable', 'Percentage Unknown']]

# Display all facilities in a tabular format
facilities


# Set the style for the plots
sns.set(style="whitegrid")

# Plot Distribution of Harbor types
plt.figure(figsize=(20, 10))
sns.countplot(x='Harbor Type', data=df)
plt.title('Distribution of Harbor Sizes')
plt.xlabel('Harbor Size')
plt.ylabel('Count')
plt.show()

# Set the style for the plots
sns.set(style="whitegrid")

# Create a new figure and a 2x2 grid of subplots
plt.figure(figsize=(20, 8))


# Plot Harbor Types by Channel Depth
sns.boxplot(x='Harbor Type', y='Channel Depth (m)', data=df)
plt.title('Harbor Types by Channel Depth')
plt.xlabel('Harbor Type')
plt.ylabel('Channel Depth (m)')



plt.show()

# Convert the facilities data to numerical format for correlation analysis
facilities_numerical = df[facilities_columns].replace({'Yes': 1, 'No': 0, 'Unknown': None}).fillna(0)

# Calculate the correlation matrix
correlation_matrix = facilities_numerical.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Facilities')
plt.show()

# Create a bar chart to visualize the distribution of ports by Country Code for the first 50 countries
plt.figure(figsize=(20, 10))
sns.countplot(data=df, x='Country Code', order=df['Country Code'].value_counts().index[:50])
plt.title('Distribution of Ports by Country Code (First 50 Countries)')
plt.xlabel('Country Code')
plt.ylabel('Number of Ports')
plt.xticks(rotation=90)
plt.show()

# Calculate the global availability of key facilities
global_availability = facilities_numerical.mean().sort_values(ascending=False).head(5)

# Create a pie chart to visualize the global availability of key facilities with a legend
plt.figure(figsize=(12, 8))
plt.pie(global_availability, labels=global_availability.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
plt.legend(title='Facilities', bbox_to_anchor=(1, 1), loc='upper left')
plt.title('Global Availability of Key Facilities (Top 5)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# List of depth-related columns
depth_columns = ['Channel Depth (m)', 'Anchorage Depth (m)', 'Cargo Pier Depth (m)']

# Create histograms to visualize the distribution of various types of depths
fig, axes = plt.subplots(len(depth_columns), 1, figsize=(18, 20))

for i, col in enumerate(depth_columns):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# plot using pie chart
plt.figure(figsize=(12, 8))
plt.pie(df['Harbor Size'].value_counts(), labels=df['Harbor Size'].value_counts().index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
plt.legend(title='Harbor Size', bbox_to_anchor=(1, 1), loc='upper left')
plt.title('Distribution of Ports by Harbor Size')
plt.show()


# Create a bar chart to visualize the distribution of ports by Region and Size
plt.figure(figsize=(20, 10))
sns.countplot(data=df, x='Region Name', hue='Harbor Size', order=df['Region Name'].value_counts().index[:15])
plt.title('Distribution of Ports by Top 15 Regions and Harbor Size')
plt.xlabel('Region Name')
plt.ylabel('Number of Ports')
plt.xticks(rotation=90)
plt.legend(title='Harbor Size')
plt.show()

# Import necessary folium plugins for marker clustering and custom icons
import folium
from folium import Icon
from folium.plugins import MarkerCluster


# Initialize the map with clustering enabled
m_final = folium.Map(location=[20, 0], zoom_start=3)
marker_cluster = MarkerCluster().add_to(m_final)



# Color mapping for Harbor Size
color_map = {
    'Very Small': 'orange',
    'Small': 'yellow',
    'Medium': 'green',
    'Large': 'blue',
    'Unknown': 'red'
}

# For the sake of demonstration, focusing on the columns related to facilities and supplies
focus_columns = ['Supplies - Fuel Oil', 'Supplies - Diesel Oil', 'Facilities - Wharves', 'Facilities - Anchorage']

# Add enhanced markers to the map
for idx, row in df.iterrows():
    latitude, longitude = row['Latitude'], row['Longitude']
    region_name = row['Region Name']
    main_port_name = row['Main Port Name']
    country_code = row['Country Code']
    world_water_body = row['World Water Body']
    harbor_size = row['Harbor Size']
    harbor_type = row['Harbor Type']
    
    # Create popup text with additional facility and supply information
    facility_info = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in focus_columns])
    popup_text = f"<b>Port:</b> {main_port_name}<br>\
                  <b>Region:</b> {region_name}<br>\
                  <b>Country:</b> {country_code}<br>\
                  <b>Water Body:</b> {world_water_body}<br>\
                  <b>Harbor Size:</b> {harbor_size}<br>\
                  <b>Harbor Type:</b> {harbor_type}<br>\
                  {facility_info}"
    
    # Create a popup with custom width
    popup = folium.Popup(popup_text, max_width=300)
    
    # Determine the icon color based on harbor size
    icon_color = color_map.get(harbor_size, 'gray')
    
    folium.Marker(
        [latitude, longitude], 
        popup=popup, 
        icon=Icon(color=icon_color)
    ).add_to(marker_cluster)

# Add legend to the map
legend_html = """
<div style="position: fixed; 
            top: 50px; left: 50px; width: 150px; height: 120px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            ">&nbsp; Harbor Size Legend <br>
              &nbsp; Very Small &nbsp; <i class="fa fa-circle" style="color:orange"></i><br>
              &nbsp; Small &nbsp; <i class="fa fa-circle" style="color:yellow"></i><br>
              &nbsp; Medium &nbsp; <i class="fa fa-circle" style="color:green"></i><br>
              &nbsp; Large &nbsp; <i class="fa fa-circle" style="color:blue"></i><br>
              &nbsp; Unknown &nbsp; <i class="fa fa-circle" style="color:red"></i>
</div>
"""
m_final.get_root().html.add_child(folium.Element(legend_html))

# Show the final map
m_final



unique_harbor_types = df['Harbor Type'].unique().tolist()

# Create a composite list of unique categorical values
unique_categorical_values = unique_harbor_types

# Create a mapping of unique categorical values to icons
# Due to the limitation of available icons, we'll repeat the list to cover all unique values
available_icons = ['leaf', 'tower', 'cloud', 'fire', 'star', 'heart', 'flag', 'flash', 'cog', 'plane', 'gift']
icon_map_unique = {value: available_icons[i % len(available_icons)] for i, value in enumerate(unique_categorical_values)}



# Initialize the map for extended categorical fields
m_categorical = folium.Map(location=[20, 0], zoom_start=3)
marker_cluster = MarkerCluster().add_to(m_categorical)


# Add markers to the map based on extended categorical fields
for idx, row in df.iterrows():
    latitude, longitude = row['Latitude'], row['Longitude']
    harbor_type = row['Harbor Type']
    shelter_afforded = row['Shelter Afforded']
    entrance_restriction_tide = row['Entrance Restriction - Tide']
    overhead_limits = row['Overhead Limits']
    good_holding_ground = row['Good Holding Ground']
    quarantine_pratique = row['Quarantine - Pratique']
    first_port_entry = row['First Port of Entry']
    
    # Create popup text with additional fields
    popup_text = f"<b>Harbor Type:</b> {harbor_type}<br>\
                  <b>Shelter Afforded:</b> {shelter_afforded}<br>\
                  <b>Entrance Restriction - Tide:</b> {entrance_restriction_tide}<br>\
                  <b>Overhead Limits:</b> {overhead_limits}<br>\
                  <b>Good Holding Ground:</b> {good_holding_ground}<br>\
                  <b>Quarantine - Pratique:</b> {quarantine_pratique}<br>\
                  <b>First Port of Entry:</b> {first_port_entry}"
    
    # Create a popup with custom width
    popup = folium.Popup(popup_text, max_width=350)
    
    # Determine the icon based on harbor type
    icon_type = icon_map_unique.get(harbor_type, 'question-sign')
    
    folium.Marker(
        [latitude, longitude], 
        popup=popup, 
        icon=folium.Icon(icon=icon_type, color='hue')
    ).add_to(marker_cluster)

# Show the extended map for categorical fields
m_categorical

# Initialize the map for quantitative fields
m_quantitative = folium.Map(location=[20, 0], zoom_start=3)
marker_cluster = MarkerCluster().add_to(m_quantitative)


# Add markers to the map based on quantitative fields
for idx, row in df.iterrows():
    latitude, longitude = row['Latitude'], row['Longitude']
    tidal_range = row['Tidal Range (m)']
    channel_depth = row['Channel Depth (m)']
    anchorage_depth = row['Anchorage Depth (m)']
    cargo_pier_depth = row['Cargo Pier Depth (m)']
    oil_terminal_depth = row['Oil Terminal Depth (m)']
    
    # Create popup text with quantitative fields
    popup_text = f"<b>Tidal Range:</b> {tidal_range} m<br>\
                  <b>Channel Depth:</b> {channel_depth} m<br>\
                  <b>Anchorage Depth:</b> {anchorage_depth} m<br>\
                  <b>Cargo Pier Depth:</b> {cargo_pier_depth} m<br>\
                  <b>Oil Terminal Depth:</b> {oil_terminal_depth} m"
    
    # Create a popup with custom width
    popup = folium.Popup(popup_text, max_width=350)
    
    # Determine the icon color based on tidal range (as an example)
    if tidal_range <= 2:
        icon_color = 'blue'
    elif 2 < tidal_range <= 5:
        icon_color = 'green'
    elif tidal_range > 5:
        icon_color = 'red'
    else:
        icon_color = 'gray'
    
    folium.Marker(
        [latitude, longitude], 
        popup=popup, 
        icon=folium.Icon(color=icon_color)
    ).add_to(marker_cluster)

# Show the map for quantitative fields
m_quantitative


# Importing Plotly library
import plotly.express as px

# Selecting relevant columns for the map
map_data = df[['Latitude', 'Longitude', 'Main Port Name', 'Country Code', 'Harbor Size', 'Channel Depth (m)', 'Anchorage Depth (m)']]

# Creating the interactive map
fig = px.scatter_geo(map_data,
                     lat='Latitude',
                     lon='Longitude',
                     hover_name='Main Port Name',
                     hover_data=['Country Code', 'Harbor Size', 'Channel Depth (m)', 'Anchorage Depth (m)'],
                     color='Harbor Size',
                     size='Channel Depth (m)',
                     color_continuous_scale='Viridis',
                     size_max=15,
                     opacity=0.7,
                     projection='natural earth',
                     template='plotly',
                     title='Enhanced Interactive Map of Ports')

# Update the layout and map style
fig.update_geos(
    showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray",
    showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="Blue",
    showrivers=True, rivercolor="Blue"
)

# Show the map
fig.show()


# Import Plotly library
import plotly.figure_factory as ff

# Calculate the correlation matrix for the facilities
correlation_matrix = df[facilities_columns].apply(lambda x: x.replace({'Yes': 1, 'No': 0, 'Unknown': None}).astype(float)).corr()

# Create the interactive heatmap
fig = ff.create_annotated_heatmap(z=correlation_matrix.values,
                                  x=list(correlation_matrix.columns),
                                  y=list(correlation_matrix.index),
                                  annotation_text=correlation_matrix.round(2).values,
                                  colorscale='Viridis')

# Update layout
fig.update_layout(title='Interactive Correlation Matrix of Facilities',
                  xaxis=dict(title='Facility'),
                  yaxis=dict(title='Facility'))

# Show figure
fig.show()

# Import libraries
import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Prepare the data
facility_options = [{'label': facility, 'value': facility} for facility in facilities_columns]
initial_facility = facilities_columns[0]
initial_map = px.scatter_geo(df[df[initial_facility] == 'Yes'],
                             lat='Latitude',
                             lon='Longitude',
                             title=f'Ports with {initial_facility}')

# Define the layout
app.layout = html.Div([
    dcc.Dropdown(
        id='facility-dropdown',
        options=facility_options,
        value=initial_facility
    ),
    dcc.Graph(
        id='facility-map',
        figure=initial_map
    )
])

# Define the callback to update the map
@app.callback(
    Output('facility-map', 'figure'),
    [Input('facility-dropdown', 'value')]
)
def update_map(selected_facility):
    filtered_df = df[df[selected_facility] == 'Yes']
    updated_map = px.scatter_geo(filtered_df,
                                 lat='Latitude',
                                 lon='Longitude',
                                 hover_name='Main Port Name',
                                 title=f'Ports with {selected_facility}',
                                 projection='natural earth',
                                 opacity=0.5)
    
    return updated_map


app.run_server(debug=True)

