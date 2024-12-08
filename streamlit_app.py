import streamlit as st
import pandas as pd
import numpy as np

st.title("Kepler Exoplanet Search Results")
st.image("pic.jpg")
st.header("Data loading, description and cleanup")
st.write("The dataset and its original description is available by the following link: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results?resource=download")
st.write("The Kepler Space Observatory is a NASA-build satellite that was launched in 2009. The telescope is dedicated to searching for exoplanets in star systems besides our own, with the ultimate goal of possibly finding other habitable planets besides our own.")
st.write("This dataset is a cumulative record of all observed Kepler \"objects of interest\" — basically, all of the approximately 10,000 exoplanet candidates Kepler has taken observations on.")
st.write("The original description of the columns can be found by the following link: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html")

path = "cumulative.csv"

try:
    data = pd.read_csv(path)
    if not data.empty:
        st.header("Data from CSV")
        st.table(data.head())
    else:
        st.error(f"Error: CSV file at {path} is empty.")
except FileNotFoundError:
    st.error(f"Error: CSV file not found at {path}")
except pd.errors.ParserError:
    st.error(f"Error: Could not parse the CSV file at {path}. Check the file format.")

# 3
# Overview of the names of the columns
st.header("Overview of the names of the columns")
ttt = ""
for t in data.columns:
    ttt += t + " "
st.write(ttt)
# 4
st.write("All the columns that include \"_err1\" or \"_err2\" in their name contain possible positive and negative errors in estimations.")
st.write("So, we exclude those columns, and will focus only on the main values")
col_to_drop = [col for col in data.columns if "_err" in col]
data = data.drop(columns=col_to_drop)
# 5
st.write("Also we will not need the following columns:")
st.write("rowid, kepid as the contain ids of the planets")
st.write("kepoi_name, kepler_name as they contain names of the planets")
st.write("koi_tce_plnt_num, koi_tce_delivname as they contain the number and the name in TCE (Threshold Crossing Event) system")
data = data.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_tce_plnt_num', 'koi_tce_delivname'])
# 6
st.write("Columns \'ra\' and \'dec\' can also be deleted because they replesent the coordinates used in the celestial coordinate system to locate the star on the sky")
data = data.drop(columns=['ra', 'dec'])

st.write("In astronomy transit (or astronomical transit) is the passage of a celestial body directly between a larger body and the observer. As viewed from a particular vantage point, the transiting body appears to move across the face of the larger body, covering a small portion of it.")
# 7
st.image("pic2.jpg")
st.write("The values in this dataset were obtained with the help of this method.")
# 8
st.write("The following columns can be dropped as they describe properties of transit estimation")
st.write("koi_time0bk - the time of the planet\'s passage through the star\'s disk (transit) in barycentric Julian date (BJD)")
st.write("koi_depth - the transit depth, expressed as a change in the brightness of the star in millionths")
st.write("koi_model_snr - signal-to-noise ratio for the transit model")
st.write("koi_impact - the impact parameter of the transit")
st.write("koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec - boolean values concerning transit estimations")
data = data.drop(columns=['koi_time0bk', 'koi_depth', 'koi_model_snr', 'koi_impact',
                          'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'])
st.write("Finally, there are three columns describing the prediction about objects being planets. \"koi_disposition\" provides the final result. \"koi_pdisposition\" provides the preliminary status of the candidate planet set by the Kepler data processing pipeline. \"koi_score\" represents the probability that the candidate is a planet (from 0 to 1).")
st.write("Of these columns, we will leave only the first one, because it contains the main results confirmed by scientists.")
# 9
st.write("Deleting columns based on the reasoning above")
data = data.drop(columns=['koi_pdisposition', 'koi_score'])
# 10
st.write("Data without unnecessary columns")
st.table(data.head())
st.header("Columns description")
st.write("The columns describe characteristics of Kepler Objects of Interest (KOIs), which are potential exoplanet candidates identified by the Kepler space telescope.")
st.write("koi_disposition — a categorical variable indicating the final classification of the KOI.")
st.write("CONFIRMED — the KOI has been confirmed as a planet.")
st.write("CANDIDATE — the KOI is a strong candidate but requires further confirmation.")
st.write("FALSE POSITIVE — the KOI has been determined not to be a planet.")
st.write("koi_period — The orbital period of the KOI (in days). This is the time it takes the object to complete one orbit around its host star.")
st.write("koi_duration — The duration of the transit (in days). This is how long the planet blocks a portion of the star\'s light as seen from Earth.")
st.write("koi_prad — The radius of the planet (in units of the radius of Earth).")
st.write("koi_teq — The equilibrium temperature of the planet's surface (in Kelvin).")
st.write("koi_insol — The stellar insolation received by the planet (in units of Earth\'s insolation). This measures the amount of energy the planet receives from its host star.")
st.write("koi_steff — The effective temperature of the surface of the host star (in Kelvin).")
st.write("koi_slogg — The base-10 logarithm of the acceleration due to gravity at the surface of the star.")
st.write("koi_srad — The radius of the host star (in units of the Sun\'s radius).")
st.write("koi_kepmag — The Kepler apparent magnitude of the host star. This is a measure of the star\'s brightness as seen from Earth. Lower values indicate brighter stars.")
st.header("Empty values processing")
st.write("Counting the number of missing values for each column")
st.write(data.isnull().sum())
# 12
st.write("as we can see, there are a few missing values in each column, so deleting the corresponding rows will not cause the loss of the main data")
data = data.dropna()
st.table(data.head())
###

# ------------------------
st.header("Selection of data with confirmed planets only")
st.write("It was noted above that the koi_disposition column contains information about whether the candidate object is a planet. If the value is CONFIRMED in this column, then the object under study is indeed a planet. We will create a DataFrame with only confirmed planets.")

st.write("Counting values of koi_disposition")
vals = data["koi_disposition"].value_counts()

#plt.figure(figsize=(6, 5))
#cmap = plt.get_cmap("copper")
#colors = cmap(np.linspace(0.3, 0.7, 3))
#plt.pie(x=vals.values, labels=vals.index, autopct='%1.2f%%', startangle=90, explode=(0, 0.1, 0), colors=colors)
#st.pyplot(plt)
st.image("s16.png")
st.write("It can be seen that there are not many confirmed planets in the entire dataset (relative to all the studied objects), but we want to work only with confirmed objects.")
st.write("Now, since the koi_disposition column contains only the CONFIRMED values, we can delete this column.")
st.write("After some rows are deleted, we need to reset indexes.")

# 17
confirmed_data = data[data["koi_disposition"] == "CONFIRMED"]
confirmed_data = confirmed_data.drop(columns=['koi_disposition'])
confirmed_data = confirmed_data.reset_index(drop=True)
st.table(confirmed_data.head())
# 18
st.header("Overview of the final dataset")
# Descriptive statistics of the dataset
st.table(confirmed_data.describe())
st.write("This table shows what statistical parameters each coulmn has. Here \"count\" represents the amount of non-empty values. \"mean\" and \"std\" stand for the mean and standard deviation of each sample. \"min\" and \"max\" indicate the minimum and the maximum values respectively. Finally, \"25%\", \"50%\" and \"75%\" display the values of Q1, Q3 quartiles and the median.")
st.image("s19.png")
# Pairplot of each 2 columns
#import seaborn as sns
# 19
#plt.figure(figsize=(8, 6))
#sns.pairplot(confirmed_data)
#st.pyplot(plt)
st.write("This set of pair plots shows how the values of each two rows are distributed with respect to each other. On the main diagonal of this matrix of plots there are histograms of each sample in the table.")
st.image("s20.png")
# 20
# Correlation matrix
#plt.figure(figsize=(8, 8))
#sns.heatmap(confirmed_data.corr(), annot=True, cmap='PRGn')
#plt.show()
st.write("The correlation table presents the results of an analysis of the relationship between each two variables.")
st.write("The correlation between the variable specified in the row and the variable specified in the column is indicated at the intersection of the row and column of such a table.")
st.write("A review of the results of the correlation table shows that we have quite a few columns correlating with each other. In just one case, the absolute value of the correlation reaches about 0.83. In two more cases, the correlation approaches the value 0.67 and 0.63. For the other columns, the correlation cannot be considered significant.")
# 21
# Boxplots for each column
#confirmed_data.plot(kind='box', subplots=True, layout=(3,3), figsize=(12, 12))
#plt.show()
st.image("s21.png")
st.write("Boxplots show the median (middle line within the box) and quartiles (lines extending from the box). The median represents the central tendency of the data.The box itself shows the interquartile range (IQR), which indicates the spread of the middle 50% of the data. Points outside the whiskers are considered outliers. These represent extreme values in the dataset. The symmetry of the boxplot can indicate whether the data is skewed.")
st.header("Outliers processing")
st.write("Here we introduce the function that returns the series without outliers.")
st.write("Here Q1 corresponds to the 25% quartile, Q3 is a 75% quartile.")
st.write("IQR is an inter-quartile range measuring the interval holding 50% of the data.")
st.write("The statistical approach recommends to consider as outliers those values that do not fit in the IQR multiplied by 1.5.")
st.write("With the help of this function we remove outliers from all columns and continue analysing the dataset.")
# 22
def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series <= (Q3 + 1.5 * IQR)) & (series >= (Q1 - 1.5 * IQR))]
# 23
st.write("A new dataset with removed outliers in each column")
df_cleaned = confirmed_data.copy()

for column in confirmed_data.columns:
    df_cleaned[column] = remove_outliers(confirmed_data[column])

# Overview of the dataset with removed outliers
st.table(df_cleaned.describe())
# 24
st.write("Counting non-empty valyes for all columns")
data_check = pd.DataFrame(df_cleaned.count(), columns=['count'])
data_check['percent'] = (1 - data_check['count'] / 2292)*100
st.table(data_check)

# 25
#plt.figure(figsize=(6, 5))
#cmap = plt.get_cmap("copper")
#colors = cmap(np.linspace(0.3, 0.7, 9))
#plt.pie(x=data_check['percent'], labels=data_check.index, colors=colors)
#plt.show()
st.image("s25.png")
st.write("The table and the diagram show which columns were the most affected by the process of removing outliers. It can be seen that the columns \'koi_period\', \'koi_prad\' and \'koi_insol\' have the highest percentages of empty values. The decision can be made to replace empty values with the median value.")
st.write("There are several approaches to handling missing values. It would be possible to delete rows with these values, but this would cause a lot of data loss. We can fill in the values with averages, but this will have a greater impact on the distribution of data. Therefore, it was decided to fill in the missing values with median values.")

# 26
# Replacing empty values with median values for chosen columns
df_cleaned['koi_period'].fillna(value=df_cleaned['koi_period'].median(), inplace=True)
df_cleaned['koi_prad'].fillna(value=df_cleaned['koi_prad'].median(), inplace=True)
df_cleaned['koi_insol'].fillna(value=df_cleaned['koi_insol'].median(), inplace=True)

# Deleting rows with emply values, because not it will not affect the data so much
df_cleaned = df_cleaned.dropna()
# Also we need to reset indexes
df_cleaned = df_cleaned.reset_index(drop=True)

st.write("Overview of the dataset after the described process")
st.table(df_cleaned.describe())

st.write("When the process of data cleaup and handing outlies is done, we can draw plots, representing the data. Here again we perform a pairplot showing the distribution of each two columns respectively, the correlation matrix and box-plots reflecting the distribution of values within each column.")
st.image("s27.png")
st.image("s28.png")
st.image("s29.png")

st.header("Data transformation")
st.write("From the description of the data we know that the column \'koi_slogg\' contains the base-10 logarithm of the acceleration due to gravity at the surface of the star. So, if we raise the base to the power of the values in the column, we will obtain the real values of the gravity of the star.")
st.write("The column 'koi_prad' contains the radius of the planet in units of the radius of Earth. From physics we know that the radius of Earth is 6378 km. So, if we multiply the values by this number, we will obtain the real radius the the planets.")
st.write("Finally, the column 'koi_srad' contains the radius of the host star in units of the Sun's radius. From physics we know that the radius of the Sun is 696230 km. So, if we multiply the values by this number, we will obtain the real radius the the stars.")
# 30
LOG_BASE = 10
EARTH_RADIUS = 6378
SUN_RADIUS = 696230

df_cleaned['gravity'] = LOG_BASE**df_cleaned['koi_slogg']
df_cleaned['planet_radius'] = df_cleaned['koi_prad'] * EARTH_RADIUS
df_cleaned['star_radius'] = df_cleaned['koi_srad'] * SUN_RADIUS

# After the transformation is done, we will not need the initial columns,
# so we delete them
df_cleaned = df_cleaned.drop(columns=['koi_slogg', 'koi_prad', 'koi_srad'])

# Overview of the transformed data
st.table(df_cleaned.head())
# -----------------------------------------------
st.header("Hypotheses")
st.header("1")
st.write("From the correlation matrix we saw that the conums \'koi_srad\' and \'koi_slogg\' had the strongest correlation.")
st.write("After the transformation of the data we now have columns \'star_radius\' and \'gravity\'.")
st.write("Let us have a closer look at their mutial distribution.")
# 31
st.image("s31.png")
st.write("It can be assumed that the data represent an inverse exponential relationship of the form:")
st.image("f3.png")
st.write("This can be transformed into the following form if we take logarithms of both parts:")
st.image("f2.png")
st.write("or just")
st.image("f1.png")
st.write("Where Ci are some constants.")
st.write("Now, we can see that if we take logarithm of the values in the column 'gravity', it will be possible to construct a linear regression model. If this model shows that there exist a linear dependency of the transformed values, it will correspond to the initial values having an inverse exponential relationship of the form described above.")
st.image("s33.png")
st.write("The graph shows that the values are substantially concentrated around a straight line with the coefficients found.")
st.write("This may indicate that the transformed data may indeed have a linear relationship.")
st.write("This, in turn, proves that the values in the columns 'gravity' and 'star_radius' may have an inverse exponential relationship. So we can consider the hypothesis confirmed.")
# -----------------------------------------------
st.header("2")
st.write("Another hypothesis can be made about the distribution the values in the columns 'koi_period' which stands for the orbital period of the planet in days and shows the time that it takes the object to complete one orbit around its host star, and 'koi_teq' which contains the data about the equilibrium temperature of the planet's surface measured in Kelvins.")
st.write("First, let us have a closer look at their mutial distribution.")
st.image("s34.png")
st.write("Here, first of all, we notice some values forming a straight line around the values of 10 in 'koi_period'. This can be explained either by the presence of certain planets in the universe that have such an orbital period and do not obey the general distribution formula, or simply by an error in the measurement of the telescope as well as its physical limitations in detecting the exact characteristics of celestial bodies.")
st.write("We can also notice a sharp border at the right end of the graph, which may be due again to the physical limitations of the telescope or the permissible field of view from the position where the telescope is located and, consequently, the inability to detect the presence of other objects with large values of the parameter in question.")
st.write("Despite the existing limitations in the capabilities of the telescope and the possible presence of anomalous planets, we can form the following hypothesis.")
st.write("It can be assumed that the data represent an inverse relationship of the form:")
st.image("f6.png")
st.write("This can be transformed into the following form if we take logarithms of both parts:")
st.image("f5.png")
st.write("or just")
st.image("f4.png")
st.write("Where Ci are some constants.")

st.write("Now, we can see that if we take logarithms of the values in the columns 'koi_period' and 'koi_teq', it will be possible to construct a linear regression model. If this model shows that there exist a linear dependency of the transformed values, it will correspond to the initial values having an inverse relationship of the form described above.")
from sklearn import linear_model
st.write("Data preparation")
x = np.array(np.log(df_cleaned['gravity'])).reshape((-1, 1))
y = np.array(df_cleaned['star_radius'])
st.write("Introducing the model")
model = linear_model.LinearRegression()
model.fit(x, y)

st.write("Obtaining the coefficients of the linear regression")
intercept = model.intercept_
slope = model.coef_[0]

st.write("intercept:")
st.write(intercept)
st.write("slope:")
st.write(slope)

st.image("s36.png")
st.write("The graph clearly demonstrates that the data points cluster tightly around a straight line with the identified coefficients. This concentration along a straight line strongly suggests that the transformed data likely exhibits a linear relationship. Consequently, this alignment supports the hypothesis that the values in the 'koi_period' and 'koi_teq' columns may indeed have an inverse relationship. Therefore, we can conclude that our assumption can be confirmed.")
# -----------------------------------------------
st.header("3")
st.write("Now, let us consider the vaues of \'planet_radius\' and \'star_radius\'.")
st.write("The histograms of the corresponding data is performed below.")
st.image("s37.png")
st.write("It can be seen that both histograms resemble a histogram of the frequencies of the normal distribution. So the first hypothesis regarding these data may be the assumption that the values in these columns are normally distributed. Further, we will test this hypothesis.")
st.write("Let us try to estimate the parameters of the normal distribution and display the normal curve with the specified parameters on a joint graph adjusted by the appropriate height factor due to the scale of the data.")
st.write("To estimate the distribution parameters, we will use two different approaches, each of which is applicable to one of the two data series.")
st.write("The first approach is based on data grouping. We will group the data into 25 rows with the same interval length. We will specify the left and right boundaries of the interval, as well as the value that is the center of the interval. Then we will count the number of values within each interval, then normalize it by the total number of values to get a polygon of interval frequencies. Next, we calculate the mean value as a weighted average over all intervals, as well as the standard deviation as the square root of the sum of the average quadratic deviations of each value in the middle of the interval from the mean. This will give us the estimated values of the mean and standard deviation of the normal distribution, which will be displayed on the graph.")
st.write("In the second approach, we estimate the mean and standard deviation over the entire data series using built-in functions. Let's build the appropriate diagrams and study the results.")
st.write("Implementing the first approach to the column \'planet_radius\'")
st.write("Celecting the number of intervals")
st.write("num_intervals = 25")
num_intervals = 25
st.write("Calculating the length of the interval")
interval_width = np.max(df_cleaned['planet_radius']) - np.min(df_cleaned['planet_radius']) / num_intervals
st.write("Creating left and right boundaries")
bins = np.linspace(np.min(df_cleaned['planet_radius']), np.max(df_cleaned['planet_radius']), num_intervals + 1)
st.write("Grouping the data by intervals and count the number of values in each interval")
grouped_data_1 = df_cleaned['planet_radius'].value_counts(bins=bins).sort_index()
st.write("Creating the DataFrame with results")
result_df_1 = pd.DataFrame({
    'Left': grouped_data_1.index.left,
    'Right': grouped_data_1.index.right,
    'Count': grouped_data_1.values
})
st.write("Adding a column with adjusted numbers within each intervel")
result_df_1['Adj'] = result_df_1['Count']/sum(result_df_1['Count'])
st.write("Calculating midpoints of each interval")
result_df_1['Midpoint'] = (result_df_1['Left'] + result_df_1['Right'])/2
st.write("Calculating the mean")
mean_planet = sum(result_df_1['Midpoint'] * result_df_1['Adj'])
st.write("Mean =")
st.write(mean_planet)
st.write("Calculating the standard deviation")
std_planet = (sum((result_df_1['Midpoint']-mean_planet)**2 * result_df_1['Adj']))**0.5
st.write("Standard deviation = ")
st.write(std_planet)
st.image("s39.png")
st.image("s40.png")
st.write("From the graphs obtained, it can already be seen that the available data does not fit well into the density graph of the normal distribution. Moreover, this is typical for both data series, regardless of which method was chosen to estimate the distribution parameters.")
st.write("Nevertheless, we will try to test our hypothesis with the help of a mathematical apparatus, namely Shapiro–Wilk test (https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test) and D'Agostino's K-squared test (https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test). For both tests we will be using built-in methods from the scipy module.")
st.write("In the final data verification, we used the criterion that if at least one of the tests gave a positive result, then we conclude that the data could have a normal distribution.")
st.write("As can be seen from the results, both tests give a negative result for both data series at a given level of accuracy. From this it can be concluded that the hypothesis of the normality of the data cannot be confirmed, despite the initial similarity of the histograms of the data series with the polygon of the frequencies of the normal distribution. From all this, we conclude that this hypothesis has been disproved.")
# -----------------------------------------------
st.header("Conclusion")
st.write("This analysis examined the Kepler Exoplanet Search Results dataset. Data cleaning was performed, addressing missing values through filling and removal as appropriate. Descriptive statistics, including mean, standard deviation, median, quartiles, minimum, and maximum values, were calculated to summarize the dataset's characteristics. Visualizations were generated to provide an overall understanding of the data, with further investigation and analysis focused on specific subsets of interest. Three hypotheses were formulated and tested, two were supported by the analysis, while one was rejected.")

# ****************************

df = df_cleaned
with st.form("search_form"):
    search_min = st.number_input("Min Gravity:")
    search_max = st.number_input("Max Gravity:")
    submitted = st.form_submit_button("Search")
if submitted:
    if search_min and search_max:
        result = df[(df['gravity'] > search_min) & (df['gravity'] < search_max)]
        if not result.empty:
            st.table(result)
        else:
            st.write("No matching entries found.")
    elif search_min:
        result = df[(df['gravity'] > search_min)]
        if not result.empty:
            st.table(result)
        else:
            st.write("No matching entries found.")
    elif search_max:
        result = df[(df['gravity'] < search_max)]
        if not result.empty:
            st.table(result)
        else:
            st.write("No matching entries found.")
    else:
        st.write("Please enter at least one search criterion.")
    st.dataframe(df)

# ****************************

with st.form("add_record_form"):
    koi_period = st.number_input("Koi Period:")
    koi_duration = st.number_input("Koi Duration:")
    koi_teq = st.number_input("Koi Teq:")
    koi_insol = st.number_input("Koi Insol:")
    koi_steff = st.number_input("Koi Steff:")
    koi_kepmag = st.number_input("Koi Kepmag:")
    gravity = st.number_input("Gravity:")
    planet_radius = st.number_input("Planet Radius:")
    star_radius = st.number_input("Star Radius:")
    
    submitted = st.form_submit_button("Submit")
if submitted:
    new_record = {
        'koi_period': koi_period,
        'koi_duration': koi_duration,
        'koi_teq': koi_teq,
        'koi_insol': koi_insol,
        'koi_steff': koi_steff,
        'koi_kepmag': koi_kepmag,
        'gravity': gravity,
        'planet_radius': planet_radius,
        'star_radius': star_radius
    }
    df.loc[len(df)] = new_record
    st.success("Запись успешно добавлена!")
