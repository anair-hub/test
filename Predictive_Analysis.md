```python
# Importing required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import hvplot.pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
```










```python
# Read the table
data=pd.read_csv("../Resources/Project_Indicators_Wide.csv")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_code</th>
      <th>country</th>
      <th>region</th>
      <th>status</th>
      <th>year</th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>...</th>
      <th>HDI</th>
      <th>HE</th>
      <th>INCI</th>
      <th>INFMR</th>
      <th>LE</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABW</td>
      <td>Aruba</td>
      <td>Latin America &amp; Caribbean</td>
      <td>NaN</td>
      <td>2000</td>
      <td>NaN</td>
      <td>112.4760</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.787</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>504.811111</td>
      <td>2.064841</td>
      <td>93.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABW</td>
      <td>Aruba</td>
      <td>Latin America &amp; Caribbean</td>
      <td>NaN</td>
      <td>2001</td>
      <td>NaN</td>
      <td>111.9155</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.853</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>516.066667</td>
      <td>2.205163</td>
      <td>93.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABW</td>
      <td>Aruba</td>
      <td>Latin America &amp; Caribbean</td>
      <td>NaN</td>
      <td>2002</td>
      <td>NaN</td>
      <td>111.3550</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.937</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>527.733333</td>
      <td>2.235515</td>
      <td>93.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABW</td>
      <td>Aruba</td>
      <td>Latin America &amp; Caribbean</td>
      <td>NaN</td>
      <td>2003</td>
      <td>NaN</td>
      <td>109.9290</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.038</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>538.977778</td>
      <td>2.108324</td>
      <td>93.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABW</td>
      <td>Aruba</td>
      <td>Latin America &amp; Caribbean</td>
      <td>NaN</td>
      <td>2004</td>
      <td>NaN</td>
      <td>108.5030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.156</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>548.577778</td>
      <td>1.765473</td>
      <td>93.3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# Dropping NA's
df=data.dropna()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_code</th>
      <th>country</th>
      <th>region</th>
      <th>status</th>
      <th>year</th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>...</th>
      <th>HDI</th>
      <th>HE</th>
      <th>INCI</th>
      <th>INFMR</th>
      <th>LE</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2000</td>
      <td>0.002667</td>
      <td>310.8305</td>
      <td>30.0</td>
      <td>0.502297</td>
      <td>48.0</td>
      <td>...</td>
      <td>0.350</td>
      <td>9.0</td>
      <td>0.333</td>
      <td>90.5</td>
      <td>55.841</td>
      <td>27.0</td>
      <td>2.3</td>
      <td>31.829117</td>
      <td>2.975057</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2001</td>
      <td>0.005333</td>
      <td>304.8580</td>
      <td>43.0</td>
      <td>0.508536</td>
      <td>59.0</td>
      <td>...</td>
      <td>0.353</td>
      <td>9.0</td>
      <td>0.318</td>
      <td>87.9</td>
      <td>56.308</td>
      <td>37.0</td>
      <td>2.4</td>
      <td>33.095904</td>
      <td>3.902805</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2002</td>
      <td>0.008000</td>
      <td>298.8855</td>
      <td>46.0</td>
      <td>0.512110</td>
      <td>62.0</td>
      <td>...</td>
      <td>0.384</td>
      <td>9.0</td>
      <td>0.386</td>
      <td>85.3</td>
      <td>56.784</td>
      <td>35.0</td>
      <td>2.6</td>
      <td>34.618102</td>
      <td>4.496719</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2003</td>
      <td>0.010667</td>
      <td>292.0365</td>
      <td>44.0</td>
      <td>0.515965</td>
      <td>66.0</td>
      <td>...</td>
      <td>0.393</td>
      <td>9.0</td>
      <td>0.391</td>
      <td>82.7</td>
      <td>57.271</td>
      <td>39.0</td>
      <td>2.7</td>
      <td>36.272510</td>
      <td>4.668344</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2004</td>
      <td>0.013333</td>
      <td>285.1880</td>
      <td>51.0</td>
      <td>0.520604</td>
      <td>72.0</td>
      <td>...</td>
      <td>0.409</td>
      <td>10.0</td>
      <td>0.389</td>
      <td>80.0</td>
      <td>57.772</td>
      <td>48.0</td>
      <td>2.9</td>
      <td>37.874413</td>
      <td>4.321560</td>
      <td>105.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
df.dtypes
```




    country_code     object
    country          object
    region           object
    status           object
    year              int64
    ALC             float64
    AMR             float64
    BCG             float64
    CANP            float64
    DTP             float64
    EDI             float64
    GDP             float64
    GDPG            float64
    HDI             float64
    HE              float64
    INCI            float64
    INFMR           float64
    LE              float64
    MCV             float64
    OBP             float64
    POPD            float64
    POPG            float64
    SR              float64
    dtype: object




```python
# Dropping unwanted columns
select_df=df.drop(columns=["country_code","country","year","status"])
select_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>region</th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>EDI</th>
      <th>GDP</th>
      <th>GDPG</th>
      <th>HDI</th>
      <th>HE</th>
      <th>INCI</th>
      <th>INFMR</th>
      <th>LE</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>South Asia</td>
      <td>0.002667</td>
      <td>310.8305</td>
      <td>30.0</td>
      <td>0.502297</td>
      <td>48.0</td>
      <td>0.235</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.350</td>
      <td>9.0</td>
      <td>0.333</td>
      <td>90.5</td>
      <td>55.841</td>
      <td>27.0</td>
      <td>2.3</td>
      <td>31.829117</td>
      <td>2.975057</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>21</th>
      <td>South Asia</td>
      <td>0.005333</td>
      <td>304.8580</td>
      <td>43.0</td>
      <td>0.508536</td>
      <td>59.0</td>
      <td>0.247</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.353</td>
      <td>9.0</td>
      <td>0.318</td>
      <td>87.9</td>
      <td>56.308</td>
      <td>37.0</td>
      <td>2.4</td>
      <td>33.095904</td>
      <td>3.902805</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>22</th>
      <td>South Asia</td>
      <td>0.008000</td>
      <td>298.8855</td>
      <td>46.0</td>
      <td>0.512110</td>
      <td>62.0</td>
      <td>0.259</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.384</td>
      <td>9.0</td>
      <td>0.386</td>
      <td>85.3</td>
      <td>56.784</td>
      <td>35.0</td>
      <td>2.6</td>
      <td>34.618102</td>
      <td>4.496719</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>23</th>
      <td>South Asia</td>
      <td>0.010667</td>
      <td>292.0365</td>
      <td>44.0</td>
      <td>0.515965</td>
      <td>66.0</td>
      <td>0.271</td>
      <td>4.515559e+09</td>
      <td>3.868380</td>
      <td>0.393</td>
      <td>9.0</td>
      <td>0.391</td>
      <td>82.7</td>
      <td>57.271</td>
      <td>39.0</td>
      <td>2.7</td>
      <td>36.272510</td>
      <td>4.668344</td>
      <td>105.9</td>
    </tr>
    <tr>
      <th>24</th>
      <td>South Asia</td>
      <td>0.013333</td>
      <td>285.1880</td>
      <td>51.0</td>
      <td>0.520604</td>
      <td>72.0</td>
      <td>0.302</td>
      <td>5.226779e+09</td>
      <td>-2.875203</td>
      <td>0.409</td>
      <td>10.0</td>
      <td>0.389</td>
      <td>80.0</td>
      <td>57.772</td>
      <td>48.0</td>
      <td>2.9</td>
      <td>37.874413</td>
      <td>4.321560</td>
      <td>105.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scatterplots comparing features with life expectancy
fig, axes = plt.subplots(6, 3, figsize=(15, 15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for x,column in enumerate(select_df.columns.difference(['region'])):
    i=math.floor(x/3)
    j=x%3
    plot=sns.scatterplot(ax=axes[i,j],x=select_df['LE'], y= select_df[column],data=select_df,hue=select_df['region'])
    handles, labels = axes[0,0].get_legend_handles_labels()  
    axes[i,j].get_legend().remove()
fig.legend(handles, labels, loc = 'upper left',title="Regions")
plt.savefig("../Images/Plots/Scatter_plot.png")
```


    
![png](output_5_0.png)
    



```python
# Distribution plot for life expectancy
fig, ax = plt.subplots(figsize=(15, 15))
sns.histplot(data=select_df,x='LE',hue='region',element='step',stat='density',kde=True)
ax.set_xlabel("Life Expectancy")
sns.move_legend(ax,'upper left', title="Regions")
plt.savefig("../Images/Plots/Distribution_plot.png")
```


    
![png](output_6_0.png)
    



```python
# Correlation matrix
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(select_df.corr(),cmap='coolwarm',annot=True)
plt.savefig("../Images/Plots/Correlation_matrix.png")
```


    
![png](output_7_0.png)
    


<a id='Multiple Linear Regression'></a>
 

[Multiple_Linear_Regression](#Multiple_Linear_Regression)

<a id='Multiple_Linear_Regression'></a>

## Multiple Linear Regression 

### Model to predict life expectancy


```python
# Defining features and target
y=select_df["LE"]
X=select_df.drop(columns=["LE","region","ALC","BCG","DTP","GDPG","POPD","SR"])
```


```python
# Splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
```


```python
# Normalize the data
X_scaler=MinMaxScaler().fit(X_train)
X_train_scaled=X_scaler.transform(X_train)
X_test_scaled=X_scaler.transform(X_test)
```


```python
# Fitting the model
from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(X_train_scaled,y_train)
```




    LinearRegression()




```python
# Predicting the life expectancy values
y_pred=lm.predict(X_test_scaled)
```


```python
# Performance of the model

from sklearn.metrics import r2_score,mean_squared_error

print (f"The coefficient of determination is {r2_score(y_test,y_pred):.3f}")
print (f"The mean squared error is {mean_squared_error(y_test,y_pred):.3f}")
```

    The coefficient of determination is 0.995
    The mean squared error is 0.462
    


```python
# Training and Testing scores

training_score=lm.score(X_train_scaled,y_train)
testing_score=lm.score(X_test_scaled,y_test)

print(f"Training Score: {training_score:.3f}")
print(f"Testing Score: {testing_score:.3f}")
```

    Training Score: 0.994
    Testing Score: 0.995
    


```python
# Plotting the residuals
plt.scatter(y_pred,y_pred-y_test)
plt.hlines(y=0,xmin=y_pred.min(),xmax=y_pred.max())
plt.xlabel("Predicted value")
plt.ylabel("Residuals")
```




    Text(0, 0.5, 'Residuals')




    
![png](output_19_1.png)
    



```python
# Calculating the intercept

print(f"The intercept is {lm.intercept_:.3f}")
```

    The intercept is 58.169
    


```python
# Calculating the coefficients

print("The coefficients are:")

for item in zip(X.columns,lm.coef_):
    print(item)
```

    The coefficients are:
    ('AMR', -8.195918557413135)
    ('CANP', 0.385422279500121)
    ('EDI', -45.67680881174396)
    ('GDP', -0.1613350050110072)
    ('HDI', 99.66561814157986)
    ('HE', 0.6816497256898723)
    ('INCI', -34.357711285092385)
    ('INFMR', -5.0999163854464165)
    ('MCV', 0.13247233027562352)
    ('OBP', -0.47133514317458614)
    ('POPG', 0.4122633830283033)
    

## Logistic Regression

### Model to predict country's status


```python
# Defining feature and target
y_s=df["status"]
X_s=df.drop(columns=["country","year","status","region","country_code"])
```


```python
# Splitting the data
X_s_train,X_s_test,y_s_train,y_s_test=train_test_split(X_s,y_s,random_state=42)
```


```python
# Standardizing the data
X_scaler=MinMaxScaler().fit(X_s_train)
X_s_train_scaled=X_scaler.transform(X_s_train)
X_s_test_scaled=X_scaler.transform(X_s_test)
```


```python
# Fitting the model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs',random_state=42)
lr.fit(X_s_train_scaled,y_s_train)
```




    LogisticRegression(random_state=42)




```python
predicted_status=lr.predict(X_s_test_scaled)
```


```python
print(f"The accuracy of the model is : {accuracy_score(y_s_test,predicted_status):.3f}")
```

    The accuracy of the model is : 0.950
    


```python
print("The confusion matrix is as follows:")
print(confusion_matrix(y_s_test,predicted_status))
```

    The confusion matrix is as follows:
    [[ 46  10]
     [ 18 481]]
    


```python
print("The classification report is as follows:")
print(classification_report(y_s_test,predicted_status))
```

    The classification report is as follows:
                  precision    recall  f1-score   support
    
       Developed       0.72      0.82      0.77        56
      Developing       0.98      0.96      0.97       499
    
        accuracy                           0.95       555
       macro avg       0.85      0.89      0.87       555
    weighted avg       0.95      0.95      0.95       555
    
    

## Random Forest

### Model to predict country's status


```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_s_train_scaled,y_s_train)
```




    RandomForestClassifier(random_state=42)




```python
predicted_status_rf=rfc.predict(X_s_test_scaled)
```


```python
print(f"The accuracy of the model is : {accuracy_score(y_s_test,predicted_status_rf):.3f}")
```

    The accuracy of the model is : 1.000
    


```python
print("The confusion matrix is as follows:")
print(confusion_matrix(y_s_test,predicted_status_rf))
```

    The confusion matrix is as follows:
    [[ 56   0]
     [  0 499]]
    


```python
print("The classification report is as follows:")
print(classification_report(y_s_test,predicted_status_rf))
```

    The classification report is as follows:
                  precision    recall  f1-score   support
    
       Developed       1.00      1.00      1.00        56
      Developing       1.00      1.00      1.00       499
    
        accuracy                           1.00       555
       macro avg       1.00      1.00      1.00       555
    weighted avg       1.00      1.00      1.00       555
    
    


```python
# Feature importance
sort_list=sorted(zip(X_s.columns,rfc.feature_importances_),key=lambda x: x[1],reverse=True)
labels, values = zip(*sort_list)
indexes = np.arange(len(labels))
width = 1
fig, ax = plt.subplots(figsize=(12, 12))
plt.bar(indexes,values,linewidth=1,tick_label=labels,edgecolor="black")
ax.set(title="Feature Importance",xlabel="Feature",ylabel="Importance")
plt.savefig("../Images/Plots/Bar_Plot.png")
```


    
![png](output_37_0.png)
    


## KMeans Clustering

### Studying clusters in data


```python
# Reindexing dataframe
df=df.reset_index(drop=True)
```


```python
# Tranforming country status
onc=OneHotEncoder(sparse=False)

encoded_df=pd.DataFrame(onc.fit_transform(df["status"].values.reshape(-1,1)))
encoded_df.columns=onc.get_feature_names_out(["status"])
encoded_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>status_Developed</th>
      <th>status_Developing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2212</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2213</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2214</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2215</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2216</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merging the dataframes
cluster_df=df.join(encoded_df)
cluster_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_code</th>
      <th>country</th>
      <th>region</th>
      <th>status</th>
      <th>year</th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>...</th>
      <th>INCI</th>
      <th>INFMR</th>
      <th>LE</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
      <th>status_Developed</th>
      <th>status_Developing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2000</td>
      <td>0.002667</td>
      <td>310.8305</td>
      <td>30.0</td>
      <td>0.502297</td>
      <td>48.0</td>
      <td>...</td>
      <td>0.333</td>
      <td>90.5</td>
      <td>55.841</td>
      <td>27.0</td>
      <td>2.3</td>
      <td>31.829117</td>
      <td>2.975057</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2001</td>
      <td>0.005333</td>
      <td>304.8580</td>
      <td>43.0</td>
      <td>0.508536</td>
      <td>59.0</td>
      <td>...</td>
      <td>0.318</td>
      <td>87.9</td>
      <td>56.308</td>
      <td>37.0</td>
      <td>2.4</td>
      <td>33.095904</td>
      <td>3.902805</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2002</td>
      <td>0.008000</td>
      <td>298.8855</td>
      <td>46.0</td>
      <td>0.512110</td>
      <td>62.0</td>
      <td>...</td>
      <td>0.386</td>
      <td>85.3</td>
      <td>56.784</td>
      <td>35.0</td>
      <td>2.6</td>
      <td>34.618102</td>
      <td>4.496719</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2003</td>
      <td>0.010667</td>
      <td>292.0365</td>
      <td>44.0</td>
      <td>0.515965</td>
      <td>66.0</td>
      <td>...</td>
      <td>0.391</td>
      <td>82.7</td>
      <td>57.271</td>
      <td>39.0</td>
      <td>2.7</td>
      <td>36.272510</td>
      <td>4.668344</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2004</td>
      <td>0.013333</td>
      <td>285.1880</td>
      <td>51.0</td>
      <td>0.520604</td>
      <td>72.0</td>
      <td>...</td>
      <td>0.389</td>
      <td>80.0</td>
      <td>57.772</td>
      <td>48.0</td>
      <td>2.9</td>
      <td>37.874413</td>
      <td>4.321560</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
# Dropping unwanted columns

cluster_df.drop(columns=["country","year","status","country_code","region"],inplace=True)
cluster_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>EDI</th>
      <th>GDP</th>
      <th>GDPG</th>
      <th>HDI</th>
      <th>HE</th>
      <th>INCI</th>
      <th>INFMR</th>
      <th>LE</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
      <th>status_Developed</th>
      <th>status_Developing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002667</td>
      <td>310.8305</td>
      <td>30.0</td>
      <td>0.502297</td>
      <td>48.0</td>
      <td>0.235</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.350</td>
      <td>9.0</td>
      <td>0.333</td>
      <td>90.5</td>
      <td>55.841</td>
      <td>27.0</td>
      <td>2.3</td>
      <td>31.829117</td>
      <td>2.975057</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.005333</td>
      <td>304.8580</td>
      <td>43.0</td>
      <td>0.508536</td>
      <td>59.0</td>
      <td>0.247</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.353</td>
      <td>9.0</td>
      <td>0.318</td>
      <td>87.9</td>
      <td>56.308</td>
      <td>37.0</td>
      <td>2.4</td>
      <td>33.095904</td>
      <td>3.902805</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.008000</td>
      <td>298.8855</td>
      <td>46.0</td>
      <td>0.512110</td>
      <td>62.0</td>
      <td>0.259</td>
      <td>4.055180e+09</td>
      <td>3.868380</td>
      <td>0.384</td>
      <td>9.0</td>
      <td>0.386</td>
      <td>85.3</td>
      <td>56.784</td>
      <td>35.0</td>
      <td>2.6</td>
      <td>34.618102</td>
      <td>4.496719</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.010667</td>
      <td>292.0365</td>
      <td>44.0</td>
      <td>0.515965</td>
      <td>66.0</td>
      <td>0.271</td>
      <td>4.515559e+09</td>
      <td>3.868380</td>
      <td>0.393</td>
      <td>9.0</td>
      <td>0.391</td>
      <td>82.7</td>
      <td>57.271</td>
      <td>39.0</td>
      <td>2.7</td>
      <td>36.272510</td>
      <td>4.668344</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.013333</td>
      <td>285.1880</td>
      <td>51.0</td>
      <td>0.520604</td>
      <td>72.0</td>
      <td>0.302</td>
      <td>5.226779e+09</td>
      <td>-2.875203</td>
      <td>0.409</td>
      <td>10.0</td>
      <td>0.389</td>
      <td>80.0</td>
      <td>57.772</td>
      <td>48.0</td>
      <td>2.9</td>
      <td>37.874413</td>
      <td>4.321560</td>
      <td>105.9</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cluster_df.dtypes
```




    ALC                  float64
    AMR                  float64
    BCG                  float64
    CANP                 float64
    DTP                  float64
    EDI                  float64
    GDP                  float64
    GDPG                 float64
    HDI                  float64
    HE                   float64
    INCI                 float64
    INFMR                float64
    LE                   float64
    MCV                  float64
    OBP                  float64
    POPD                 float64
    POPG                 float64
    SR                   float64
    status_Developed     float64
    status_Developing    float64
    dtype: object




```python
# Scaling the data
cluster_scaled=MinMaxScaler().fit_transform(cluster_df)
```


```python
# Applying PCA to reduce dimensions from 21 to 4

from sklearn.decomposition import PCA

pca=PCA(n_components=4)
cluster_pca=pca.fit_transform(cluster_scaled)
```


```python
# Checking the explained variance ratio - Variables will be adjusted to use fewer parameters in next iteration
pca.explained_variance_ratio_
```




    array([0.53456714, 0.20126763, 0.0589989 , 0.05359823])




```python
# Adding pca values to df

pca_df=pd.DataFrame(cluster_pca,columns=["PC1","PC2","PC3","PC4"])
```


```python
# Identifying best value for k

from sklearn.cluster import KMeans

inertia=[]
k = list(range(1,11))

# Looping through definined k options
for value in k:
    km = KMeans(n_clusters=value,random_state=42)
    km.fit(pca_df)
    inertia.append(km.inertia_)
    
# Elbow curve
elbow_df=pd.DataFrame({"k":k,"inertia":inertia})

elbow_df.plot(x="k",y="inertia",xlabel="Number of clusters",ylabel="Inertia")
```

    C:\Users\DCANOWERKCOMPUTA\anaconda3\envs\mlenv\lib\site-packages\sklearn\cluster\_kmeans.py:1037: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=9.
      "KMeans is known to have a memory leak on Windows "
    




    <AxesSubplot:xlabel='Number of clusters', ylabel='Inertia'>




    
![png](output_48_2.png)
    



```python
# KMeans Clustering

kmodel=KMeans(n_clusters=3,random_state=42)
kmodel.fit(pca_df)
```




    KMeans(n_clusters=3, random_state=42)




```python
# Predicting the cluster

cluster=kmodel.predict(pca_df)

pca_df["class"]=kmodel.labels_
pca_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.128039</td>
      <td>0.714938</td>
      <td>0.684560</td>
      <td>0.105317</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.053151</td>
      <td>0.611978</td>
      <td>0.480666</td>
      <td>0.029218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.988180</td>
      <td>0.566578</td>
      <td>0.493590</td>
      <td>0.035084</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.946992</td>
      <td>0.532654</td>
      <td>0.463901</td>
      <td>0.020235</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.867272</td>
      <td>0.453745</td>
      <td>0.333850</td>
      <td>-0.006954</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_df=df.join(pca_df)
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_code</th>
      <th>country</th>
      <th>region</th>
      <th>status</th>
      <th>year</th>
      <th>ALC</th>
      <th>AMR</th>
      <th>BCG</th>
      <th>CANP</th>
      <th>DTP</th>
      <th>...</th>
      <th>MCV</th>
      <th>OBP</th>
      <th>POPD</th>
      <th>POPG</th>
      <th>SR</th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2000</td>
      <td>0.002667</td>
      <td>310.8305</td>
      <td>30.0</td>
      <td>0.502297</td>
      <td>48.0</td>
      <td>...</td>
      <td>27.0</td>
      <td>2.3</td>
      <td>31.829117</td>
      <td>2.975057</td>
      <td>105.9</td>
      <td>-1.128039</td>
      <td>0.714938</td>
      <td>0.684560</td>
      <td>0.105317</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2001</td>
      <td>0.005333</td>
      <td>304.8580</td>
      <td>43.0</td>
      <td>0.508536</td>
      <td>59.0</td>
      <td>...</td>
      <td>37.0</td>
      <td>2.4</td>
      <td>33.095904</td>
      <td>3.902805</td>
      <td>105.9</td>
      <td>-1.053151</td>
      <td>0.611978</td>
      <td>0.480666</td>
      <td>0.029218</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2002</td>
      <td>0.008000</td>
      <td>298.8855</td>
      <td>46.0</td>
      <td>0.512110</td>
      <td>62.0</td>
      <td>...</td>
      <td>35.0</td>
      <td>2.6</td>
      <td>34.618102</td>
      <td>4.496719</td>
      <td>105.9</td>
      <td>-0.988180</td>
      <td>0.566578</td>
      <td>0.493590</td>
      <td>0.035084</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2003</td>
      <td>0.010667</td>
      <td>292.0365</td>
      <td>44.0</td>
      <td>0.515965</td>
      <td>66.0</td>
      <td>...</td>
      <td>39.0</td>
      <td>2.7</td>
      <td>36.272510</td>
      <td>4.668344</td>
      <td>105.9</td>
      <td>-0.946992</td>
      <td>0.532654</td>
      <td>0.463901</td>
      <td>0.020235</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AFG</td>
      <td>Afghanistan</td>
      <td>South Asia</td>
      <td>Developing</td>
      <td>2004</td>
      <td>0.013333</td>
      <td>285.1880</td>
      <td>51.0</td>
      <td>0.520604</td>
      <td>72.0</td>
      <td>...</td>
      <td>48.0</td>
      <td>2.9</td>
      <td>37.874413</td>
      <td>4.321560</td>
      <td>105.9</td>
      <td>-0.867272</td>
      <td>0.453745</td>
      <td>0.333850</td>
      <td>-0.006954</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
combined_df.hvplot.scatter(x="PC1",y="PC2",by="class",hover_cols=["country","year"])

```






<div id='1002'>





  <div class="bk-root" id="033f02ec-2c86-43a0-a5a1-79509a9f808d" data-root-id="1002"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"5bff0b63-e17f-4000-96f5-441681c55833":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1069","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1087","type":"Scatter"},{"attributes":{},"id":"1041","type":"AllLabels"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"1072"}]},"id":"1086","type":"LegendItem"},{"attributes":{"axis":{"id":"1018"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1021","type":"Grid"},{"attributes":{"axis":{"id":"1022"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1025","type":"Grid"},{"attributes":{"data":{"PC1":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[797]},"PC2":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[797]},"class":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"country":["Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Afghanistan","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Angola","Azerbaijan","Azerbaijan","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Burundi","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Benin","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Burkina Faso","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","Central African Republic","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","C\u00f4te d\u2019Ivoire","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Cameroon","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Kinshasa","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Congo - Brazzaville","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Comoros","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Ethiopia","Micronesia (Federated States of)","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Ghana","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Guinea","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Gambia","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Guinea-Bissau","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Equatorial Guinea","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Haiti","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","India","India","India","India","India","India","India","India","India","India","India","India","India","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Kenya","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Cambodia","Kiribati","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Laos","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Liberia","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Lesotho","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Madagascar","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Mali","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Myanmar (Burma)","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mozambique","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Mauritania","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Malawi","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Namibia","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Niger","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nigeria","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Nepal","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Pakistan","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Papua New Guinea","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Rwanda","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Senegal","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","Sierra Leone","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Eswatini","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Chad","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Togo","Tajikistan","Tajikistan","Tajikistan","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Timor-Leste","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Tanzania","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Uganda","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia","Zambia"],"year":[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2000,2001,2002,2003,2004,2005,2006,2007,2008,2000,2001,2002,2003,2004,2005,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2004,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2008,2009,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2000,2001,2002,2003,2004,2005,2006,2007,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]},"selected":{"id":"1046"},"selection_policy":{"id":"1060"}},"id":"1045","type":"ColumnDataSource"},{"attributes":{},"id":"1060","type":"UnionRenderers"},{"attributes":{"data":{"PC1":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1187]},"PC2":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1187]},"class":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"country":["Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","Albania","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","United Arab Emirates","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Argentina","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Armenia","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Azerbaijan","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bangladesh","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Bosnia & Herzegovina","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belarus","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Belize","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Bolivia","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brazil","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Brunei","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Bhutan","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Botswana","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","Chile","China","China","China","China","China","China","China","China","China","China","China","China","China","China","China","China","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Colombia","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Cape Verde","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Costa Rica","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Cuba","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Dominican Republic","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Algeria","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Ecuador","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Egypt","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Estonia","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Finland","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","Fiji","France","France","France","France","France","France","France","France","France","France","France","France","France","France","France","France","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Micronesia (Federated States of)","Gabon","Gabon","Gabon","Gabon","Gabon","Gabon","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Georgia","Ghana","Ghana","Ghana","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guatemala","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Guyana","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Honduras","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","Indonesia","India","India","India","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iran","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Iraq","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jamaica","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Jordan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kazakhstan","Kenya","Kenya","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Kyrgyzstan","Cambodia","Cambodia","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","Kiribati","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","South Korea","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","Kuwait","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","St. Lucia","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Sri Lanka","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Morocco","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Moldova","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Maldives","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","Mexico","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","North Macedonia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mongolia","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Mauritius","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Malaysia","Namibia","Namibia","Namibia","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nicaragua","Nepal","Nepal","Nepal","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Oman","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Panama","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Peru","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Philippines","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Paraguay","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Qatar","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Saudi Arabia","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","Solomon Islands","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","El Salvador","Serbia","Serbia","Serbia","Serbia","Serbia","Serbia","Serbia","Serbia","Serbia","Serbia","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","S\u00e3o Tom\u00e9 & Pr\u00edncipe","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Seychelles","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Thailand","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tajikistan","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tonga","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Tunisia","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Turkey","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uruguay","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","Uzbekistan","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","St. Vincent & Grenadines","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vietnam","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Vanuatu","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","Samoa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa","South Africa"],"year":[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2009,2010,2011,2012,2013,2014,2015,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2014,2015,2000,2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2006,2007,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2008,2009,2010,2011,2012,2013,2014,2015]},"selected":{"id":"1067"},"selection_policy":{"id":"1083"}},"id":"1066","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1071","type":"Scatter"},{"attributes":{"children":[{"id":"1003"},{"id":"1009"},{"id":"1238"}],"margin":[0,0,0,0],"name":"Row01706","tags":["embedded"]},"id":"1002","type":"Row"},{"attributes":{},"id":"1014","type":"LinearScale"},{"attributes":{"tools":[{"id":"1006"},{"id":"1026"},{"id":"1027"},{"id":"1028"},{"id":"1029"},{"id":"1030"}]},"id":"1032","type":"Toolbar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1092","type":"Scatter"},{"attributes":{"end":1.0252818441705425,"reset_end":1.0252818441705425,"reset_start":-0.8159750645278472,"start":-0.8159750645278472,"tags":[[["PC2","PC2",null]]]},"id":"1005","type":"Range1d"},{"attributes":{"coordinates":null,"data_source":{"id":"1088"},"glyph":{"id":"1091"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1093"},"nonselection_glyph":{"id":"1092"},"selection_glyph":{"id":"1111"},"view":{"id":"1095"}},"id":"1094","type":"GlyphRenderer"},{"attributes":{"callback":null,"renderers":[{"id":"1051"},{"id":"1072"},{"id":"1094"}],"tags":["hv_created"],"tooltips":[["class","@{class}"],["PC1","@{PC1}"],["PC2","@{PC2}"],["country","@{country}"],["year","@{year}"]]},"id":"1006","type":"HoverTool"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1031","type":"BoxAnnotation"},{"attributes":{},"id":"1026","type":"SaveTool"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1111","type":"Scatter"},{"attributes":{},"id":"1027","type":"PanTool"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01710","sizing_mode":"stretch_width"},"id":"1003","type":"Spacer"},{"attributes":{},"id":"1030","type":"ResetTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1050","type":"Scatter"},{"attributes":{},"id":"1028","type":"WheelZoomTool"},{"attributes":{},"id":"1044","type":"AllLabels"},{"attributes":{},"id":"1046","type":"Selection"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"1064"},{"id":"1086"},{"id":"1110"}],"location":[0,0],"title":"class"},"id":"1063","type":"Legend"},{"attributes":{"overlay":{"id":"1031"}},"id":"1029","type":"BoxZoomTool"},{"attributes":{},"id":"1023","type":"BasicTicker"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"1051"}]},"id":"1064","type":"LegendItem"},{"attributes":{"coordinates":null,"data_source":{"id":"1045"},"glyph":{"id":"1048"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1050"},"nonselection_glyph":{"id":"1049"},"selection_glyph":{"id":"1065"},"view":{"id":"1052"}},"id":"1051","type":"GlyphRenderer"},{"attributes":{"coordinates":null,"data_source":{"id":"1066"},"glyph":{"id":"1069"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1071"},"nonselection_glyph":{"id":"1070"},"selection_glyph":{"id":"1087"},"view":{"id":"1073"}},"id":"1072","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1049","type":"Scatter"},{"attributes":{},"id":"1040","type":"BasicTickFormatter"},{"attributes":{"below":[{"id":"1018"}],"center":[{"id":"1021"},{"id":"1025"}],"height":300,"left":[{"id":"1022"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1051"},{"id":"1072"},{"id":"1094"}],"right":[{"id":"1063"}],"sizing_mode":"fixed","title":{"id":"1010"},"toolbar":{"id":"1032"},"width":700,"x_range":{"id":"1004"},"x_scale":{"id":"1014"},"y_range":{"id":"1005"},"y_scale":{"id":"1016"}},"id":"1009","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1019","type":"BasicTicker"},{"attributes":{},"id":"1089","type":"Selection"},{"attributes":{"source":{"id":"1045"}},"id":"1052","type":"CDSView"},{"attributes":{},"id":"1067","type":"Selection"},{"attributes":{},"id":"1107","type":"UnionRenderers"},{"attributes":{"coordinates":null,"group":null,"text_color":"black","text_font_size":"12pt"},"id":"1010","type":"Title"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"1094"}]},"id":"1110","type":"LegendItem"},{"attributes":{},"id":"1016","type":"LinearScale"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1048","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1065","type":"Scatter"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1091","type":"Scatter"},{"attributes":{"end":1.6932586586914984,"reset_end":1.6932586586914984,"reset_start":-1.4178143335699678,"start":-1.4178143335699678,"tags":[[["PC1","PC1",null]]]},"id":"1004","type":"Range1d"},{"attributes":{"source":{"id":"1088"}},"id":"1095","type":"CDSView"},{"attributes":{},"id":"1083","type":"UnionRenderers"},{"attributes":{"axis_label":"PC2","coordinates":null,"formatter":{"id":"1043"},"group":null,"major_label_policy":{"id":"1044"},"ticker":{"id":"1023"}},"id":"1022","type":"LinearAxis"},{"attributes":{"source":{"id":"1066"}},"id":"1073","type":"CDSView"},{"attributes":{"axis_label":"PC1","coordinates":null,"formatter":{"id":"1040"},"group":null,"major_label_policy":{"id":"1041"},"ticker":{"id":"1019"}},"id":"1018","type":"LinearAxis"},{"attributes":{"data":{"PC1":{"__ndarray__":"lBtl0cNF8D85yVCOQ+zwP0e0faitTvE/5ZTGnaO08T9nvzqLMeTxPwnUA6wEJfI/4TE4KbZn8j/EIpyDP67yP8vQifW6A/M/I0CeAuUh8z98prc40kPzP03aQaP2Z/M/OsQs/THE8z9VHjw/5+3zP3LWqsI8zfM/E/dEShQT9D+l8dN3ZXz1P0338v0PwPU/QLUf4CIG9j+TxAf/sFb2P6uLIJdRavY/g90/aZXB9j+L0LnpMQn3P5FvEYPXU/c/K66KxVRs9z+O9Lt9PpH3P1LDFDIKn/c/lN3GdL3C9z/514I52tv3P23sks/rJPg/ovXj0Ppo+D/NUnpMnHL4P8LwGD62MvM/qDJrHSHd8z8WZ8xf0D30P8MVTwpPNvQ/yQcmiVx59D/f78sT/F70P9EW3tUlrvQ/hjDHlIEm9T+mNsWCjYL1P/5nEn2qbPU/nP4Fkh2P9T+GbydScuj1P+MGvWej7vU/l7E+fT379T+7N+ZsAQv2P1Uij3DROfY/mIeE7aOD8z/2TNFRzhH0P07CQMazVvQ/7Rr+Ieen9D/SJYdAk830P2//NF2K6fQ/3mPb/NZJ9T9QDrJGlTT1P3r6ymkCPfU/YSAFORpe9T8xNYk0n1j1P5aMWBpIhvU/x3WiBF2d9T+c6EoCJdb1P3h3vNps3/U/2kDin7oI9j+MqprCxsf1P2iw1nCI0vU/LARrFKI29j9UKWb1g6/2P+KT16HBXfY/Jf7VQHLM9j/RzAIE7wz3P677tV1kSfc/ByAiwxJ19z8FiqE851X3P33TXSJjZfc/WHZ2Pza19z8CbRGiBu33P7zuFLfmkPg/jIbkv4b5+D9Ud1R6kyD5PzTesk6WJfc/BLpgdi8j9z94moDDe0T3Py1pdUzyw/c/HZm/LiLp9z+B0VMsxUf4P/6NZThWWfg/9wUm0aiT+D8BBnabdXX4P55RJw/bU/I/vgd1eFWp8j9dugQoRznzPzJ2eeNRsvM/fx/FT9P28z+5q44xrHH0PyltpJNZ5/Q/xj8mU4dA9T/KnzvsmHn1P83ihJ33W/U/KknRPW6f9T8GWG+Ho/T1P03HtiX0K/Y/0x2M1mFd9j8wg2+MJcr2P34aNVG9y/Y/s6Dh2o078D8pc0kqm87wP8toJkXdcvE//Nn3AAcD8j//lzfPDcjyP/Hh6jg2W/M/kfVMBOmb8z+xK6dlBS30PwEq2A2EYPQ/HLpqVv8k9D/cNBVi9kb0PzIzGPjkP/Q/zSl6Mo4m9D+xR3Y4ntn0P3r+P/sC6PQ/qstsMfFo9T8htXvLUX3yP6n9tXlGmvI/Mj4m3sT68j8xCr0/QErzP8rb3QBCVPM/lJvRRXWQ8z9QoM+WNPDzP9ZALPCxOPQ/KAtZUwKL9D9CNDYaWrb0P8PwtL9Q7vQ/asXnHeIt9T/SLghUMSn1PweVQ9FjtfU/dQmE3jbC9T9eUOT4hOb1P+kmuWhMg/M/k0gnyxfb8z8Bh6qz5wv0P/1YmAmVYfQ/ov0zZMGW9D/k4E3GRIj0P7LJFlaRA/U/WxykaU7+9D/bH+Jk8D31Pz9LEEGIRPU/csFEWVqE9T8D+AcXabT1P8gMIXEYsPU/iUtOtoHQ9T8svjetjCf2P28Hkm0pX/U/lDhqWqYc8D/9WDhQUf3wP2sorkJygfE/kambudhi8T9lntt3/czxP0rF+OqKx/E/OGNr+FYe8j//5Xog9O7yP+XXKMfimvM/ZJnxl/aJ8z/dkuPdN2XzP5idMg/pVPM/H3HppSBe8z+IVAak4YjzP375h7kJh/M/FlEbPHyQ8z+5ekvYaEzzP3FOyo5HW/M/dyXV3BSo8z9SLNDgTtbzPwuEd3fRAvQ/+h+TZw2p9D8vv6PMqtX0PxbNlX4wBvU/sW60kVE29T8qjaivD2H1P5hnsrFnCvY/A4peElMi9j/AJYCFVln2P84TtND3g/Y/KjTTJHCs9j8Tc5LgFdb2P4OHdVfvyPI/+GoFClTq8j8mXJhv1kbzPwHbzDI1PPM/rw4oBdim8z8pcXHaYxL0P7k689abZ/Q/Q+m5WdzX9D+l85fLyFf1P34XsdisXvU/Hxs9Mu5o9T9Hucx577T1P8e2TDkl4PU/PQdtpI/k9T8tIRpNYA72P0+52Ta5GPY/O5SXwJQe9T+Cbdt2HD71P+NqBYLQLPU/+jEUdcr29T8KTV4uvfT1P1W7+1yhYfY/6pGy0l759j9McHX8oQH3P2zKEwv5Qvc/fxuCwaom9z/EtyT9KUv3PyuM2nByhPc/todmCY5s9z/8L1HTDWX3P6r0PTcBx/c/Gixv6Wfa9z/iW79Vep31P/zjDsM88/U/uH2FXAI79j8ZNsO2QID2PwVzR+QgWfY/bFdZhul49j8JjlvwM772P1vGktfj4vY/l06FVfXg9j+VWaMfJ/32P//y9y1nLvc/5L9BqUxZ9z9T9zjFU3z3P5s7cBAXCfg/aScZ4g4W+D8ejZwFQSj4Pw==","dtype":"float64","order":"little","shape":[233]},"PC2":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[233]},"class":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],"country":["Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Bulgaria","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Czechia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Croatia","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Hungary","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Ireland","Japan","Japan","Japan","Japan","Japan","Japan","Japan","Japan","Japan","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Lithuania","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Latvia","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Poland","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Portugal","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Romania","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Singapore","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovakia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Slovenia","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden","Sweden"],"year":[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]},"selected":{"id":"1089"},"selection_policy":{"id":"1107"}},"id":"1088","type":"ColumnDataSource"},{"attributes":{},"id":"1043","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1093","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"1070","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01711","sizing_mode":"stretch_width"},"id":"1238","type":"Spacer"}],"root_ids":["1002"]},"title":"Bokeh Application","version":"2.4.2"}};
    var render_items = [{"docid":"5bff0b63-e17f-4000-96f5-441681c55833","root_ids":["1002"],"roots":{"1002":"033f02ec-2c86-43a0-a5a1-79509a9f808d"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
class_df=combined_df.groupby(["region","class"])[["region"]].count()
print(class_df.to_string())
```

                                      region
    region                     class        
    East Asia & Pacific        0          99
                               1         235
                               2          25
    Europe & Central Asia      0           5
                               1         261
                               2         208
    Latin America & Caribbean  0          16
                               1         368
    Middle East & North Africa 1         184
    South Asia                 0          78
                               1          50
    Sub-Saharan Africa         0         599
                               1          89
    


```python
combined_df[["country","class"]].value_counts()
```




    country                           class
    Afghanistan                       0        16
    Mauritania                        0        16
    Panama                            1        16
    Pakistan                          0        16
    Oman                              1        16
                                               ..
    Kenya                             1         2
    Azerbaijan                        0         2
    Cambodia                          1         2
    Kiribati                          0         1
    Micronesia (Federated States of)  0         1
    Length: 159, dtype: int64




```python

```


```python

```


```python

```


```python

```
