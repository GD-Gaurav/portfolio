{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1><center>PART A</center></h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1) Import numpy, pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2) Read the heart.csv file into a dataframe called df, and  view a 2% random sample of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>50</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>110</td>\n",
       "      <td>254</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>159</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>51</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>110</td>\n",
       "      <td>175</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>123</td>\n",
       "      <td>0</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>179</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>276</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>128</td>\n",
       "      <td>303</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>159</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>238</th>\n",
       "      <td>77</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>125</td>\n",
       "      <td>304</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>162</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>255</th>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>142</td>\n",
       "      <td>309</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>147</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
       "109   50    0   0       110   254    0        0      159      0      0.0   \n",
       "27    51    1   2       110   175    0        1      123      0      0.6   \n",
       "179   57    1   0       150   276    0        0      112      1      0.6   \n",
       "59    57    0   0       128   303    0        0      159      0      0.0   \n",
       "238   77    1   0       125   304    0        0      162      1      0.0   \n",
       "255   45    1   0       142   309    0        0      147      1      0.0   \n",
       "\n",
       "     slope  ca  thal  target  \n",
       "109      2   0     2       1  \n",
       "27       2   0     2       1  \n",
       "179      1   1     1       0  \n",
       "59       2   1     2       1  \n",
       "238      2   3     2       0  \n",
       "255      1   3     3       0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "df = pd.read_csv('/Users/golu/Downloads/kaggleUCI/heart.csv')\n",
    "df.sample(frac=0.02)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3) Get info on df columns (what are the dtypes on each column)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 303 entries, 0 to 302\n",
      "Data columns (total 14 columns):\n",
      "age         303 non-null int64\n",
      "sex         303 non-null int64\n",
      "cp          303 non-null int64\n",
      "trestbps    303 non-null int64\n",
      "chol        303 non-null int64\n",
      "fbs         303 non-null int64\n",
      "restecg     303 non-null int64\n",
      "thalach     303 non-null int64\n",
      "exang       303 non-null int64\n",
      "oldpeak     303 non-null float64\n",
      "slope       303 non-null int64\n",
      "ca          303 non-null int64\n",
      "thal        303 non-null int64\n",
      "target      303 non-null int64\n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 33.2 KB\n"
     ]
    }
   ],
   "source": [
    "# your code here\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4) Split the data into Xtrain, Xtest, ytrain, ytest - with 20% in test, and random_state=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "X = df.drop(['target'], axis=1)\n",
    "y = df['target']\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)\n",
    "\n",
    "### Following code to deal with SetttingWithCopyWarning, and ensure we are working with a copy of the data and not a view\n",
    "Xtrain = Xtrain.copy()\n",
    "Xtest = Xtest.copy()\n",
    "ytrain = ytrain.copy()\n",
    "ytest = ytest.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5) Get basic stats on Xtrain (such as count, mean, std, etc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.00000</td>\n",
       "      <td>242.00000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "      <td>242.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>54.475207</td>\n",
       "      <td>0.694215</td>\n",
       "      <td>0.979339</td>\n",
       "      <td>131.380165</td>\n",
       "      <td>246.371901</td>\n",
       "      <td>0.14876</td>\n",
       "      <td>0.53719</td>\n",
       "      <td>149.524793</td>\n",
       "      <td>0.318182</td>\n",
       "      <td>1.023140</td>\n",
       "      <td>1.409091</td>\n",
       "      <td>0.727273</td>\n",
       "      <td>2.289256</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.155719</td>\n",
       "      <td>0.461694</td>\n",
       "      <td>1.024385</td>\n",
       "      <td>17.409520</td>\n",
       "      <td>51.509276</td>\n",
       "      <td>0.35659</td>\n",
       "      <td>0.52397</td>\n",
       "      <td>23.560318</td>\n",
       "      <td>0.466736</td>\n",
       "      <td>1.098264</td>\n",
       "      <td>0.612796</td>\n",
       "      <td>1.006205</td>\n",
       "      <td>0.610258</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>71.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>47.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>212.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>134.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>55.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>240.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>154.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>61.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>274.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>166.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>77.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>192.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>2.00000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>5.600000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              age         sex          cp    trestbps        chol        fbs  \\\n",
       "count  242.000000  242.000000  242.000000  242.000000  242.000000  242.00000   \n",
       "mean    54.475207    0.694215    0.979339  131.380165  246.371901    0.14876   \n",
       "std      9.155719    0.461694    1.024385   17.409520   51.509276    0.35659   \n",
       "min     29.000000    0.000000    0.000000   94.000000  126.000000    0.00000   \n",
       "25%     47.000000    0.000000    0.000000  120.000000  212.000000    0.00000   \n",
       "50%     55.000000    1.000000    1.000000  130.000000  240.000000    0.00000   \n",
       "75%     61.000000    1.000000    2.000000  140.000000  274.000000    0.00000   \n",
       "max     77.000000    1.000000    3.000000  192.000000  564.000000    1.00000   \n",
       "\n",
       "         restecg     thalach       exang     oldpeak       slope          ca  \\\n",
       "count  242.00000  242.000000  242.000000  242.000000  242.000000  242.000000   \n",
       "mean     0.53719  149.524793    0.318182    1.023140    1.409091    0.727273   \n",
       "std      0.52397   23.560318    0.466736    1.098264    0.612796    1.006205   \n",
       "min      0.00000   71.000000    0.000000    0.000000    0.000000    0.000000   \n",
       "25%      0.00000  134.500000    0.000000    0.000000    1.000000    0.000000   \n",
       "50%      1.00000  154.000000    0.000000    0.800000    1.000000    0.000000   \n",
       "75%      1.00000  166.000000    1.000000    1.600000    2.000000    1.000000   \n",
       "max      2.00000  202.000000    1.000000    5.600000    2.000000    4.000000   \n",
       "\n",
       "             thal  \n",
       "count  242.000000  \n",
       "mean     2.289256  \n",
       "std      0.610258  \n",
       "min      0.000000  \n",
       "25%      2.000000  \n",
       "50%      2.000000  \n",
       "75%      3.000000  \n",
       "max      3.000000  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "Xtrain.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6) Check if there are any NaNs in any of the columns in Xtrain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "cp          0\n",
       "trestbps    0\n",
       "chol        0\n",
       "fbs         0\n",
       "restecg     0\n",
       "thalach     0\n",
       "exang       0\n",
       "oldpeak     0\n",
       "slope       0\n",
       "ca          0\n",
       "thal        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "Xtrain.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7) Create two lists called numeric_features (with age, trestbps, chol, thalach, oldspeak), and categorical_features (with sex, cp, fbs, restecg, exang, slope, ca, thal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "numeric_features = ['age','trestbps', 'chol','thalach','oldpeak']\n",
    "categorical_features = ['sex','cp','fbs', 'restecg','exang','slope','ca','thal']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8) Standard Scale all of the numeric features in Xtrain, include transformed numeric features in Xtrain, and drop original numeric features in Xtrain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, **fit_params).transform(X)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>exang</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>ss_age</th>\n",
       "      <th>ss_trestbps</th>\n",
       "      <th>ss_chol</th>\n",
       "      <th>ss_thalach</th>\n",
       "      <th>ss_oldpeak</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>62</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>-0.270906</td>\n",
       "      <td>-0.770147</td>\n",
       "      <td>-1.174488</td>\n",
       "      <td>1.721500</td>\n",
       "      <td>-0.933529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>127</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1.370810</td>\n",
       "      <td>1.186855</td>\n",
       "      <td>0.595846</td>\n",
       "      <td>0.955920</td>\n",
       "      <td>-0.933529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0.276333</td>\n",
       "      <td>1.071737</td>\n",
       "      <td>-2.341741</td>\n",
       "      <td>0.998453</td>\n",
       "      <td>-0.751046</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>287</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.276333</td>\n",
       "      <td>1.301972</td>\n",
       "      <td>-0.279594</td>\n",
       "      <td>0.615663</td>\n",
       "      <td>-0.933529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>-0.489801</td>\n",
       "      <td>-0.655030</td>\n",
       "      <td>-0.046143</td>\n",
       "      <td>0.530598</td>\n",
       "      <td>0.070128</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     sex  cp  fbs  restecg  exang  slope  ca  thal    ss_age  ss_trestbps  \\\n",
       "62     1   3    0        0      0      1   0     1 -0.270906    -0.770147   \n",
       "127    0   2    0        1      0      2   1     2  1.370810     1.186855   \n",
       "111    1   2    1        1      0      2   1     3  0.276333     1.071737   \n",
       "287    1   1    0        0      0      2   1     2  0.276333     1.301972   \n",
       "108    0   1    0        1      0      2   0     2 -0.489801    -0.655030   \n",
       "\n",
       "      ss_chol  ss_thalach  ss_oldpeak  \n",
       "62  -1.174488    1.721500   -0.933529  \n",
       "127  0.595846    0.955920   -0.933529  \n",
       "111 -2.341741    0.998453   -0.751046  \n",
       "287 -0.279594    0.615663   -0.933529  \n",
       "108 -0.046143    0.530598    0.070128  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "ss = StandardScaler(with_mean=True, with_std=True)\n",
    "dfnumss = pd.DataFrame(ss.fit_transform(Xtrain[numeric_features]), columns=['ss_'+x for x in numeric_features],index = Xtrain.index)\n",
    "Xtrain = pd.concat([Xtrain, dfnumss], axis=1)\n",
    "Xtrain = Xtrain.drop(numeric_features, axis=1)\n",
    "Xtrain.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9) OneHotEncode all of the categorical features in Xtrain, include transformed categorical features in Xtrain, and drop original categorical features in Xtrain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ss_age</th>\n",
       "      <th>ss_trestbps</th>\n",
       "      <th>ss_chol</th>\n",
       "      <th>ss_thalach</th>\n",
       "      <th>ss_oldpeak</th>\n",
       "      <th>x0_0</th>\n",
       "      <th>x0_1</th>\n",
       "      <th>x1_0</th>\n",
       "      <th>x1_1</th>\n",
       "      <th>x1_2</th>\n",
       "      <th>...</th>\n",
       "      <th>x5_2</th>\n",
       "      <th>x6_0</th>\n",
       "      <th>x6_1</th>\n",
       "      <th>x6_2</th>\n",
       "      <th>x6_3</th>\n",
       "      <th>x6_4</th>\n",
       "      <th>x7_0</th>\n",
       "      <th>x7_1</th>\n",
       "      <th>x7_2</th>\n",
       "      <th>x7_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>222</th>\n",
       "      <td>1.151915</td>\n",
       "      <td>0.381030</td>\n",
       "      <td>0.693117</td>\n",
       "      <td>1.040985</td>\n",
       "      <td>0.343852</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>1.808601</td>\n",
       "      <td>1.647326</td>\n",
       "      <td>1.082201</td>\n",
       "      <td>0.530598</td>\n",
       "      <td>-0.568563</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>170</th>\n",
       "      <td>0.166885</td>\n",
       "      <td>-0.079441</td>\n",
       "      <td>0.187307</td>\n",
       "      <td>-0.320046</td>\n",
       "      <td>-0.386080</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>-0.927592</td>\n",
       "      <td>0.611266</td>\n",
       "      <td>-1.349576</td>\n",
       "      <td>0.445534</td>\n",
       "      <td>0.343852</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>158</th>\n",
       "      <td>0.385781</td>\n",
       "      <td>-0.367235</td>\n",
       "      <td>-0.513045</td>\n",
       "      <td>-0.234982</td>\n",
       "      <td>-0.568563</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       ss_age  ss_trestbps   ss_chol  ss_thalach  ss_oldpeak  x0_0  x0_1  \\\n",
       "222  1.151915     0.381030  0.693117    1.040985    0.343852     0     1   \n",
       "25   1.808601     1.647326  1.082201    0.530598   -0.568563     1     0   \n",
       "170  0.166885    -0.079441  0.187307   -0.320046   -0.386080     0     1   \n",
       "35  -0.927592     0.611266 -1.349576    0.445534    0.343852     1     0   \n",
       "158  0.385781    -0.367235 -0.513045   -0.234982   -0.568563     0     1   \n",
       "\n",
       "     x1_0  x1_1  x1_2  ...  x5_2  x6_0  x6_1  x6_2  x6_3  x6_4  x7_0  x7_1  \\\n",
       "222     0     0     0  ...     0     0     1     0     0     0     0     0   \n",
       "25      0     1     0  ...     1     0     0     1     0     0     0     0   \n",
       "170     0     0     1  ...     0     0     1     0     0     0     0     1   \n",
       "35      0     0     1  ...     0     1     0     0     0     0     0     0   \n",
       "158     0     1     0  ...     0     0     0     0     0     1     0     0   \n",
       "\n",
       "     x7_2  x7_3  \n",
       "222     1     0  \n",
       "25      1     0  \n",
       "170     0     0  \n",
       "35      1     0  \n",
       "158     0     1  \n",
       "\n",
       "[5 rows x 30 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "ohe = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')\n",
    "Xcat = pd.DataFrame(ohe.fit_transform(Xtrain[categorical_features]), \n",
    "                    columns=ohe.get_feature_names(), index=Xtrain.index)\n",
    "Xtrain = pd.concat([Xtrain, Xcat], axis=1)\n",
    "Xtrain = Xtrain.drop(categorical_features, axis=1)\n",
    "\n",
    "Xtrain.sample(frac=0.02)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10) Fit a Logistic Regression Model to training data with random_state=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='warn',\n",
       "          n_jobs=None, penalty='l2', random_state=1, solver='warn',\n",
       "          tol=0.0001, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "lr = LogisticRegression(random_state = 1)  \n",
    "lr.fit(Xtrain, ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11) Standard Scale all of the numeric features in Xtest, include transformed numeric features in Xtest, and drop original numeric features in Xtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/golu/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  \n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>exang</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>ss_age</th>\n",
       "      <th>ss_trestbps</th>\n",
       "      <th>ss_chol</th>\n",
       "      <th>ss_thalach</th>\n",
       "      <th>ss_oldpeak</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>204</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>0.823571</td>\n",
       "      <td>1.647326</td>\n",
       "      <td>-1.602481</td>\n",
       "      <td>-0.192449</td>\n",
       "      <td>4.723443</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0.166885</td>\n",
       "      <td>-0.079441</td>\n",
       "      <td>-0.493590</td>\n",
       "      <td>0.573130</td>\n",
       "      <td>-0.933529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>219</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>-0.708697</td>\n",
       "      <td>-0.079441</td>\n",
       "      <td>0.187307</td>\n",
       "      <td>0.020212</td>\n",
       "      <td>-0.933529</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>174</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>0.604676</td>\n",
       "      <td>-0.079441</td>\n",
       "      <td>-0.785404</td>\n",
       "      <td>-0.745368</td>\n",
       "      <td>1.256267</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>184</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>-0.489801</td>\n",
       "      <td>1.071737</td>\n",
       "      <td>-0.065598</td>\n",
       "      <td>-0.915497</td>\n",
       "      <td>1.438750</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     sex  cp  fbs  restecg  exang  slope  ca  thal    ss_age  ss_trestbps  \\\n",
       "204    0   0    0        0      0      0   3     3  0.823571     1.647326   \n",
       "159    1   1    0        0      0      2   0     3  0.166885    -0.079441   \n",
       "219    1   0    1        0      1      2   2     3 -0.708697    -0.079441   \n",
       "174    1   0    0        0      1      1   2     3  0.604676    -0.079441   \n",
       "184    1   0    0        0      0      1   0     3 -0.489801     1.071737   \n",
       "\n",
       "      ss_chol  ss_thalach  ss_oldpeak  \n",
       "204 -1.602481   -0.192449    4.723443  \n",
       "159 -0.493590    0.573130   -0.933529  \n",
       "219  0.187307    0.020212   -0.933529  \n",
       "174 -0.785404   -0.745368    1.256267  \n",
       "184 -0.065598   -0.915497    1.438750  "
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "dfnumss = pd.DataFrame(ss.transform(Xtest[numeric_features]), columns=['ss_'+x for x in numeric_features],index = Xtest.index)\n",
    "Xtest = pd.concat([Xtest, dfnumss], axis=1)\n",
    "Xtest = Xtest.drop(numeric_features, axis=1)\n",
    "Xtest.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 12) OneHotEncode all of the categorical features in Xtest, include transformed categorical features in Xtest, and drop original categorical features in Xtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ss_age</th>\n",
       "      <th>ss_trestbps</th>\n",
       "      <th>ss_chol</th>\n",
       "      <th>ss_thalach</th>\n",
       "      <th>ss_oldpeak</th>\n",
       "      <th>x0_0</th>\n",
       "      <th>x0_1</th>\n",
       "      <th>x1_0</th>\n",
       "      <th>x1_1</th>\n",
       "      <th>x1_2</th>\n",
       "      <th>...</th>\n",
       "      <th>x5_2</th>\n",
       "      <th>x6_0</th>\n",
       "      <th>x6_1</th>\n",
       "      <th>x6_2</th>\n",
       "      <th>x6_3</th>\n",
       "      <th>x6_4</th>\n",
       "      <th>x7_0</th>\n",
       "      <th>x7_1</th>\n",
       "      <th>x7_2</th>\n",
       "      <th>x7_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>223</th>\n",
       "      <td>0.166885</td>\n",
       "      <td>3.949681</td>\n",
       "      <td>0.809842</td>\n",
       "      <td>-0.702836</td>\n",
       "      <td>2.716131</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 30 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       ss_age  ss_trestbps   ss_chol  ss_thalach  ss_oldpeak  x0_0  x0_1  \\\n",
       "223  0.166885     3.949681  0.809842   -0.702836    2.716131     1     0   \n",
       "\n",
       "     x1_0  x1_1  x1_2  ...  x5_2  x6_0  x6_1  x6_2  x6_3  x6_4  x7_0  x7_1  \\\n",
       "223     1     0     0  ...     0     0     0     1     0     0     0     0   \n",
       "\n",
       "     x7_2  x7_3  \n",
       "223     0     1  \n",
       "\n",
       "[1 rows x 30 columns]"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "Xcat = pd.DataFrame(ohe.transform(Xtest[categorical_features]), \n",
    "                    columns=ohe.get_feature_names(), index=Xtest.index)\n",
    "Xtest = pd.concat([Xtest, Xcat], axis=1)\n",
    "Xtest = Xtest.drop(categorical_features, axis=1)\n",
    "\n",
    "Xtest.sample(frac=0.02)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 13) Predict and Evaluate Logisitic Regression Model on Xtest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7868852459016393\n"
     ]
    }
   ],
   "source": [
    "# your code here\n",
    "ypred = lr.predict(Xtest)\n",
    "\n",
    "from sklearn import metrics\n",
    "print (metrics.accuracy_score(ytest, ypred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1><center>Part B</center></h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 14) (a) Read in heart.csv in to a dataframe called df; (b) Split the data into Xtrain, Xtest, ytrain, ytest - with 20% in test, and random_state=1; (c) Create two lists called numeric_features (with age, trestbps, chol, thalach, oldspeak), and categorical_features (with sex, cp, fbs, restecg, exang, slope, ca, thal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "df = pd.read_csv('/Users/golu/Downloads/kaggleUCI/heart.csv')\n",
    "df.sample(frac=0.02)\n",
    "X = df.drop(['target'], axis=1)\n",
    "y = df['target']\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)\n",
    "\n",
    "### Following code to deal with SetttingWithCopyWarning, and ensure we are working with a copy of the data and not a view\n",
    "Xtrain = Xtrain.copy()\n",
    "Xtest = Xtest.copy()\n",
    "ytrain = ytrain.copy()\n",
    "ytest = ytest.copy()\n",
    "\n",
    "numeric_features = ['age','trestbps', 'chol','thalach','oldpeak']\n",
    "categorical_features = ['sex','cp','fbs', 'restecg','exang','slope','ca','thal']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 15) Create a pipeline called \"numeric_transformer\" with a StandardScaler step called \"ss\" (use the same parameters that you used in Part A above)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "from sklearn.pipeline import Pipeline\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('ss', StandardScaler(with_mean=True, with_std=True))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 16) Create a pipeline called \"categorical_transformer\" with a OneHotEncoder step called \"ohe\" (use the same parameters that you used in Part A above)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('ohe', OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore'))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 17) Create a column transformer called \"preprocessor\" with two transformers: (a) the first transformer called \"num\" which uses the numeric_transformer you defined above on the numeric_features; and (b) the second transformer called \"cat\" which uses the categorical_transformer you defined above on the categorical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "from sklearn.compose import ColumnTransformer\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numeric_features),\n",
    "        ('cat', categorical_transformer, categorical_features)],\n",
    "    remainder='drop') "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 18) Create a pipeline called \"clf\" with two steps: (a) the first step called \"pp\" which invokes the preprocessor you defined above; and (b) the second step called \"lr\" which involkes a logisitc regression model  (use the same parameters that you used in Part A above)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# your code here\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LogisticRegression  \n",
    "clf = Pipeline(steps=[('pp', preprocessor),\n",
    "                      ('lr', LogisticRegression(random_state = 1))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 19) Fit the clf pipeline to the trainign data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/base.py:467: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, y, **fit_params).transform(X)\n",
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "     steps=[('pp', ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,\n",
       "         transformer_weights=None,\n",
       "         transformers=[('num', Pipeline(memory=None,\n",
       "     steps=[('ss', StandardScaler(copy=True, with_mean=True, with_std=True))]), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak...e, penalty='l2', random_state=1, solver='warn',\n",
       "          tol=0.0001, verbose=0, warm_start=False))])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# your code here\n",
    "clf.fit(Xtrain, ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 20) Predict and Evaluate clf pipeline on Xtest (you should end up with same results as in Part A above)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7868852459016393\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/golu/anaconda3/lib/python3.7/site-packages/sklearn/pipeline.py:451: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.\n",
      "  Xt = transform.transform(Xt)\n"
     ]
    }
   ],
   "source": [
    "# your code here\n",
    "ypred = clf.predict(Xtest)\n",
    "from sklearn import metrics\n",
    "print (metrics.accuracy_score(ytest, ypred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "# leave this cell blank"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
