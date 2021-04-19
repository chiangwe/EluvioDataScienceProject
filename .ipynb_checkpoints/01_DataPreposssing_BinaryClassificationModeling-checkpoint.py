{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "dbd0e53d",
   "metadata": {},
   "source": [
    "# Importing Packages "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "989fffd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/chiangwe/anaconda3/envs/NetHawkes/bin/python\n"
     ]
    }
   ],
   "source": [
    "#!pip install tqdm\n",
    "#!pip install nltk\n",
    "#!pip install matplotlib\n",
    "#!pip install tensorflow_hub\n",
    "#!pip install tensorflow_datasets\n",
    "#!pip install  obspy\n",
    "#!pip install tensorflow_datasets\n",
    "#!pip install tensorflow_hub\n",
    "import sys\n",
    "print(sys.executable)\n",
    "import tensorflow_datasets as tfds"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "9eeefd3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#============ Importing Packages ============# \n",
    "\n",
    "#--------- Drawing Packages ---------#\n",
    "\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, NullFormatter, LogLocator)\n",
    "from set_size import set_size\n",
    "from collections import Counter\n",
    "import tensorflow_hub as hub\n",
    "\n",
    "#--------- Utilities Packages ---------#\n",
    "import os\n",
    "import re\n",
    "import pdb\n",
    "import shelve\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from scipy.sparse import csr_matrix, hstack\n",
    "from tqdm import tqdm\n",
    "\n",
    "import nltk\n",
    "import obspy\n",
    "#nltk.download('wordnet')\n",
    "#nltk.download('stopwords')\n",
    "#nltk.download('word_tokenize')\n",
    "from nltk.corpus import stopwords \n",
    "from nltk.tokenize import word_tokenize \n",
    "\n",
    "# Scipy Signal\n",
    "from scipy import signal\n",
    "# Detrend the Signal\n",
    "from obspy.signal.detrend import polynomial\n",
    "\n",
    "#stop_words = set(stopwords.words('english')) \n",
    "\n",
    "#--------- Analiing results  ---------#\n",
    "\n",
    "#import xgboost\n",
    "#import shap\n",
    "\n",
    "#--------- Remove Warnings ---------#\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e230a94",
   "metadata": {},
   "source": [
    "# Checking the statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "daea50f4",
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
       "      <th>time_created</th>\n",
       "      <th>date_created</th>\n",
       "      <th>up_votes</th>\n",
       "      <th>down_votes</th>\n",
       "      <th>title</th>\n",
       "      <th>over_18</th>\n",
       "      <th>author</th>\n",
       "      <th>category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1201232046</td>\n",
       "      <td>2008-01-25</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>Scores killed in Pakistan clashes</td>\n",
       "      <td>False</td>\n",
       "      <td>polar</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1201232075</td>\n",
       "      <td>2008-01-25</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>Japan resumes refuelling mission</td>\n",
       "      <td>False</td>\n",
       "      <td>polar</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1201232523</td>\n",
       "      <td>2008-01-25</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>US presses Egypt on Gaza border</td>\n",
       "      <td>False</td>\n",
       "      <td>polar</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1201233290</td>\n",
       "      <td>2008-01-25</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Jump-start economy: Give health care to all</td>\n",
       "      <td>False</td>\n",
       "      <td>fadi420</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1201274720</td>\n",
       "      <td>2008-01-25</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>Council of Europe bashes EU&amp;UN terror blacklist</td>\n",
       "      <td>False</td>\n",
       "      <td>mhermans</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   time_created date_created  up_votes  down_votes  \\\n",
       "0    1201232046   2008-01-25         3           0   \n",
       "1    1201232075   2008-01-25         2           0   \n",
       "2    1201232523   2008-01-25         3           0   \n",
       "3    1201233290   2008-01-25         1           0   \n",
       "4    1201274720   2008-01-25         4           0   \n",
       "\n",
       "                                             title  over_18    author  \\\n",
       "0                Scores killed in Pakistan clashes    False     polar   \n",
       "1                 Japan resumes refuelling mission    False     polar   \n",
       "2                  US presses Egypt on Gaza border    False     polar   \n",
       "3     Jump-start economy: Give health care to all     False   fadi420   \n",
       "4  Council of Europe bashes EU&UN terror blacklist    False  mhermans   \n",
       "\n",
       "    category  \n",
       "0  worldnews  \n",
       "1  worldnews  \n",
       "2  worldnews  \n",
       "3  worldnews  \n",
       "4  worldnews  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "It doesn' have down_votes.\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('Eluvio_DS_Challenge.csv')\n",
    "display(df.head(5))\n",
    "print(df['down_votes'].sum())\n",
    "print(\"It doesn' have down_votes.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4a4c71da",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of posts:  509236\n",
      "Date range:  (Timestamp('2008-01-25 00:00:00'), Timestamp('2016-11-22 00:00:00'))  tol days:  3224 days 00:00:00\n",
      "Stats for number of reddits from posts:\n"
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
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>n_post</th>\n",
       "      <td>3223.0</td>\n",
       "      <td>158.000621</td>\n",
       "      <td>97.360446</td>\n",
       "      <td>1.0</td>\n",
       "      <td>74.0</td>\n",
       "      <td>129.0</td>\n",
       "      <td>241.0</td>\n",
       "      <td>458.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         count        mean        std  min   25%    50%    75%    max\n",
       "n_post  3223.0  158.000621  97.360446  1.0  74.0  129.0  241.0  458.0"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
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
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>up_votes</th>\n",
       "      <td>3223.0</td>\n",
       "      <td>90.152178</td>\n",
       "      <td>65.953134</td>\n",
       "      <td>1.245902</td>\n",
       "      <td>35.178536</td>\n",
       "      <td>79.960396</td>\n",
       "      <td>128.165878</td>\n",
       "      <td>477.19084</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>down_votes</th>\n",
       "      <td>3223.0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             count       mean        std       min        25%        50%  \\\n",
       "up_votes    3223.0  90.152178  65.953134  1.245902  35.178536  79.960396   \n",
       "down_votes  3223.0   0.000000   0.000000  0.000000   0.000000   0.000000   \n",
       "\n",
       "                   75%        max  \n",
       "up_votes    128.165878  477.19084  \n",
       "down_votes    0.000000    0.00000  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABj4AAAFnCAYAAAAWgjCMAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAD4oklEQVR4nOzdeZwT9f0/8Nc7e3Aox3KKLKcgKqgIiCjW+6y2XlVprdWqtbX0+tpfW7TWg9ZKW++q9cADFUUUFRRRELnlXOQ+l2WB5WbZXRb2TPL5/ZGZ7CSZJJNkkkmyr+fjAZtM5vjk2M1n5v35vN+ilAIREREREREREREREVE2cDndACIiIiIiIiIiIiIiIrsw8EFERERERERERERERFmDgQ8iIiIiIiIiIiIiIsoaDHwQEREREREREREREVHWYOCDiIiIiIiIiIiIiIiyBgMfRERERERERERERESUNRj4ICIioqwjIqUislZEVonICm1ZBxGZJSJbtZ8FhvUfEJFiEdksIlc613IiIiIiIiIiShQDH0RERJStLlZKDVZKDdPujwEwWynVH8Bs7T5E5DQAowAMBHAVgJdEJMeJBhMRERERERFR4hj4ICIioubiOgATtNsTAFxvWD5JKVWvlNoOoBjA8NQ3j4iIiIiIiIjskOt0AxLRqVMn1bt3b6ebQURElDaKiooOKaU6O92ONKAAzBQRBeAVpdSrALoqpfYCgFJqr4h00dbtDmCJYdsybVlY7IMQEREFYh8kNdgHISIiChSuD5LRgY/evXtjxYoVTjeDiIgobYjIDqfbkCZGKqX2aMGNWSKyKcK6YrJMhawkci+AewGgZ8+e7IMQEREZsA+SGrwOQkREFChcH4SproiIiCjrKKX2aD8PAPgEvtRV+0WkGwBoPw9oq5cB6GHYvBDAHpN9vqqUGqaUGta5Mwe0EhEREREREaUrBj6IiIgoq4jIcSLSRr8N4AoA6wBMA3CHttodAKZqt6cBGCUiLUSkD4D+AJalttVEREREREREZJeMTnVFREREZKIrgE9EBPD1dd5TSn0pIssBTBaRuwHsBHAzACil1ovIZAAbALgBjFZKeZxpOhERERERERElKusCH42NjSgrK0NdXZ3TTUkLLVu2RGFhIfLy8pxuChERUUoopUoAnGmyvBzApWG2eRzA44kcl32QxLDPQkRERERE2YznjImJ9Zwx6wIfZWVlaNOmDXr37g1tpGezpZRCeXk5ysrK0KdPH6ebQ0RElNXYB4kf+yxERERERJTteM4Yv3jOGbOuxkddXR06duzIDw8AEUHHjh0ZRSQiIkoB9kHixz4LERERERFlO54zxi+ec8asC3wA4IfHgK8FERFR6vB7N3587YiIiIiIKNvxvCd+sb52WRn4ICIiIiIiIiIiIiKi5omBjyxQWlqK9957z+lmEBERUTP06aefYsOGDU43g4iIiIiIiNKQU+eMDHxkAQY+iIiIyCkMfBARERERESVH0Y4KvDinGEU7KpxuStycOmfMTfkRU+ixz9Zjw54jtu7ztBPb4pEfDIy4TmlpKa6++mqcf/75+Pbbb9G9e3dMnToVrVq1Cln3oosuwuDBg7Fs2TIcOXIEb7zxBoYPH47Dhw/jrrvuQklJCVq3bo1XX30VZ5xxBubNm4ff//73AHx5zebPn48xY8Zg48aNGDx4MO644w783//9n63PmYiIiGLjZB/k2muvxbp16wAATz75JI4ePYq5c+ea9jeCeb1e9O3bF6tWrUL79u0BAP369cOiRYtQV1eHu+66CwcPHkTnzp3x5ptvoqysDNOmTcO8efPwj3/8A1OmTAEAjB49GgcPHkTr1q3x2muv4ZRTTsGHH36Ixx57DDk5OWjXrh3mz59v6+tDRERERESUbop2VGBJSTlG9O2I1nFse9v4JWhwe5Gf68LEe0ZgaK+ChNrTnM4ZOeMjSbZu3YrRo0dj/fr1aN++vf9NNXPs2DF8++23eOmll3DXXXcBAB555BGcddZZWLNmDf75z3/iZz/7GQDfh/HFF1/EqlWrsGDBArRq1Qrjxo3D9773PaxatYpBDyJqVtbtrsL4BSVON4MoI5j1N4K5XC5cd911+OSTTwAAS5cuRe/evdG1a1f85je/wc9+9jOsWbMGt912G373u9/hvPPOww9/+EP85z//wapVq3DSSSfh3nvvxX//+18UFRXhySefxK9//WsAwNixY/HVV19h9erVmDZtWsqeNxER+awtq8LrC7c73QwiIqJmQw9cPDVzsz+AEYslJeVocHvhVUCj24slJeVJaqlPtp0zZvWMj2ijIpOpT58+GDx4MABg6NChKC0tDbvuj3/8YwDABRdcgCNHjqCyshILFy70B0suueQSlJeXo6qqCiNHjsT999+P2267DTfeeCMKCwuT/VSIiNLWtf9dCAC453t9HW4JUSAn+yDhmPU39BE6RrfeeivGjh2Ln//855g0aRJuvfVWAMDixYvx8ccfAwBuv/12/PnPfw7Z9ujRo/j2229x8803+5fV19cDAEaOHIk777wTt9xyC2688Ua7nx4REUXxgxd8/aa7z+/jcEuIiIiah+DARb3bE9P2I/p2RH6uC41uL/JyXRjRt2OSWuqTbeeMWR34cFKLFi38t3NyclBbWxt2XREJua+UMl1vzJgxuOaaa/DFF19gxIgR+Prrr+1rNBEREWW03NxceL1No4jq6ur8t836G2bOPfdcFBcX4+DBg/j000/x0EMPma5ntr3X60X79u2xatWqkMdefvllLF26FNOnT8fgwYOxatUqdOyY3I47ERERERGRU4IDFy1yc2LafmivAky8Z4Q/VVaiaa6A5nXOyFRXaeCDDz4AACxcuBDt2rVDu3btcMEFF2DixIkAgLlz56JTp05o27Yttm3bhtNPPx1/+ctfMGzYMGzatAlt2rRBdXW1k0+BiIiI0kDXrl1x4MABlJeXo76+Hp9//rn/MbP+hhkRwQ033ID7778fp556qr+jed5552HSpEkAgIkTJ+L8888HgIB+SNu2bdGnTx98+OGHAAClFFavXg0A2LZtG8455xyMHTsWnTp1wq5du5LwChAREREREaUHPXBx/xUDMPGeEcjPjf1S/NBeBRh9cT9bgh5A8zpn5IyPNFBQUIDzzjvPXzgGAB599FH8/Oc/xxlnnIHWrVtjwoQJAIBnn30Wc+bMQU5ODk477TRcffXVcLlcyM3NxZlnnok777yTdT6IiIiaqby8PDz88MM455xz0KdPH5xyyin+x8z6G+HceuutOPvss/HWW2/5lz3//PO466678J///MdfqA4ARo0ahV/84hd4/vnn8dFHH2HixIm477778I9//AONjY0YNWoUzjzzTPzpT3/C1q1boZTCpZdeijPPPDMprwEREREREVG6GNqrwB+02Lhxn8OtaV7njGKWUilTDBs2TK1YsSJg2caNG3Hqqac61KLYXXTRRXjyyScxbNiwpB0j014TIiKreo+ZDgAoHXeNwy1JHyJSpJRK3pcKAci8Pkgq+ht2SOfXkIgo0yW738Q+SGqY9UGIiCgzpPP5TiafM4brgzDVFRERERERERERERER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\n",
      "text/plain": [
       "<Figure size 1992.53x410.483 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=set_size(width=4000, height_adjust=1, fraction=0.5, subplots=(1,3)))\n",
    "print(\"Number of posts: \", df.shape[0])\n",
    "\n",
    "# Check dates stats\n",
    "min_date = pd.to_datetime(df['date_created'].min());\n",
    "max_date = pd.to_datetime(df['date_created'].max());\n",
    "\n",
    "print(\"Date range: \", (min_date, max_date), \" tol days: \", max_date-min_date )\n",
    "print(\"Stats for number of reddits from posts:\")\n",
    "\n",
    "n_po = pd.DataFrame( df.groupby('date_created')['title'].count().rename('n_post') )\n",
    "avg_votes = pd.DataFrame( df.groupby('date_created')['up_votes', 'down_votes'].mean() )\n",
    "display(n_po.describe().T)\n",
    "display(avg_votes.describe().T)\n",
    "\n",
    "ax1  = n_po.reset_index().reset_index().plot.line(x='index', y='n_post', rot=0, ylabel='number of post', ax=ax1)\n",
    "ax2 = avg_votes.reset_index().reset_index().plot.line(x='index', y='up_votes', rot=0, ylabel='average_up_votes', ax=ax2)\n",
    "ax3 = df.plot.line(x='time_created', y='up_votes', rot=0, ylabel='up_vote', style='.', ax=ax3)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a57f5d31",
   "metadata": {},
   "source": [
    "### We can not just use mean up_vote to lable popular \n",
    "### Reddit has a user growth that reflect to the upvote trend\n",
    "### We will use the previous four weeks of day of the week to decide the label\n",
    "### Popular: more the average "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4c7ca8e6",
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
       "      <th>time_created</th>\n",
       "      <th>date_created</th>\n",
       "      <th>up_votes</th>\n",
       "      <th>down_votes</th>\n",
       "      <th>title</th>\n",
       "      <th>over_18</th>\n",
       "      <th>author</th>\n",
       "      <th>category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>509191</th>\n",
       "      <td>1479810775</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>18</td>\n",
       "      <td>0</td>\n",
       "      <td>56-car pileup in China leaves 17 dead</td>\n",
       "      <td>False</td>\n",
       "      <td>LenonTV</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509188</th>\n",
       "      <td>1479810217</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>World s First Head Transplant Will Use Virtual...</td>\n",
       "      <td>False</td>\n",
       "      <td>Short_Term_Account</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509200</th>\n",
       "      <td>1479812515</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>Extraordinarily hot  Arctic temperatures alar...</td>\n",
       "      <td>False</td>\n",
       "      <td>rawrstevo</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509155</th>\n",
       "      <td>1479801913</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>British Navy fire warning shots at Spanish Nav...</td>\n",
       "      <td>False</td>\n",
       "      <td>cholopapi</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509167</th>\n",
       "      <td>1479805472</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>Turkey to drop child-sex assault bill after pr...</td>\n",
       "      <td>False</td>\n",
       "      <td>ICASL</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509208</th>\n",
       "      <td>1479813152</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>Eagles destroy nine WA mining drones and cost ...</td>\n",
       "      <td>False</td>\n",
       "      <td>Short_Term_Account</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509102</th>\n",
       "      <td>1479783131</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>43</td>\n",
       "      <td>0</td>\n",
       "      <td>Death toll in India train derailment hits 146</td>\n",
       "      <td>False</td>\n",
       "      <td>doctor316</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509178</th>\n",
       "      <td>1479808720</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>Turkish President Erdogan:  I can t say if Isr...</td>\n",
       "      <td>False</td>\n",
       "      <td>ICASL</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509093</th>\n",
       "      <td>1479777027</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>47</td>\n",
       "      <td>0</td>\n",
       "      <td>Vancouver mayor on housing crisis:  I never dr...</td>\n",
       "      <td>False</td>\n",
       "      <td>ManiaforBeatles</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509131</th>\n",
       "      <td>1479793320</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>70</td>\n",
       "      <td>0</td>\n",
       "      <td>British PM signals £2bn a year science funding...</td>\n",
       "      <td>False</td>\n",
       "      <td>cyanocittaetprocyon</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509106</th>\n",
       "      <td>1479784270</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>77</td>\n",
       "      <td>0</td>\n",
       "      <td>Japan earthquake: 7.4 magnitude quake prompts ...</td>\n",
       "      <td>False</td>\n",
       "      <td>hamzatahir671</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509147</th>\n",
       "      <td>1479799623</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>88</td>\n",
       "      <td>0</td>\n",
       "      <td>Britain tells Trump -  There is no vacancy  fo...</td>\n",
       "      <td>False</td>\n",
       "      <td>ManiaforBeatles</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509118</th>\n",
       "      <td>1479789187</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>105</td>\n",
       "      <td>0</td>\n",
       "      <td>Lockheed Martin Lands Massive $1.2 Billion Con...</td>\n",
       "      <td>False</td>\n",
       "      <td>bob21doh</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509144</th>\n",
       "      <td>1479798089</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>169</td>\n",
       "      <td>0</td>\n",
       "      <td>Rice farming in India much older than thought,...</td>\n",
       "      <td>False</td>\n",
       "      <td>avatharam</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509087</th>\n",
       "      <td>1479774030</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>188</td>\n",
       "      <td>0</td>\n",
       "      <td>Ukraine detains two Russian soldiers near Crim...</td>\n",
       "      <td>False</td>\n",
       "      <td>krolique</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509107</th>\n",
       "      <td>1479784292</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>259</td>\n",
       "      <td>0</td>\n",
       "      <td>India All Set to Launch a $2 Billion Renewable...</td>\n",
       "      <td>False</td>\n",
       "      <td>tewrld</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509108</th>\n",
       "      <td>1479784317</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>407</td>\n",
       "      <td>0</td>\n",
       "      <td>Jean-Claude Juncker mounts fresh call for Euro...</td>\n",
       "      <td>False</td>\n",
       "      <td>Alexandra-perez</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509142</th>\n",
       "      <td>1479797823</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>536</td>\n",
       "      <td>0</td>\n",
       "      <td>Turkey s PM withdraws bill that would pardon m...</td>\n",
       "      <td>False</td>\n",
       "      <td>Jurryaany</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509088</th>\n",
       "      <td>1479774884</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>1816</td>\n",
       "      <td>0</td>\n",
       "      <td>New Zealand hit by 6.3 magnitude earthquake in...</td>\n",
       "      <td>False</td>\n",
       "      <td>Gecko5567</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>509094</th>\n",
       "      <td>1479777217</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>3360</td>\n",
       "      <td>0</td>\n",
       "      <td>It s time to decriminalize drugs, commission r...</td>\n",
       "      <td>False</td>\n",
       "      <td>maxwellhill</td>\n",
       "      <td>worldnews</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        time_created date_created  up_votes  down_votes  \\\n",
       "509191    1479810775   2016-11-22        18           0   \n",
       "509188    1479810217   2016-11-22        19           0   \n",
       "509200    1479812515   2016-11-22        19           0   \n",
       "509155    1479801913   2016-11-22        23           0   \n",
       "509167    1479805472   2016-11-22        26           0   \n",
       "509208    1479813152   2016-11-22        41           0   \n",
       "509102    1479783131   2016-11-22        43           0   \n",
       "509178    1479808720   2016-11-22        45           0   \n",
       "509093    1479777027   2016-11-22        47           0   \n",
       "509131    1479793320   2016-11-22        70           0   \n",
       "509106    1479784270   2016-11-22        77           0   \n",
       "509147    1479799623   2016-11-22        88           0   \n",
       "509118    1479789187   2016-11-22       105           0   \n",
       "509144    1479798089   2016-11-22       169           0   \n",
       "509087    1479774030   2016-11-22       188           0   \n",
       "509107    1479784292   2016-11-22       259           0   \n",
       "509108    1479784317   2016-11-22       407           0   \n",
       "509142    1479797823   2016-11-22       536           0   \n",
       "509088    1479774884   2016-11-22      1816           0   \n",
       "509094    1479777217   2016-11-22      3360           0   \n",
       "\n",
       "                                                    title  over_18  \\\n",
       "509191              56-car pileup in China leaves 17 dead    False   \n",
       "509188  World s First Head Transplant Will Use Virtual...    False   \n",
       "509200   Extraordinarily hot  Arctic temperatures alar...    False   \n",
       "509155  British Navy fire warning shots at Spanish Nav...    False   \n",
       "509167  Turkey to drop child-sex assault bill after pr...    False   \n",
       "509208  Eagles destroy nine WA mining drones and cost ...    False   \n",
       "509102      Death toll in India train derailment hits 146    False   \n",
       "509178  Turkish President Erdogan:  I can t say if Isr...    False   \n",
       "509093  Vancouver mayor on housing crisis:  I never dr...    False   \n",
       "509131  British PM signals £2bn a year science funding...    False   \n",
       "509106  Japan earthquake: 7.4 magnitude quake prompts ...    False   \n",
       "509147  Britain tells Trump -  There is no vacancy  fo...    False   \n",
       "509118  Lockheed Martin Lands Massive $1.2 Billion Con...    False   \n",
       "509144  Rice farming in India much older than thought,...    False   \n",
       "509087  Ukraine detains two Russian soldiers near Crim...    False   \n",
       "509107  India All Set to Launch a $2 Billion Renewable...    False   \n",
       "509108  Jean-Claude Juncker mounts fresh call for Euro...    False   \n",
       "509142  Turkey s PM withdraws bill that would pardon m...    False   \n",
       "509088  New Zealand hit by 6.3 magnitude earthquake in...    False   \n",
       "509094  It s time to decriminalize drugs, commission r...    False   \n",
       "\n",
       "                     author   category  \n",
       "509191              LenonTV  worldnews  \n",
       "509188   Short_Term_Account  worldnews  \n",
       "509200            rawrstevo  worldnews  \n",
       "509155            cholopapi  worldnews  \n",
       "509167                ICASL  worldnews  \n",
       "509208   Short_Term_Account  worldnews  \n",
       "509102            doctor316  worldnews  \n",
       "509178                ICASL  worldnews  \n",
       "509093      ManiaforBeatles  worldnews  \n",
       "509131  cyanocittaetprocyon  worldnews  \n",
       "509106        hamzatahir671  worldnews  \n",
       "509147      ManiaforBeatles  worldnews  \n",
       "509118             bob21doh  worldnews  \n",
       "509144            avatharam  worldnews  \n",
       "509087             krolique  worldnews  \n",
       "509107               tewrld  worldnews  \n",
       "509108      Alexandra-perez  worldnews  \n",
       "509142            Jurryaany  worldnews  \n",
       "509088            Gecko5567  worldnews  \n",
       "509094          maxwellhill  worldnews  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Check vote stats\n",
    "\n",
    "df_avg_count = df.groupby(['date_created']).apply(lambda x: ( x['up_votes'].sum(), x['up_votes'].count()  ) ).reset_index()\n",
    "df_avg_count['sum'] = df_avg_count[0].apply(lambda x: x[0])\n",
    "df_avg_count['cnt'] = df_avg_count[0].apply(lambda x: x[1])\n",
    "\n",
    "df_avg_count = df_avg_count[['date_created', 'sum', 'cnt']].set_index('date_created')\n",
    "\n",
    "\n",
    "date_index =  [ each.strftime(\"%Y-%m-%d\") for each in \\\n",
    "               pd.date_range(start=min_date.strftime(\"%Y-%m-%d\"), end=max_date.strftime(\"%Y-%m-%d\"))]\n",
    "df_avg_count = df_avg_count.reindex(date_index, fill_value=0)\n",
    "#df_avg_count['tol_val'] = df_avg_count.apply(lambda x: x['avg']*x['cnt'], axis=1)\n",
    "\n",
    "def every_seven_day(values):\n",
    "    N = len(values)\n",
    "    output =  np.sum(values)\n",
    "    return output\n",
    "\n",
    "df_avg_count['sum'] = df_avg_count['sum'].rolling(28).apply(every_seven_day, raw=True)\n",
    "df_avg_count['cnt'] = df_avg_count['cnt'].rolling(28).apply(every_seven_day, raw=True)\n",
    "df_avg_count['avg_weekly'] = df_avg_count.apply(lambda x: x['sum']/x['cnt'], axis= 1)\n",
    "df_avg_count.reset_index()[['date_created', 'avg_weekly']]\n",
    "\n",
    "dict_thresh = df_avg_count.reset_index()[['date_created', 'avg_weekly']].iloc[27:, :];\n",
    "first_date = dict_thresh['date_created'].min()\n",
    "dict_thresh = dict(zip(dict_thresh['date_created'],dict_thresh['avg_weekly'] ))\n",
    "\n",
    "df_sel = df[df['date_created']>=first_date].reset_index(drop=True)\n",
    "df_sel['label']     = df_sel.apply(lambda x: x['up_votes'] >= dict_thresh[x['date_created']], axis=1) \n",
    "df_sel['threshold'] = df_sel.apply(lambda x: dict_thresh[x['date_created']], axis=1) \n",
    "\n",
    "#display(df[df['date_created'] == '2016-11-22'])\n",
    "display(  df[df['date_created'] == '2016-11-22'].sort_values('up_votes').tail(20)  )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "1a2d030c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of pos:  53996  Out of:  508901\n",
      "53996\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAbcAAAEWCAYAAADl19mgAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAAzhklEQVR4nO3de3xddZ3v/9cnSZui3IbACFKYVi0jMCClpTaiEAQB8TcPcIBz8HLaM1wqpVUZx3Faz5mZnuMZcBiFeFRKUwu2PiooFgv+VEALGRhJqYWClXIrUGukCAZBVNrm8jl/fNdyr72zd7Kvyc7a7+fjsR97Z+21vmut7CSffG+fr7k7IiIiadI03hcgIiJSbQpuIiKSOgpuIiKSOgpuIiKSOgpuIiKSOi3jfQHlOvjgg33atGnjfRkiIjJOHnrood+4+yH53puwwW3atGls3rx5vC9DRETGiZn9otB7apYUEZHUUXATEZHUUXATEZHUmbB9biIikq2/v5/e3l5279493pdSVVOmTGHq1KlMmjSp6GMU3EREUqK3t5f99tuPadOmYWbjfTlV4e709fXR29vL9OnTiz5OzZIiIimxe/du2traUhPYAMyMtra2kmujCm4iIimSpsAWK+eeFNxEROpFTw9cfXV4loqoz01EpB709MDpp8PevTB5MmzYAO3t431VE5ZqbiIi9aC7OwS2wcHw3N093lc0oSm4iYjUg46OUGNrbg7PHR1jc94qN4Xu2LGDt7/97cyfP5/jjz+eCy64gD/+8Y9s2LCBmTNnctxxx3HxxRezZ88eAJYsWcIxxxzD8ccfz6c//emqXAOoWVJEpD60t4emyO7uENjGokmyRk2hTz75JKtWreLkk0/m4osv5tprr2XFihVs2LCBo446innz5rF8+XLmzZvHd7/7XZ544gnMjFdeeaXye4qo5iYiUi/a22Hp0rHra6tRU+gRRxzBySefDMBHP/pRNmzYwPTp0znqqKMAmD9/Pvfddx/7778/U6ZM4dJLL+W2227jDW94Q1XODwpuIiKNq0ZNocUO3W9paWHTpk2cf/75rF+/nrPPPrsq5wcFNxGRxhU3hX7uc1Udnblz5056oj68m2++mTPOOIMdO3awfft2AL7xjW9w6qmn8vvf/55XX32Vc845h87OTh555JGqnB/U5yYi0tja26veDHr00UezevVq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\n",
      "text/plain": [
       "<Figure size 498.132x307.863 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig1, ax1 = plt.subplots(1, 1, figsize=set_size(width=1000, height_adjust=1, fraction=0.5, subplots=(1,1)))\n",
    "ax1 = df_sel[df_sel['label'] == True].plot.line(x='time_created', y='up_votes', rot=0, style='.', color='red' , ax=ax1)\n",
    "ax1 = df_sel[df_sel['label'] == False].plot.line(x='time_created', y='up_votes', rot=0, style='.', color='blue' , ax=ax1)\n",
    "ax1.legend(['pos','neg'])\n",
    "print(\"Number of pos: \", df_sel['label'].values.sum(), \" Out of: \", df_sel.shape[0])\n",
    "\n",
    "print( df_sel['label'].values.sum() )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd6c2806",
   "metadata": {},
   "source": [
    "#### If we only use average to determine, very popular posts will cause strong bias.\n",
    "#### 075 percientt uqnatin "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8944a9d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "date_index =  [ each.strftime(\"%Y-%m-%d\") for each in \\\n",
    "               pd.date_range(start=min_date.strftime(\"%Y-%m-%d\"), end=max_date.strftime(\"%Y-%m-%d\"))]\n",
    "df_array = df.groupby(['date_created']).apply(lambda x: np.stack(x['up_votes'])).rename('array').reindex(date_index, fill_value=[]).reset_index()\n",
    "#display(df_array)\n",
    "df_array['date_created_ym'] =  df_array['date_created'].apply( lambda x: pd.to_datetime(x).strftime('%Y-%m'))\n",
    "df_ym_quan = df_array.groupby('date_created_ym').apply(lambda x: np.quantile( np.hstack(x['array']), 0.75) ).rename('thresh').reset_index()\n",
    "\n",
    "#df_array['075percent'] = df_array.rolling(28).apply(lambda x: every_seven_day(x['array']), raw=True)\n",
    "\n",
    "#display(df_ym_quan)\n",
    "dict_thresh = dict(zip(df_ym_quan['date_created_ym'],df_ym_quan['thresh'] ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "833dab83",
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
       "      <th>time_created</th>\n",
       "      <th>date_created</th>\n",
       "      <th>up_votes</th>\n",
       "      <th>down_votes</th>\n",
       "      <th>title</th>\n",
       "      <th>over_18</th>\n",
       "      <th>author</th>\n",
       "      <th>category</th>\n",
       "      <th>label</th>\n",
       "      <th>threshold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1203577161</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>US Reactions To Pakistan Election Results Mixed</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1203577230</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>Iraq Unemployment Too Becomes an Epidemic</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1203577396</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>Taxpayers left with Northern Rock  rubbish  th...</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1203577541</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>Britain s FEMALE Spitfire pilots to receive ba...</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1203584599</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>E. Timor s Ramos-Horta Out of Coma</td>\n",
       "      <td>False</td>\n",
       "      <td>PaperLess</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508896</th>\n",
       "      <td>1479816764</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>Heil Trump : Donald Trump s  alt-right  white...</td>\n",
       "      <td>False</td>\n",
       "      <td>nonamenoglory</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508897</th>\n",
       "      <td>1479816772</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>There are people speculating that this could b...</td>\n",
       "      <td>False</td>\n",
       "      <td>SummerRay</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508898</th>\n",
       "      <td>1479817056</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Professor receives Arab Researchers Award</td>\n",
       "      <td>False</td>\n",
       "      <td>AUSharjah</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508899</th>\n",
       "      <td>1479817157</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Nigel Farage attacks response to Trump ambassa...</td>\n",
       "      <td>False</td>\n",
       "      <td>smilyflower</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>508900</th>\n",
       "      <td>1479817346</td>\n",
       "      <td>2016-11-22</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Palestinian wielding knife shot dead in West B...</td>\n",
       "      <td>False</td>\n",
       "      <td>superislam</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>508901 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        time_created date_created  up_votes  down_votes  \\\n",
       "0         1203577161   2008-02-21         3           0   \n",
       "1         1203577230   2008-02-21         3           0   \n",
       "2         1203577396   2008-02-21         9           0   \n",
       "3         1203577541   2008-02-21         8           0   \n",
       "4         1203584599   2008-02-21         4           0   \n",
       "...              ...          ...       ...         ...   \n",
       "508896    1479816764   2016-11-22         5           0   \n",
       "508897    1479816772   2016-11-22         1           0   \n",
       "508898    1479817056   2016-11-22         1           0   \n",
       "508899    1479817157   2016-11-22         1           0   \n",
       "508900    1479817346   2016-11-22         1           0   \n",
       "\n",
       "                                                    title  over_18  \\\n",
       "0         US Reactions To Pakistan Election Results Mixed    False   \n",
       "1               Iraq Unemployment Too Becomes an Epidemic    False   \n",
       "2       Taxpayers left with Northern Rock  rubbish  th...    False   \n",
       "3       Britain s FEMALE Spitfire pilots to receive ba...    False   \n",
       "4                      E. Timor s Ramos-Horta Out of Coma    False   \n",
       "...                                                   ...      ...   \n",
       "508896   Heil Trump : Donald Trump s  alt-right  white...    False   \n",
       "508897  There are people speculating that this could b...    False   \n",
       "508898          Professor receives Arab Researchers Award    False   \n",
       "508899  Nigel Farage attacks response to Trump ambassa...    False   \n",
       "508900  Palestinian wielding knife shot dead in West B...    False   \n",
       "\n",
       "               author   category  label  threshold  \n",
       "0            igeldard  worldnews  False        5.0  \n",
       "1            igeldard  worldnews  False        5.0  \n",
       "2            igeldard  worldnews   True        5.0  \n",
       "3            igeldard  worldnews   True        5.0  \n",
       "4           PaperLess  worldnews  False        5.0  \n",
       "...               ...        ...    ...        ...  \n",
       "508896  nonamenoglory  worldnews  False       21.0  \n",
       "508897      SummerRay  worldnews  False       21.0  \n",
       "508898      AUSharjah  worldnews  False       21.0  \n",
       "508899    smilyflower  worldnews  False       21.0  \n",
       "508900     superislam  worldnews  False       21.0  \n",
       "\n",
       "[508901 rows x 10 columns]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_sel['label']     = df_sel.apply(lambda x: x['up_votes'] >= dict_thresh[x['date_created'][0:7]], axis=1) \n",
    "df_sel['threshold'] = df_sel.apply(lambda x: dict_thresh[x['date_created'][0:7]], axis=1) \n",
    "display(df_sel)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d18aab0c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of pos:  130251  Out of:  508901\n",
      "130251\n"
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
       "      <th>time_created</th>\n",
       "      <th>date_created</th>\n",
       "      <th>up_votes</th>\n",
       "      <th>down_votes</th>\n",
       "      <th>title</th>\n",
       "      <th>over_18</th>\n",
       "      <th>author</th>\n",
       "      <th>category</th>\n",
       "      <th>label</th>\n",
       "      <th>threshold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>506195</th>\n",
       "      <td>1478906001</td>\n",
       "      <td>2016-11-11</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Brazil politicians linked to Petrobras scandal...</td>\n",
       "      <td>False</td>\n",
       "      <td>CodDex</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>507895</th>\n",
       "      <td>1479457557</td>\n",
       "      <td>2016-11-18</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Many Indians are offering to donate their kidn...</td>\n",
       "      <td>False</td>\n",
       "      <td>bitoffreshair</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>507907</th>\n",
       "      <td>1479461501</td>\n",
       "      <td>2016-11-18</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Abe Woos Trump With Golf, Just Like His Grandf...</td>\n",
       "      <td>False</td>\n",
       "      <td>ManiaforBeatles</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>507917</th>\n",
       "      <td>1479464178</td>\n",
       "      <td>2016-11-18</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>David Attenborough gets his own app</td>\n",
       "      <td>False</td>\n",
       "      <td>Heskimo88</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>507919</th>\n",
       "      <td>1479464299</td>\n",
       "      <td>2016-11-18</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Victims  group withdraws from historical child...</td>\n",
       "      <td>False</td>\n",
       "      <td>Ammar_Shabbir</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504278</th>\n",
       "      <td>1478142085</td>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Mosul battle: Militiamen  torture  IS suspects...</td>\n",
       "      <td>False</td>\n",
       "      <td>phuocnguyen286</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504504</th>\n",
       "      <td>1478204036</td>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>At least 19 killed after trains crash in Pakistan</td>\n",
       "      <td>False</td>\n",
       "      <td>Hypatia_Alexandria_</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504280</th>\n",
       "      <td>1478142852</td>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Search Resumes for 44 Missing From Indonesia B...</td>\n",
       "      <td>False</td>\n",
       "      <td>pitinglistrik</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504286</th>\n",
       "      <td>1478144678</td>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Hong Kong may ask Beijing to intervene over pr...</td>\n",
       "      <td>False</td>\n",
       "      <td>cyanocittaetprocyon</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>504294</th>\n",
       "      <td>1478150009</td>\n",
       "      <td>2016-11-03</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>IS leader, Abu Bakr al-Baghdadi, confident in ...</td>\n",
       "      <td>False</td>\n",
       "      <td>just_the_Tayyip</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        time_created date_created  up_votes  down_votes  \\\n",
       "506195    1478906001   2016-11-11         1           0   \n",
       "507895    1479457557   2016-11-18         1           0   \n",
       "507907    1479461501   2016-11-18         1           0   \n",
       "507917    1479464178   2016-11-18         1           0   \n",
       "507919    1479464299   2016-11-18         1           0   \n",
       "...              ...          ...       ...         ...   \n",
       "504278    1478142085   2016-11-03         1           0   \n",
       "504504    1478204036   2016-11-03         1           0   \n",
       "504280    1478142852   2016-11-03         1           0   \n",
       "504286    1478144678   2016-11-03         1           0   \n",
       "504294    1478150009   2016-11-03         1           0   \n",
       "\n",
       "                                                    title  over_18  \\\n",
       "506195  Brazil politicians linked to Petrobras scandal...    False   \n",
       "507895  Many Indians are offering to donate their kidn...    False   \n",
       "507907  Abe Woos Trump With Golf, Just Like His Grandf...    False   \n",
       "507917                David Attenborough gets his own app    False   \n",
       "507919  Victims  group withdraws from historical child...    False   \n",
       "...                                                   ...      ...   \n",
       "504278  Mosul battle: Militiamen  torture  IS suspects...    False   \n",
       "504504  At least 19 killed after trains crash in Pakistan    False   \n",
       "504280  Search Resumes for 44 Missing From Indonesia B...    False   \n",
       "504286  Hong Kong may ask Beijing to intervene over pr...    False   \n",
       "504294  IS leader, Abu Bakr al-Baghdadi, confident in ...    False   \n",
       "\n",
       "                     author   category  label  threshold  \n",
       "506195               CodDex  worldnews  False       21.0  \n",
       "507895        bitoffreshair  worldnews  False       21.0  \n",
       "507907      ManiaforBeatles  worldnews  False       21.0  \n",
       "507917            Heskimo88  worldnews  False       21.0  \n",
       "507919        Ammar_Shabbir  worldnews  False       21.0  \n",
       "...                     ...        ...    ...        ...  \n",
       "504278       phuocnguyen286  worldnews  False       21.0  \n",
       "504504  Hypatia_Alexandria_  worldnews  False       21.0  \n",
       "504280        pitinglistrik  worldnews  False       21.0  \n",
       "504286  cyanocittaetprocyon  worldnews  False       21.0  \n",
       "504294      just_the_Tayyip  worldnews  False       21.0  \n",
       "\n",
       "[300 rows x 10 columns]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
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
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>time_created</th>\n",
       "      <td>5252.0</td>\n",
       "      <td>1.478898e+09</td>\n",
       "      <td>557894.104697</td>\n",
       "      <td>1.477959e+09</td>\n",
       "      <td>1.478374e+09</td>\n",
       "      <td>1.478950e+09</td>\n",
       "      <td>1.479377e+09</td>\n",
       "      <td>1.479817e+09</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>up_votes</th>\n",
       "      <td>5252.0</td>\n",
       "      <td>1.938006e+02</td>\n",
       "      <td>849.791156</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>2.000000e+00</td>\n",
       "      <td>6.000000e+00</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>9.298000e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>down_votes</th>\n",
       "      <td>5252.0</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>0.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>threshold</th>\n",
       "      <td>5252.0</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>2.100000e+01</td>\n",
       "      <td>2.100000e+01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               count          mean            std           min           25%  \\\n",
       "time_created  5252.0  1.478898e+09  557894.104697  1.477959e+09  1.478374e+09   \n",
       "up_votes      5252.0  1.938006e+02     849.791156  0.000000e+00  2.000000e+00   \n",
       "down_votes    5252.0  0.000000e+00       0.000000  0.000000e+00  0.000000e+00   \n",
       "threshold     5252.0  2.100000e+01       0.000000  2.100000e+01  2.100000e+01   \n",
       "\n",
       "                       50%           75%           max  \n",
       "time_created  1.478950e+09  1.479377e+09  1.479817e+09  \n",
       "up_votes      6.000000e+00  2.100000e+01  9.298000e+03  \n",
       "down_votes    0.000000e+00  0.000000e+00  0.000000e+00  \n",
       "threshold     2.100000e+01  2.100000e+01  2.100000e+01  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 498.132x307.863 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig1, ax1 = plt.subplots(1, 1, figsize=set_size(width=1000, height_adjust=1, fraction=0.5, subplots=(1,1)))\n",
    "ax1 = df_sel[df_sel['label'] == True].plot.line(x='time_created', y='up_votes', rot=0, style='.', color='red' , ax=ax1)\n",
    "ax1 = df_sel[df_sel['label'] == False].plot.line(x='time_created', y='up_votes', rot=0, style='.', color='blue' , ax=ax1)\n",
    "ax1.legend(['pos','neg'])\n",
    "ax1.set_yscale('log')\n",
    "print(\"Number of pos: \", df_sel['label'].values.sum(), \" Out of: \", df_sel.shape[0])\n",
    "\n",
    "print( df_sel['label'].values.sum() )\n",
    "display( df_sel[df_sel['date_created'] >= '2016-11-01'].sort_values('up_votes').head(1000).tail(300)  )\n",
    "display(df_sel[df_sel['date_created'] >= '2016-11-01'].describe().transpose())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e36444c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ae52d514",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b680b4f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9f705db2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df = pd.read_csv('Eluvio_DS_Challenge.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "7f81c412",
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
       "      <th>contract</th>\n",
       "      <th>orig</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ain't</td>\n",
       "      <td>am not / are not / is not / has not / have not</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>aren't</td>\n",
       "      <td>are not / am not</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>can't</td>\n",
       "      <td>cannot</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>can't've</td>\n",
       "      <td>cannot have</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>'cause</td>\n",
       "      <td>because</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   contract                                            orig\n",
       "0     ain't  am not / are not / is not / has not / have not\n",
       "1    aren't                                are not / am not\n",
       "2     can't                                          cannot\n",
       "3  can't've                                     cannot have\n",
       "4    'cause                                         because"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# read-in contractions\n",
    "contractions = pd.read_csv('contractions.csv')\n",
    "display(contractions.head(5))\n",
    "contractions = dict(zip(contractions['contract'], contractions['orig']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "231b976f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    '''Text Preprocessing '''\n",
    "    remove_stopwords = True\n",
    "    # Convert words to lower case\n",
    "    text = text.lower()\n",
    "    \n",
    "    # Expand contractions\n",
    "    if True:\n",
    "        text = text.split()\n",
    "        new_text = []\n",
    "        for word in text:\n",
    "            if word in contractions:\n",
    "                new_text.append(contractions[word])\n",
    "            else:\n",
    "                new_text.append(word)\n",
    "        text = \" \".join(new_text)\n",
    "    \n",
    "    # Format words and remove unwanted characters\n",
    "    text = re.sub(r'https?:\\/\\/.*[\\r\\n]*', '', text, flags=re.MULTILINE)\n",
    "    text = re.sub(r'\\<a href', ' ', text)\n",
    "    text = re.sub(r'&amp;', '', text) \n",
    "    text = re.sub(r'[_\"\\-;%()|+&=*%.,!?:#$@\\[\\]/]', ' ', text)\n",
    "    text = re.sub(r'<br />', ' ', text)\n",
    "    text = re.sub(r'\\'', ' ', text)\n",
    "    \n",
    "    # remove stopwords\n",
    "    if remove_stopwords:\n",
    "        text = text.split()\n",
    "        stops = set(stopwords.words(\"english\"))\n",
    "        text = [w for w in text if not w in stops]\n",
    "        text = \" \".join(text)\n",
    "\n",
    "    # Tokenize each word\n",
    "    text =  nltk.WordPunctTokenizer().tokenize(text)\n",
    "    \n",
    "    # Lemmatize each token\n",
    "    lemm = nltk.stem.WordNetLemmatizer()\n",
    "    text = list(map(lambda word:list(map(lemm.lemmatize, word)), text))\n",
    "    \n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "a829e9a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_sel['title_clean'] = df_sel.apply(lambda x: clean_text(x['title']), axis=1)\n",
    "df_sel['title_clean'] = df_sel['title_clean'].apply(lambda x: [\"\".join(each) for each in x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "89af5464",
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
       "      <th>time_created</th>\n",
       "      <th>date_created</th>\n",
       "      <th>up_votes</th>\n",
       "      <th>down_votes</th>\n",
       "      <th>title</th>\n",
       "      <th>over_18</th>\n",
       "      <th>author</th>\n",
       "      <th>category</th>\n",
       "      <th>label</th>\n",
       "      <th>threshold</th>\n",
       "      <th>title_clean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1203577161</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>US Reactions To Pakistan Election Results Mixed</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[us, reactions, pakistan, election, results, m...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1203577230</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>Iraq Unemployment Too Becomes an Epidemic</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[iraq, unemployment, becomes, epidemic]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1203577396</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>Taxpayers left with Northern Rock  rubbish  th...</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[taxpayers, left, northern, rock, rubbish, tha...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1203577541</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>Britain s FEMALE Spitfire pilots to receive ba...</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[britain, female, spitfire, pilots, receive, b...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1203584599</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>E. Timor s Ramos-Horta Out of Coma</td>\n",
       "      <td>False</td>\n",
       "      <td>PaperLess</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[e, timor, ramos, horta, coma]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1203586639</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Biggest Brain Drain From UK In 50 Years</td>\n",
       "      <td>False</td>\n",
       "      <td>ThyLabyrinth</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[biggest, brain, drain, uk, 50, years]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1203589450</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Navy missile hits dying spy satellite, says P...</td>\n",
       "      <td>False</td>\n",
       "      <td>PaperLess</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[navy, missile, hits, dying, spy, satellite, s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1203590839</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>Israeli MP blames tolerance of gays for earthq...</td>\n",
       "      <td>False</td>\n",
       "      <td>moriquendo</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[israeli, mp, blames, tolerance, gays, earthqu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1203591476</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>6</td>\n",
       "      <td>0</td>\n",
       "      <td>Iranian cleric criticises Ahmadinejad for spea...</td>\n",
       "      <td>False</td>\n",
       "      <td>moriquendo</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[iranian, cleric, criticises, ahmadinejad, spe...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1203592553</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>In wake of Kososvo independence senior Palesti...</td>\n",
       "      <td>False</td>\n",
       "      <td>moriquendo</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[wake, kososvo, independence, senior, palestin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>1203595144</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>The Castropedia: Fidel s Cuba in facts and fig...</td>\n",
       "      <td>False</td>\n",
       "      <td>shenglong</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[castropedia, fidel, cuba, facts, figures, wor...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>1203600901</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>We will always turn to his arsenal of ideas</td>\n",
       "      <td>False</td>\n",
       "      <td>GirlGeorge</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[always, turn, arsenal, ideas]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>1203601443</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>Britain says U.S. rendition flights used base ...</td>\n",
       "      <td>False</td>\n",
       "      <td>EllieElliott</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[britain, says, u, rendition, flights, used, b...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>1203605289</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>U.S.-Backed Nabucco Pipeline Takes Baby Steps</td>\n",
       "      <td>False</td>\n",
       "      <td>jips</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[u, backed, nabucco, pipeline, takes, baby, st...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>1203608223</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>28</td>\n",
       "      <td>0</td>\n",
       "      <td>Banda Sisters: In one of India s poorest regio...</td>\n",
       "      <td>False</td>\n",
       "      <td>anonymgrl</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[banda, sisters, one, india, poorest, regions,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>1203609570</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>Actually UK airports were used in renditions (...</td>\n",
       "      <td>False</td>\n",
       "      <td>andy4443</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[actually, uk, airports, used, renditions, dei...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>1203609745</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Computer thefts in Brazil theft  linked to gas...</td>\n",
       "      <td>False</td>\n",
       "      <td>EllieElliott</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[computer, thefts, brazil, theft, linked, gas,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>1203614875</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>So why can t I name my kid  Brfxxccxxmnpccccll...</td>\n",
       "      <td>False</td>\n",
       "      <td>NoSalt</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[name, kid, brfxxccxxmnpcccclll, mmnprxvclmnck...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>1203615660</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>American waterboarding in times gone by: the P...</td>\n",
       "      <td>False</td>\n",
       "      <td>WaterDragon</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>False</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[american, waterboarding, times, gone, philipp...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>1203615997</td>\n",
       "      <td>2008-02-21</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>Australia confirms Iraq troop withdrawal</td>\n",
       "      <td>False</td>\n",
       "      <td>igeldard</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[australia, confirms, iraq, troop, withdrawal]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    time_created date_created  up_votes  down_votes  \\\n",
       "0     1203577161   2008-02-21         3           0   \n",
       "1     1203577230   2008-02-21         3           0   \n",
       "2     1203577396   2008-02-21         9           0   \n",
       "3     1203577541   2008-02-21         8           0   \n",
       "4     1203584599   2008-02-21         4           0   \n",
       "5     1203586639   2008-02-21         0           0   \n",
       "6     1203589450   2008-02-21         0           0   \n",
       "7     1203590839   2008-02-21         5           0   \n",
       "8     1203591476   2008-02-21         6           0   \n",
       "9     1203592553   2008-02-21         2           0   \n",
       "10    1203595144   2008-02-21        13           0   \n",
       "11    1203600901   2008-02-21         1           0   \n",
       "12    1203601443   2008-02-21         4           0   \n",
       "13    1203605289   2008-02-21         3           0   \n",
       "14    1203608223   2008-02-21        28           0   \n",
       "15    1203609570   2008-02-21         2           0   \n",
       "16    1203609745   2008-02-21         1           0   \n",
       "17    1203614875   2008-02-21         0           0   \n",
       "18    1203615660   2008-02-21         4           0   \n",
       "19    1203615997   2008-02-21         8           0   \n",
       "\n",
       "                                                title  over_18        author  \\\n",
       "0     US Reactions To Pakistan Election Results Mixed    False      igeldard   \n",
       "1           Iraq Unemployment Too Becomes an Epidemic    False      igeldard   \n",
       "2   Taxpayers left with Northern Rock  rubbish  th...    False      igeldard   \n",
       "3   Britain s FEMALE Spitfire pilots to receive ba...    False      igeldard   \n",
       "4                  E. Timor s Ramos-Horta Out of Coma    False     PaperLess   \n",
       "5             Biggest Brain Drain From UK In 50 Years    False  ThyLabyrinth   \n",
       "6    Navy missile hits dying spy satellite, says P...    False     PaperLess   \n",
       "7   Israeli MP blames tolerance of gays for earthq...    False    moriquendo   \n",
       "8   Iranian cleric criticises Ahmadinejad for spea...    False    moriquendo   \n",
       "9   In wake of Kososvo independence senior Palesti...    False    moriquendo   \n",
       "10  The Castropedia: Fidel s Cuba in facts and fig...    False     shenglong   \n",
       "11        We will always turn to his arsenal of ideas    False    GirlGeorge   \n",
       "12  Britain says U.S. rendition flights used base ...    False  EllieElliott   \n",
       "13      U.S.-Backed Nabucco Pipeline Takes Baby Steps    False          jips   \n",
       "14  Banda Sisters: In one of India s poorest regio...    False     anonymgrl   \n",
       "15  Actually UK airports were used in renditions (...    False      andy4443   \n",
       "16  Computer thefts in Brazil theft  linked to gas...    False  EllieElliott   \n",
       "17  So why can t I name my kid  Brfxxccxxmnpccccll...    False        NoSalt   \n",
       "18  American waterboarding in times gone by: the P...    False   WaterDragon   \n",
       "19           Australia confirms Iraq troop withdrawal    False      igeldard   \n",
       "\n",
       "     category  label  threshold  \\\n",
       "0   worldnews  False        5.0   \n",
       "1   worldnews  False        5.0   \n",
       "2   worldnews   True        5.0   \n",
       "3   worldnews   True        5.0   \n",
       "4   worldnews  False        5.0   \n",
       "5   worldnews  False        5.0   \n",
       "6   worldnews  False        5.0   \n",
       "7   worldnews   True        5.0   \n",
       "8   worldnews   True        5.0   \n",
       "9   worldnews  False        5.0   \n",
       "10  worldnews   True        5.0   \n",
       "11  worldnews  False        5.0   \n",
       "12  worldnews  False        5.0   \n",
       "13  worldnews  False        5.0   \n",
       "14  worldnews   True        5.0   \n",
       "15  worldnews  False        5.0   \n",
       "16  worldnews  False        5.0   \n",
       "17  worldnews  False        5.0   \n",
       "18  worldnews  False        5.0   \n",
       "19  worldnews   True        5.0   \n",
       "\n",
       "                                          title_clean  \n",
       "0   [us, reactions, pakistan, election, results, m...  \n",
       "1             [iraq, unemployment, becomes, epidemic]  \n",
       "2   [taxpayers, left, northern, rock, rubbish, tha...  \n",
       "3   [britain, female, spitfire, pilots, receive, b...  \n",
       "4                      [e, timor, ramos, horta, coma]  \n",
       "5              [biggest, brain, drain, uk, 50, years]  \n",
       "6   [navy, missile, hits, dying, spy, satellite, s...  \n",
       "7   [israeli, mp, blames, tolerance, gays, earthqu...  \n",
       "8   [iranian, cleric, criticises, ahmadinejad, spe...  \n",
       "9   [wake, kososvo, independence, senior, palestin...  \n",
       "10  [castropedia, fidel, cuba, facts, figures, wor...  \n",
       "11                     [always, turn, arsenal, ideas]  \n",
       "12  [britain, says, u, rendition, flights, used, b...  \n",
       "13  [u, backed, nabucco, pipeline, takes, baby, st...  \n",
       "14  [banda, sisters, one, india, poorest, regions,...  \n",
       "15  [actually, uk, airports, used, renditions, dei...  \n",
       "16  [computer, thefts, brazil, theft, linked, gas,...  \n",
       "17  [name, kid, brfxxccxxmnpcccclll, mmnprxvclmnck...  \n",
       "18  [american, waterboarding, times, gone, philipp...  \n",
       "19     [australia, confirms, iraq, troop, withdrawal]  "
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sel.head(20)\n",
    "#df_sel.to_csv('Eluvio_DS_Challenge_processes.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf334135",
   "metadata": {},
   "source": [
    "# Work on simple models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "407b41a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#============== Packages for word2vec ==============#\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "\n",
    "#============== Packages for classification ==============#\n",
    "from sklearn.linear_model import LinearRegression, PoissonRegressor\n",
    "from sklearn.preprocessing import normalize\n",
    "\n",
    "from sklearn.model_selection import KFold\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import balanced_accuracy_score\n",
    "from sklearn.svm import SVC\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "519a9e94",
   "metadata": {},
   "outputs": [],
   "source": [
    "#========= Read in =========#\n",
    "df = pd.read_csv('Eluvio_DS_Challenge_processes.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "8f49824d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#========= TDIDF =========#\n",
    "\n",
    "bow_converter = CountVectorizer(tokenizer=lambda doc: doc)\n",
    "x = bow_converter.fit_transform(df['title_clean'])\n",
    "\n",
    "words = bow_converter.get_feature_names()\n",
    "\n",
    "bigram_converter = CountVectorizer(tokenizer=lambda doc: doc,ngram_range=[2,2]) \n",
    "trigram_converter = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3])\n",
    "\n",
    "tfidf_transform = TfidfTransformer(norm=None)\n",
    "X_tfidf = tfidf_transform.fit_transform(x)\n",
    "\n",
    "X_tfidf = normalize(X_tfidf,axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "5206cb3c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "n_pos:  130251 n_neg:  378650\n",
      "ls_b_accu:  0.500015057722\n"
     ]
    }
   ],
   "source": [
    "#========= Define Binary Labels =========#\n",
    "\n",
    "#========= Median positive and negative =========#\n",
    "y_true = df['label']\n",
    "print(\"n_pos: \", (y_true==1).sum(), \"n_neg: \", (y_true==0).sum() ) \n",
    "#========= Kfold Logistic Regression =========#\n",
    "\n",
    "kf = KFold(n_splits=3)\n",
    "\n",
    "ls_b_accu = []\n",
    "for train_index, test_index in kf.split(X_tfidf):\n",
    "    clf = LogisticRegression(random_state=0).fit(X_tfidf[train_index,:], y_true[train_index] )\n",
    "    y_pred = clf.predict(X_tfidf[test_index, :])\n",
    "    \n",
    "    baccu = balanced_accuracy_score(y_true[test_index], y_pred)\n",
    "    \n",
    "    #print(baccu)\n",
    "    ls_b_accu.append(baccu)\n",
    "print(\"ls_b_accu: \", np.mean(ls_b_accu))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "0e3d15f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "n_pos:  130251 n_neg:  378650\n",
      "{False: False, True: True}\n"
     ]
    }
   ],
   "source": [
    "#========= Median positive and negative =========#\n",
    "y_true = df['label']\n",
    "print(\"n_pos: \", (y_true==1).sum(), \"n_neg: \", (y_true==0).sum() ) \n",
    "\n",
    "# example of a model defined with the sequential api\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_true, test_size=0.33, random_state=42)\n",
    "\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "import tensorflow as tf\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.utils import class_weight\n",
    "from tensorflow.keras.metrics import Metric\n",
    "from sklearn.metrics import balanced_accuracy_score\n",
    "\n",
    "class_weights = class_weight.compute_class_weight('balanced',\n",
    "                                                 np.unique(y_train),\n",
    "                                                 y_train)\n",
    "class_weights = dict(zip( np.unique(y_train), class_weights))\n",
    "sample\n",
    "print(dict(zip( np.unique(y_train), class_weights)) )\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "1ac59464",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_20\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_58 (Dense)             (None, 10)                3610      \n",
      "_________________________________________________________________\n",
      "dense_59 (Dense)             (None, 2)                 22        \n",
      "_________________________________________________________________\n",
      "dense_60 (Dense)             (None, 1)                 3         \n",
      "=================================================================\n",
      "Total params: 3,635\n",
      "Trainable params: 3,635\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Define a simple sequential model\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)))\n",
    "    model.add(Dense(2, activation= 'relu', kernel_initializer='he_normal'))\n",
    "    model.add(Dense(1, activation='sigmoid'))\n",
    "\n",
    "    # compile the model\n",
    "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')])\n",
    "    \n",
    "    return model\n",
    "\n",
    "# Create a basic model instance\n",
    "model = create_model()\n",
    "\n",
    "# Display the model's architecture\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "9375f10d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 11s - loss: 0.6821 - binary_accuracy: 0.5621 - val_loss: 0.6499 - val_binary_accuracy: 0.6640\n",
      "Epoch 2/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 11s - loss: 0.6775 - binary_accuracy: 0.6104 - val_loss: 0.6909 - val_binary_accuracy: 0.5821\n",
      "Epoch 3/70\n",
      "5328/5328 - 10s - loss: 0.6771 - binary_accuracy: 0.6138 - val_loss: 0.6511 - val_binary_accuracy: 0.6708\n",
      "Epoch 4/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/01_sim_tdidfNN_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 10s - loss: 0.6769 - binary_accuracy: 0.6165 - val_loss: 0.7179 - val_binary_accuracy: 0.5326\n",
      "Epoch 5/70\n",
      "5328/5328 - 10s - loss: 0.6767 - binary_accuracy: 0.6187 - val_loss: 0.6601 - val_binary_accuracy: 0.6482\n",
      "Epoch 6/70\n",
      "5328/5328 - 11s - loss: 0.6765 - binary_accuracy: 0.6217 - val_loss: 0.6790 - val_binary_accuracy: 0.6202\n",
      "Epoch 7/70\n",
      "5328/5328 - 10s - loss: 0.6764 - binary_accuracy: 0.6205 - val_loss: 0.6722 - val_binary_accuracy: 0.6378\n"
     ]
    }
   ],
   "source": [
    "# Callback define\n",
    "patience = 3; epochs = 70;\n",
    "checkpoint_filepath = './check_point/01_sim_tdidfNN_mdl.ckpt';\n",
    "\n",
    "early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='max')\n",
    "\n",
    "model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(\n",
    "    filepath=checkpoint_filepath,\n",
    "    save_weights_only=False, monitor='val_loss', mode='max', save_best_only=True)\n",
    "\n",
    "# fit the model\n",
    "history = model.fit(X_train.toarray(), y_train, validation_data=(X_test.toarray(), y_test), \\\n",
    "                    epochs=epochs, batch_size=64, verbose=2, \\\n",
    "                    callbacks=[early_stopping, model_checkpoint_callback], class_weight=class_weights)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "25f11c10",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy_score:  0.53257154426\n",
      "balanced_accuracy_score:  0.570787778678\n",
      "61500 63401 15098 27939\n",
      "(167938, 1)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Load model and evaluate on test\n",
    "model = tf.keras.models.load_model('./check_point/01_sim_tdidfNN_mdl.ckpt')\n",
    "pred_test = model.predict(X_test.toarray()) > 0.5;\n",
    "\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score\n",
    "print( \"accuracy_score: \", accuracy_score(y_test, pred_test) )\n",
    "print( \"balanced_accuracy_score: \", balanced_accuracy_score(y_test, pred_test) )\n",
    "\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()\n",
    "print(tn, fp, fn, tp)\n",
    "print(pred_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a20a39a3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e54909b2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c77e99e7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "12f3accc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_22\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "keras_layer_3 (KerasLayer)   (None, 50)                48190600  \n",
      "_________________________________________________________________\n",
      "dense_63 (Dense)             (None, 10)                510       \n",
      "_________________________________________________________________\n",
      "dense_64 (Dense)             (None, 10)                110       \n",
      "_________________________________________________________________\n",
      "dense_65 (Dense)             (None, 10)                110       \n",
      "_________________________________________________________________\n",
      "dense_66 (Dense)             (None, 1)                 11        \n",
      "=================================================================\n",
      "Total params: 48,191,341\n",
      "Trainable params: 48,191,341\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Define a sequential model\n",
    "# Use Token based text embedding trained on English Google News 7B corpus\n",
    "def create_model(): \n",
    "    \n",
    "    embedding = \"https://tfhub.dev/google/nnlm-en-dim50/2\"\n",
    "    hub_layer = hub.KerasLayer(embedding, input_shape=[], \n",
    "                           dtype=tf.string, trainable=True)\n",
    "    model = tf.keras.Sequential()\n",
    "    model.add(hub_layer)\n",
    "    model.add(tf.keras.layers.Dense(10, activation='relu'))\n",
    "    model.add(tf.keras.layers.Dense(10, activation='relu'))\n",
    "    model.add(tf.keras.layers.Dense(10, activation='relu'))\n",
    "    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))\n",
    "    \n",
    "    # compile the model\n",
    "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')])\n",
    "    \n",
    "    return model\n",
    "    \n",
    "# Create a basic model instance\n",
    "model = create_model()\n",
    "\n",
    "# Display the model's architecture\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "a79e60f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.6714 - binary_accuracy: 0.5814 - val_loss: 0.6775 - val_binary_accuracy: 0.5814\n",
      "Epoch 2/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.6598 - binary_accuracy: 0.6068 - val_loss: 0.6882 - val_binary_accuracy: 0.5630\n",
      "Epoch 3/70\n",
      "5328/5328 - 53s - loss: 0.6531 - binary_accuracy: 0.6098 - val_loss: 0.6606 - val_binary_accuracy: 0.6006\n",
      "Epoch 4/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.6449 - binary_accuracy: 0.6139 - val_loss: 0.6937 - val_binary_accuracy: 0.5454\n",
      "Epoch 5/70\n",
      "5328/5328 - 54s - loss: 0.6361 - binary_accuracy: 0.6154 - val_loss: 0.6884 - val_binary_accuracy: 0.5571\n",
      "Epoch 6/70\n",
      "WARNING:tensorflow:5 out of the last 5 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:5 out of the last 5 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.6276 - binary_accuracy: 0.6207 - val_loss: 0.7045 - val_binary_accuracy: 0.5374\n",
      "Epoch 7/70\n",
      "5328/5328 - 54s - loss: 0.6203 - binary_accuracy: 0.6244 - val_loss: 0.6841 - val_binary_accuracy: 0.5613\n",
      "Epoch 8/70\n",
      "5328/5328 - 54s - loss: 0.6136 - binary_accuracy: 0.6272 - val_loss: 0.6892 - val_binary_accuracy: 0.5569\n",
      "Epoch 9/70\n",
      "WARNING:tensorflow:6 out of the last 6 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:6 out of the last 6 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.6075 - binary_accuracy: 0.6277 - val_loss: 0.7074 - val_binary_accuracy: 0.5399\n",
      "Epoch 10/70\n",
      "5328/5328 - 53s - loss: 0.6027 - binary_accuracy: 0.6293 - val_loss: 0.7051 - val_binary_accuracy: 0.5445\n",
      "Epoch 11/70\n",
      "WARNING:tensorflow:7 out of the last 7 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:7 out of the last 7 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5978 - binary_accuracy: 0.6305 - val_loss: 0.7252 - val_binary_accuracy: 0.5291\n",
      "Epoch 12/70\n",
      "5328/5328 - 54s - loss: 0.5941 - binary_accuracy: 0.6329 - val_loss: 0.6903 - val_binary_accuracy: 0.5743\n",
      "Epoch 13/70\n",
      "5328/5328 - 54s - loss: 0.5904 - binary_accuracy: 0.6308 - val_loss: 0.7245 - val_binary_accuracy: 0.5327\n",
      "Epoch 14/70\n",
      "WARNING:tensorflow:8 out of the last 8 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:8 out of the last 8 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5869 - binary_accuracy: 0.6309 - val_loss: 0.7290 - val_binary_accuracy: 0.5371\n",
      "Epoch 15/70\n",
      "5328/5328 - 54s - loss: 0.5841 - binary_accuracy: 0.6322 - val_loss: 0.7255 - val_binary_accuracy: 0.5391\n",
      "Epoch 16/70\n",
      "5328/5328 - 54s - loss: 0.5813 - binary_accuracy: 0.6340 - val_loss: 0.7270 - val_binary_accuracy: 0.5526\n",
      "Epoch 17/70\n",
      "WARNING:tensorflow:9 out of the last 9 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:9 out of the last 9 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5787 - binary_accuracy: 0.6337 - val_loss: 0.7497 - val_binary_accuracy: 0.5506\n",
      "Epoch 18/70\n",
      "5328/5328 - 53s - loss: 0.5771 - binary_accuracy: 0.6336 - val_loss: 0.7453 - val_binary_accuracy: 0.5375\n",
      "Epoch 19/70\n",
      "5328/5328 - 54s - loss: 0.5749 - binary_accuracy: 0.6335 - val_loss: 0.7454 - val_binary_accuracy: 0.5568\n",
      "Epoch 20/70\n",
      "WARNING:tensorflow:10 out of the last 10 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:10 out of the last 10 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5729 - binary_accuracy: 0.6337 - val_loss: 0.7605 - val_binary_accuracy: 0.5468\n",
      "Epoch 21/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5713 - binary_accuracy: 0.6338 - val_loss: 0.7699 - val_binary_accuracy: 0.5408\n",
      "Epoch 22/70\n",
      "5328/5328 - 54s - loss: 0.5694 - binary_accuracy: 0.6342 - val_loss: 0.7689 - val_binary_accuracy: 0.5336\n",
      "Epoch 23/70\n",
      "5328/5328 - 54s - loss: 0.5680 - binary_accuracy: 0.6333 - val_loss: 0.7660 - val_binary_accuracy: 0.5525\n",
      "Epoch 24/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5663 - binary_accuracy: 0.6344 - val_loss: 0.7733 - val_binary_accuracy: 0.5369\n",
      "Epoch 25/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5654 - binary_accuracy: 0.6338 - val_loss: 0.7805 - val_binary_accuracy: 0.5425\n",
      "Epoch 26/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5639 - binary_accuracy: 0.6335 - val_loss: 0.7946 - val_binary_accuracy: 0.5477\n",
      "Epoch 27/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5626 - binary_accuracy: 0.6329 - val_loss: 0.8023 - val_binary_accuracy: 0.5412\n",
      "Epoch 28/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5617 - binary_accuracy: 0.6342 - val_loss: 0.8134 - val_binary_accuracy: 0.5472\n",
      "Epoch 29/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5599 - binary_accuracy: 0.6345 - val_loss: 0.8211 - val_binary_accuracy: 0.5235\n",
      "Epoch 30/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5600 - binary_accuracy: 0.6339 - val_loss: 0.8302 - val_binary_accuracy: 0.5409\n",
      "Epoch 31/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5590 - binary_accuracy: 0.6342 - val_loss: 0.8318 - val_binary_accuracy: 0.5233\n",
      "Epoch 32/70\n",
      "5328/5328 - 54s - loss: 0.5578 - binary_accuracy: 0.6350 - val_loss: 0.8295 - val_binary_accuracy: 0.5329\n",
      "Epoch 33/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5572 - binary_accuracy: 0.6350 - val_loss: 0.8348 - val_binary_accuracy: 0.5310\n",
      "Epoch 34/70\n",
      "5328/5328 - 54s - loss: 0.5561 - binary_accuracy: 0.6342 - val_loss: 0.8298 - val_binary_accuracy: 0.5277\n",
      "Epoch 35/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 56s - loss: 0.5548 - binary_accuracy: 0.6363 - val_loss: 0.8376 - val_binary_accuracy: 0.5391\n",
      "Epoch 36/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5543 - binary_accuracy: 0.6366 - val_loss: 0.8393 - val_binary_accuracy: 0.5354\n",
      "Epoch 37/70\n",
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:11 out of the last 11 calls to <function recreate_function.<locals>.restored_function_body at 0x7f35ca50d0e0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings is likely due to passing python objects instead of tensors. Also, tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. Please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for more details.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/02_pre_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 55s - loss: 0.5538 - binary_accuracy: 0.6372 - val_loss: 0.8586 - val_binary_accuracy: 0.5237\n",
      "Epoch 38/70\n",
      "5328/5328 - 53s - loss: 0.5531 - binary_accuracy: 0.6366 - val_loss: 0.8521 - val_binary_accuracy: 0.5171\n",
      "Epoch 39/70\n",
      "5328/5328 - 54s - loss: 0.5527 - binary_accuracy: 0.6362 - val_loss: 0.8445 - val_binary_accuracy: 0.5150\n",
      "Epoch 40/70\n",
      "5328/5328 - 54s - loss: 0.5509 - binary_accuracy: 0.6369 - val_loss: 0.8484 - val_binary_accuracy: 0.5216\n"
     ]
    }
   ],
   "source": [
    "# Train and Test seprate\n",
    "X_train, X_test, y_train, y_test = train_test_split(df['title_clean'], y_true, test_size=0.33, random_state=42)\n",
    "\n",
    "# Callback define\n",
    "patience = 3; epochs = 70;\n",
    "checkpoint_filepath = './check_point/02_pre_nnlm-en-dim50_mdl.ckpt';\n",
    "\n",
    "early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='max')\n",
    "\n",
    "model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(\n",
    "    filepath=checkpoint_filepath,\n",
    "    save_weights_only=False, monitor='val_loss', mode='max', save_best_only=True)\n",
    "\n",
    "\n",
    "# fit the model\n",
    "history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), \\\n",
    "                    epochs=epochs, batch_size=64, verbose=2, \\\n",
    "                    callbacks=[early_stopping, model_checkpoint_callback], class_weight=class_weights)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "830aac05",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy_score:  0.523687313175\n",
      "balanced_accuracy_score:  0.555273779482\n",
      "61261 63640 16351 26686\n",
      "(167938, 1)\n"
     ]
    }
   ],
   "source": [
    "# Load model and evaluate on test\n",
    "model = tf.keras.models.load_model('./check_point/02_pre_nnlm-en-dim50_mdl.ckpt')\n",
    "pred_test = model.predict(X_test) > 0.5;\n",
    "\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score\n",
    "print( \"accuracy_score: \", accuracy_score(y_test, pred_test) )\n",
    "print( \"balanced_accuracy_score: \", balanced_accuracy_score(y_test, pred_test) )\n",
    "\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()\n",
    "print(tn, fp, fn, tp)\n",
    "print(pred_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14c4f6e9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c3e64dd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "56f4dc65",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_29\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_70 (Dense)             (None, 10)                510       \n",
      "_________________________________________________________________\n",
      "dense_71 (Dense)             (None, 2)                 22        \n",
      "_________________________________________________________________\n",
      "dense_72 (Dense)             (None, 1)                 3         \n",
      "=================================================================\n",
      "Total params: 535\n",
      "Trainable params: 535\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Define a sequential model\n",
    "# Use Token based text embedding trained on English Google News 7B corpus\n",
    "# Use pretrain embedding\n",
    "\n",
    "embed = hub.load(\"https://tfhub.dev/google/nnlm-en-dim50/2\")\n",
    "X_train = embed(df['title_clean'].values).numpy()\n",
    "\n",
    "# Train and Test seprate\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_train, y_true, test_size=0.33, random_state=42)\n",
    "\n",
    "# Define a simple sequential model\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)))\n",
    "    model.add(Dense( 2, activation= 'relu', kernel_initializer='he_normal'))\n",
    "    model.add(Dense( 1, activation='sigmoid'))\n",
    "\n",
    "    # compile the model\n",
    "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')])\n",
    "    \n",
    "    return model\n",
    "\n",
    "# Create a basic model instance\n",
    "model = create_model()\n",
    "\n",
    "# Display the model's architecture\n",
    "model.summary()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "de9e33a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 10s - loss: 0.6843 - binary_accuracy: 0.5673 - val_loss: 0.6699 - val_binary_accuracy: 0.6410\n",
      "Epoch 2/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 10s - loss: 0.6823 - binary_accuracy: 0.6026 - val_loss: 0.6783 - val_binary_accuracy: 0.6170\n",
      "Epoch 3/70\n",
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: ./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5328/5328 - 10s - loss: 0.6820 - binary_accuracy: 0.6045 - val_loss: 0.6934 - val_binary_accuracy: 0.5721\n",
      "Epoch 4/70\n",
      "5328/5328 - 10s - loss: 0.6818 - binary_accuracy: 0.6034 - val_loss: 0.6821 - val_binary_accuracy: 0.6046\n",
      "Epoch 5/70\n",
      "5328/5328 - 10s - loss: 0.6817 - binary_accuracy: 0.6047 - val_loss: 0.6757 - val_binary_accuracy: 0.6218\n",
      "Epoch 6/70\n",
      "5328/5328 - 10s - loss: 0.6815 - binary_accuracy: 0.6022 - val_loss: 0.6851 - val_binary_accuracy: 0.5918\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Callback define\n",
    "patience = 3; epochs = 70;\n",
    "checkpoint_filepath = './check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt';\n",
    "\n",
    "early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='max')\n",
    "\n",
    "model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(\n",
    "    filepath=checkpoint_filepath,\n",
    "    save_weights_only=False, monitor='val_loss', mode='max', save_best_only=True)\n",
    "\n",
    "\n",
    "# fit the model\n",
    "history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), \\\n",
    "                    epochs=epochs, batch_size=64, verbose=2, \\\n",
    "                    callbacks=[early_stopping, model_checkpoint_callback], class_weight=class_weights)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "93d9494b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy_score:  0.572092081602\n",
      "balanced_accuracy_score:  0.560729923598\n",
      "72947 51954 19908 23129\n",
      "(167938, 1)\n"
     ]
    }
   ],
   "source": [
    "# Load model and evaluate on test\n",
    "model = tf.keras.models.load_model('./check_point/03_preEmbed_nnlm-en-dim50_mdl.ckpt')\n",
    "pred_test = model.predict(X_test) > 0.5;\n",
    "\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score\n",
    "print( \"accuracy_score: \", accuracy_score(y_test, pred_test) )\n",
    "print( \"balanced_accuracy_score: \", balanced_accuracy_score(y_test, pred_test) )\n",
    "\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()\n",
    "print(tn, fp, fn, tp)\n",
    "print(pred_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ef98117",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bcfb9383",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ead7a446",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd64b0f3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "068863cf",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a127fc1f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05e36685",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "492da904",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(df['title_clean'], y_true, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "498cf156",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',\n",
    "              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e56e1871",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nhistory = model.fit(x=X_train, y=y_train,\\n                    epochs=150, batch_size=32, verbose=2, class_weight=class_weights,\\n                    validation_data=(X_test, y_test))\\n'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "history = model.fit(x=X_train, y=y_train,\n",
    "                    epochs=150, batch_size=32, verbose=2, class_weight=class_weights,\n",
    "                    validation_data=(X_test, y_test))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "0a3c909a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.01530321910933341"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.score(X_tfidf, df['up_votes'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "38d6d5c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "112.23628642679246"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reg.predict(X_tfidf).mean()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "68983a27",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5138.749231150503"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "112.23628337352426"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display(X_tfidf.sum(0).mean() )\n",
    "display(df['up_votes'].mean() )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "783aaa1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neural_network import MLPRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "ca259929",
   "metadata": {},
   "outputs": [],
   "source": [
    "#regr = MLPRegressor(random_state=1, max_iter=500).fit(X_tfidf, df['up_votes'].values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "id": "01bea477",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "47902\n",
      "(509236, 9)\n"
     ]
    }
   ],
   "source": [
    "print((df['up_votes'].values > df['up_votes'].values.mean()).sum())\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "4be149c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_true = (df['up_votes'].values>np.quantile( df['up_votes'].values, 0.50))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "208673bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "clf = LogisticRegression(random_state=0).fit(X_tfidf, y_true )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "e37e31ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = clf.predict(X_tfidf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "f49ab569",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "729f1224",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "       False       0.59      0.75      0.66    275854\n",
      "        True       0.57      0.38      0.46    233382\n",
      "\n",
      "    accuracy                           0.58    509236\n",
      "   macro avg       0.58      0.57      0.56    509236\n",
      "weighted avg       0.58      0.58      0.57    509236\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_true, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c70695c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "# Customize my own metrics\n",
    "\n",
    "class BalancedAccuracy(Metric):\n",
    "    def __init__(self, name=\"balanced_accuracy\", **kwargs):\n",
    "        super(BalancedAccuracy, self).__init__(name=name, **kwargs)\n",
    "        self.balanced_accuracy = self.add_weight(name=\"ctp\", initializer=\"zeros\")\n",
    "\n",
    "    def update_state(self, y_true, y_pred, sample_weight=None):\n",
    "        y_pred = y_pred.nupmy()\n",
    "        y_true = y_true.nupmy()\n",
    "\n",
    "        value = balanced_accuracy_score(y_true, y_pred, sample_weight)\n",
    "        #values = tf.multiply(values, sample_weight)\n",
    "        self.balanced_accuracy.assign_add((value))\n",
    "\n",
    "    def result(self):\n",
    "        return self.balanced_accuracy\n",
    "\n",
    "    def reset_states(self):\n",
    "        # The state of the metric will be reset at the start of each epoch.\n",
    "        self.balanced_accuracy.assign(0.0)\n",
    "'''"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (NetHawkes)",
   "language": "python",
   "name": "nethawkes"
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
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
