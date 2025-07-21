 1/1:
# install jupyterthemes
pip install jupyterthemes

# upgrade to latest version
pip install --upgrade jupyterthemes
 1/2:
# install jupyterthemes
!pip install jupyterthemes

# upgrade to latest version
!pip install --upgrade jupyterthemes
 2/1:
# install jupyterthemes
pip install jupyterthemes

# upgrade to latest version
pip install --upgrade jupyterthemes
 3/1: wbc.head()
 5/1:
from ipywidgets import IntSlider
from ipywidgets.embed import embed_minimal_html

slider = IntSlider(value=40)
embed_minimal_html('export.html', views=[slider], title='Widgets export')
 5/2:
from ipywidgets import IntSlider
from ipywidgets.embed import embed_minimal_html

slider = IntSlider(value=40)
embed_minimal_html('export.html', views=[slider], title='Widgets export')
 5/3:
from ipywidgets import IntSlider
from ipywidgets.embed import embed_minimal_html

slider = IntSlider(value=40)
embed_minimal_html('export.html', views=[slider], title='Widgets export')
 5/4:
from ipywidgets import IntSlider
from ipywidgets.embed import embed_minimal_html

slider = IntSlider(value=40)
embed_minimal_html('export.html', views=[slider], title='Widgets export')
display(slider)
14/1: ## Median times
17/1: %pip install pycox
17/2:
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

#pycox is built on top of PyTorch and torchtuples, where the latter is just a simple way of training neural nets with less boilerplate code.

import torch # For building the networks 
import torchtuples as tt # Some useful functions

#We import the metabric dataset, the LogisticHazard method (paper_link) also known as Nnet-survival, and EvalSurv which simplifies the evaluation procedure at the end.

#You can alternatively replace LogisticHazard with, for example, PMF or DeepHitSingle, which should both work in this notebook.

from pycox.datasets import metabric
# from pycox.models import LogisticHazard
# from pycox.models import PMF
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)
17/3: !pip install sklearn_pandas
17/4:
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

#pycox is built on top of PyTorch and torchtuples, where the latter is just a simple way of training neural nets with less boilerplate code.

import torch # For building the networks 
import torchtuples as tt # Some useful functions

#We import the metabric dataset, the LogisticHazard method (paper_link) also known as Nnet-survival, and EvalSurv which simplifies the evaluation procedure at the end.

#You can alternatively replace LogisticHazard with, for example, PMF or DeepHitSingle, which should both work in this notebook.

from pycox.datasets import metabric
# from pycox.models import LogisticHazard
# from pycox.models import PMF
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)
17/5: !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
17/6:
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

#pycox is built on top of PyTorch and torchtuples, where the latter is just a simple way of training neural nets with less boilerplate code.

import torch # For building the networks 
import torchtuples as tt # Some useful functions

#We import the metabric dataset, the LogisticHazard method (paper_link) also known as Nnet-survival, and EvalSurv which simplifies the evaluation procedure at the end.

#You can alternatively replace LogisticHazard with, for example, PMF or DeepHitSingle, which should both work in this notebook.

from pycox.datasets import metabric
# from pycox.models import LogisticHazard
# from pycox.models import PMF
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)
17/7:

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
df_train.head()
17/8:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
17/9:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
17/10:
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
17/11:
num_durations = 10

labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
# labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
17/12:
num_durations = 10

# labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
17/13: type(labtrans)
17/14: type(labtrans)
17/15: labtrans.cuts
17/16: y_train
17/17: labtrans.cuts[y_train[0]]
17/18:
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
17/19:
# model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
# model = PMF(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
model = DeepHitSingle(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
17/20:
batch_size = 256
epochs = 100
callbacks = [tt.cb.EarlyStopping()]
17/21: log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
17/22: _ = log.plot()
17/23: surv = model.predict_surv_df(x_test)
17/24:
surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
17/25: surv = model.interpolate(10).predict_surv_df(x_test)
17/26:
surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')
17/27: ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
17/28: ev.concordance_td('antolini')
18/1: from lifelines.plotting import plot_lifetimes
17/29: !pip install lifelines
17/30: !pip install lifelines
17/31: from lifelines.plotting import plot_lifetimes
17/32:
ax = plot_lifetimes(df_train.head().duration, event_observed=df_train.head().event)

ax.set_xlim(0, 25)
ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Survival Metabric data")
print("Observed lifetimes at time %d:\n" % (CURRENT_TIME), observed_lifetimes)
17/33:
ax = plot_lifetimes(df_train.head().duration, event_observed=df_train.head().event)

ax.set_xlim(0, 25)
ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Metabric data")
ax.set_ylabel("Patient no.")
17/34:
N = 20
ax = plot_lifetimes(df_train.head(N).duration, event_observed=df_train.head(N).event)

ax.set_xlim(0, 25)
ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Metabric data")
ax.set_ylabel("Patient no.")
17/35:
N = 20
ax = plot_lifetimes(df_train.head(N).duration, event_observed=df_train.head(N).event)

#ax.set_xlim(0, 25)
#ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Metabric data")
ax.set_ylabel("Patient no.")
17/36:
N = 20
ax = plot_lifetimes(df_train.head(N).duration, event_observed=df_train.head(N).event)

#ax.set_xlim(0, 25)
#ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Metabric data")
ax.set_ylabel("Patient no.")
ax.legend()
17/37:
N = 20
ax = plot_lifetimes(df_train.head(N).duration, event_observed=df_train.head(N).event)

#ax.set_xlim(0, 25)
#ax.vlines(10, 0, 30, lw=2, linestyles='--')
ax.set_xlabel("Time")
ax.set_title("Metabric data")
ax.set_ylabel("Patient no.")
17/38: surv
17/39: model
17/40: dir(model)
17/41:
import pycox
dir(pycox)
17/42: dir(pycox.models)
17/43: hazard = model.predict_hazard(x_test)
17/44:
import numpy as np
import matplotlib.pyplot as plt

# For visualizing the survival data: 
from lifelines.plotting import plot_lifetimes

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

#pycox is built on top of PyTorch and torchtuples, where the latter is just a simple way of training neural nets with less boilerplate code.

import torch # For building the networks 
import torchtuples as tt # Some useful functions

#We import the metabric dataset, the LogisticHazard method (paper_link) also known as Nnet-survival, and EvalSurv which simplifies the evaluation procedure at the end.

#You can alternatively replace LogisticHazard with, for example, PMF or DeepHitSingle, which should both work in this notebook.

from pycox.datasets import metabric
# from pycox.models import LogisticHazard
# from pycox.models import PMF
#from pycox.models import DeepHitSingle
from pycox.models import CoxPH #CoxPH = DeepSurv
from pycox.evaluation import EvalSurv

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)
17/45:
num_durations = 10

# labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
# labtrans = DeepHitSingle.label_transform(num_durations)
labtrans = CoxPH.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
17/46:
num_durations = 10

# Label transform (note: different for CoxPH):
# labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
17/47:
Label transform to CoxPH:

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
We need no label transforms

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val
17/48:
#Label transform to CoxPH:

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
We need no label transforms

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val
17/49:
#Label transform to CoxPH:

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

#We need no label transforms
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val
17/50: type(labtrans)
17/51:
#Label transform to CoxPH:

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

#We need no label transforms
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val
20/1:
import numpy as np
import pandas as pd
20/2:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1.csv"
map =  pd.read_csv(filename)
20/3: map.head()
20/4:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1.csv"
map =  pd.read_csv(filename, dtype="str")
20/5: map.head()
20/6:
#Check that keys are unique (abbreveations are the keys)
map.dublicates(subset=["HISLab_abr"])
20/7:
#Check that keys are unique (abbreveations are the keys)
map.duplicates(subset=["HISLab_abr"])
20/8:
#Check that keys are unique (abbreveations are the keys)
map.duplicated(subset=["HISLab_abr"])
20/9:
#Check that keys are unique (abbreveations are the keys)
map.duplicated(subset=["HUSLab_abr"])
20/10:
#Check that keys are unique (abbreveations are the keys)
np.sum(map.duplicated(subset=["HUSLab_abr"]))
20/11:
#Check that keys are unique (abbreveations are the keys)
np.sum(map.duplicated(subset=["HUSLab_abr"]))
map[map.duplicated(subset=["HUSLab_abr"])]
20/12:
#Check that keys are unique (abbreveations are the keys)
if np.sum(map.duplicated(subset=["HUSLab_abr"]))==0:
    print("No duplicates")
map[map.duplicated(subset=["HUSLab_abr"])]
20/13:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isnan()
20/14:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
20/15:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()

unmapped
20/16:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
20/17:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(!unmapped))
20/18:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", not np.sum(unmapped))
20/19:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(not unmapped))
20/20:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(~unmapped))
20/21:
# Remove unmapped rows
mapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(mapped))
print("Unmapped: ", np.sum(~mapped))
map_mapped = map[mapped]
head(map_mapped)
20/22:
# Remove unmapped rows
mapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(mapped))
print("Unmapped: ", np.sum(~mapped))
map_mapped = map[mapped]
map_mapped.head()
20/23:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1.csv"
map =  pd.read_csv(filename, dtype="str")
20/24:
# Remove unmapped rows
mapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(mapped))
print("Unmapped: ", np.sum(~mapped))
map_mapped = map[mapped]
map_mapped.head()
20/25:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(~unmapped))
map_mapped = map[~unmapped]
map_mapped.head()
20/26:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1"
map =  pd.read_csv(filename+".csv", dtype="str")
20/27:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(~unmapped))
map_mapped = map[~unmapped]
map_mapped.head()
20/28:
# Remove unmapped rows
unmapped = map["OMOP_Genomic"].isna()
print("Mapped: ", np.sum(unmapped))
print("Unmapped: ", np.sum(~unmapped))
map_mapped = map[~unmapped]
map_mapped.head()
20/29:
map_mapped.write_csv("filename"+"_clean"+".csv")
import os
os.listdir()
20/30: map_mapped.to_csv("filename"+"_clean"+".csv")
20/31: map_mapped.to_csv("filename"+"_clean"+".csv")
20/32: map_mapped.to_csv(filename+"_clean"+".csv")
21/1:
import numpy as np
import pandas as pd
21/2:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str")
21/3: map.head()
21/4:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)
21/5:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=1)
21/6: map.head()
21/7:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)
21/8: map.head()
21/9:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)

fn = "LABfi.usagi.csv"
LABfi = pd.read_csv(fn, dtype="str")
21/10: LABfi.head()
21/11: LABfi.head(2)
21/12:
# join on HUSLab code
genomictests = (
    map
    .merge(LABfi, how="left", left_on="HUSLab_code", right_on="sourceCode")
)
genomictest.head()
21/13:
# join on HUSLab code
genomictests = (
    map
    .merge(LABfi, how="left", left_on="HUSLab_code", right_on="sourceCode")
)
genomictests.head()
21/14:
# join on HUSLab code
genomictests = (
    map.filter(item=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]], 
        how="left", left_on="HUSLab_code", right_on="sourceCode", suffixes=("", "_LABfi"))
)
genomictests.head()
21/15:
# join on HUSLab code
genomictests = (
    map.filter(item=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]), 
        how="left", left_on="HUSLab_code", right_on="sourceCode", suffixes=("", "_LABfi"))
)
genomictests.head()
21/16:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]), 
        how="left", left_on="HUSLab_code", right_on="sourceCode", suffixes=("", "_LABfi"))
)
genomictests.head()
21/17:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .join(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]), 
        how="left", left_on="HUSLab_code", right_on="sourceCode", suffixes=("", "_LABfi"))
)
genomictests.head()
21/18:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]), 
        how="left", left_on="HUSLab_code", right_on="sourceCode", suffixes=("", "_LABfi"))
)
genomictests.head()
21/19:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"])
        .columns.map(lambda x: str(x) + '_LABfi'), 
        how="left", left_on="HUSLab_code", right_on="sourceCode")
)
genomictests.head()
21/20:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"])
        .columns.map(lambda x: str(x) + '_LABfi'), 
        how="left", left_on="HUSLab_code", right_on="sourceCode_LABfi")
)
genomictests.head()
21/21:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]).columns.map(lambda x: str(x) + '_LABfi'), 
        how="left", left_on="HUSLab_code", right_on="sourceCode_LABfi")
)
genomictests.head()
21/22:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"])
        how="left", left_on="HUSLab_code", right_on="sourceCode_LABfi")
)
genomictests.head()
21/23:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]),
        how="left", left_on="HUSLab_code", right_on="sourceCode_LABfi")
)
genomictests.head()
21/24:
# join on HUSLab code
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["sourceCode", "sourceName", "concept_name"]),
        how="left", left_on="HUSLab_code", right_on="sourceCode")
)
genomictests.head()
21/25:
# join on HUSLab code
LABfi = LABfi.columns.map(lambda x: 'LAPfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LAPfi_sourceCode", "LAPfi_sourceName", "LAPfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LAPfi_sourceCode")
)
genomictests.head()
21/26:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LAPfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LAPfi_sourceCode", "LAPfi_sourceName", "LAPfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LAPfi_sourceCode")
)
genomictests.head()
21/27:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)

fn = "LABfi.usagi.csv"
LABfi = pd.read_csv(fn, dtype="str")
21/28:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LAPfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LAPfi_sourceCode", "LAPfi_sourceName", "LAPfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LAPfi_sourceCode")
)
genomictests.head()
21/29:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LAPfi_sourceCode", "LAPfi_sourceName", "LAPfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LAPfi_sourceCode")
)
genomictests.head()
21/30:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)

fn = "LABfi.usagi.csv"
LABfi = pd.read_csv(fn, dtype="str")
21/31:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LAPfi_sourceCode", "LAPfi_sourceName", "LAPfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LAPfi_sourceCode")
)
genomictests.head()
21/32:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LABfi_sourceCode", "LABfi_sourceName", "LABfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LABfi_sourceCode")
)
genomictests.head()
21/33:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)

fn = "LABfi.usagi.csv"
LABfi = pd.read_csv(fn, dtype="str")
21/34:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LABfi_sourceCode", "LABfi_sourceName", "LABfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LABfi_sourceCode")
)
genomictests.head()
21/35: genomictests
21/36: genomictests.to_csv("Genomictests_pcr_v1.csv")
22/1:
import numpy as np
import pandas as pd
22/2:
filename = "Qpati parser-PCR_mapping_4ETL_v1.1_clean"
map =  pd.read_csv(filename+".csv", dtype="str", index_col=0)

fn = "LABfi.usagi.csv"
LABfi = pd.read_csv(fn, dtype="str")
22/3:
fn = "koodisto.labcodes.120_1387444168447.txt"
thl_koodistopalvelu = pd.read_csv(fn,dtype="str")
22/4:
fn = "koodisto.labcodes.120_1387444168447.txt"
thl_koodistopalvelu = pd.read_csv(fn,dtype="str",  sep=";")
22/5:
fn = "koodisto.labcodes.120_1387444168447.txt"
thl_koodistopalvelu = pd.read_csv(fn,dtype="str",  sep=";", encoding="ansi")
22/6: thl_koodistopalvelu.head()
22/7: thl_koodistopalvelu.head(2)
22/8: thl_koodistopalvelu.head(2)
22/9: LABfi.head(2)
22/10:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
thl_koodistopalvelu.columns = thl_koodistopalvelu.columns.map(lambda x: 'thl_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LABfi_sourceCode", "LABfi_sourceName", "LABfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LABfi_sourceCode"
    )
    .merge(
        thl_koodistopalvelu.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"])
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation"
    )
)
genomictests.head()
22/11:
# join on HUSLab code
#LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
#thl_koodistopalvelu.columns = thl_koodistopalvelu.columns.map(lambda x: 'thl_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LABfi_sourceCode", "LABfi_sourceName", "LABfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LABfi_sourceCode"
    )
    .merge(
        thl_koodistopalvelu.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation"
    )
)
genomictests.head()
22/12:
# join on HUSLab code
LABfi.columns = LABfi.columns.map(lambda x: 'LABfi_' + str(x))
thl_koodistopalvelu.columns = thl_koodistopalvelu.columns.map(lambda x: 'thl_' + str(x))
genomictests = (
    map.filter(items=["HUSLab_abr", "HUSLab_code", "OMOP_Genomics"])
    .merge(
        LABfi.filter(items=["LABfi_sourceCode", "LABfi_sourceName", "LABfi_concept_name"]),
        how="left", left_on="HUSLab_code", right_on="LABfi_sourceCode"
    )
    .merge(
        thl_koodistopalvelu.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation"
    )
)
genomictests.head()
22/13: genomictests
22/14: genomictests.to_csv("Genomictests_pcr_v1.1.csv")
22/15:
# Look at test missing in thl:
genomictests.query("thl_CodeId=NaN")
22/16:
# Look at test missing in thl:
genomictests.query("thl_CodeId==NaN")
22/17:
# Look at test missing in thl:
genomictests.query('thl_CodeId=="NaN"')
22/18:
# Look at test missing in thl:
genomictests.query('thl_CodeId.isnan()')
22/19:
# Look at test missing in thl:
genomictests.query('thl_CodeId.isna()')
22/20: import difflib
22/21:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltest)
    print(row["HUSLab_abr"], closestmatch)
22/22:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows:
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltest)
    print(row["HUSLab_abr"], closestmatch)
22/23:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltest)
    print(row["HUSLab_abr"], closestmatch)
22/24:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], closestmatch)
22/25:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
22/26:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    df[idx, "thl"] = closestmatch[0]
22/27:
import difflib
df = genomictests.query('thl_CodeId.isna()')
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    df.loc[idx, "thl"] = closestmatch[0]
22/28:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = df["HUSLab_abr"]
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    df2.loc[idx, "thl"] = closestmatch[0]
22/29:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = df["HUSLab_abr"]
df2["thl"] = "NaN"
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    df2.loc[idx, "thl"] = closestmatch[0]
22/30: df2
22/31:
df2
df2["thl"] = "NaN"
df2
22/32:
df2
df2.thl = "NaN"
df2
22/33:
df2
df2["thl"] = ""
df2
22/34:
df2 = df["HUSLab_abr"].copy()
df2["thl"] = ""
22/35:
df2 = df["HUSLab_abr"].copy()
df2["thl"] = ""
df
22/36:
df2 = df["HUSLab_abr"].copy()
df2["thl"] = ""
df2
22/37:
df2 = df["HUSLab_abr"].copy()
df2["thl"] = ""
df["C"] = np.nan
df2
22/38:
df2 = df["HUSLab_abr"].copy()
df2["thl"] = ""
df2["C"] = np.nan
df2
22/39:
df2 = df["HUSLab_abr]
df2["thl"] = ""
df2["C"] = np.nan
df2
22/40:
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
df2["C"] = np.nan
df2
22/41:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    df2.loc[idx, "thl"] = closestmatch[0]
22/42: df2
22/43:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    try:
        df2.loc[idx, "thl"] = closestmatch[0]
22/44:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    try:
        df2.loc[idx, "thl"] = closestmatch[0]
    execpt:
22/45:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    try:
        df2.loc[idx, "thl"] = closestmatch[0]
22/46:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else:
        print(f"No match for {row["HUSLab_abr"]})
    end
22/47:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows(): 
    closestmatch = difflib.get_close_matches(row["HUSLab_abr"], thltests)
    print(row["HUSLab_abr"], *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else:
        print(f"No match for {row\[\"HUSLab_abr\"\]})
    end
22/48:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else:
        print(f"No match for {abr})
    end
22/49:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else
        print(f"No match for {abr})
    end
22/50:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else:
        print(f"No match for {abr}")
    end
22/51:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else
        print(f"No match for {abr}")
    end
22/52:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch>0): 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
    end
22/53:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
    end
22/54:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
22/55:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    #print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
df2
22/56:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    #print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch
    else: 
        print(f"No match for {abr}")
df2
22/57:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    #print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = [closestmatch]
    else: 
        print(f"No match for {abr}")
df2
22/58:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests)
    #print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
df2
22/59:
import difflib
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df2["thl"] = ""
thltests = thl_koodistopalvelu["thl_Abbreviation"]
for idx, row in df.iterrows():
    abr = row["HUSLab_abr"]
    closestmatch = difflib.get_close_matches(abr, thltests, n=1)
    #print(abr, *closestmatch)
    if len(closestmatch)>0: 
        df2.loc[idx, "thl"] = closestmatch[0]
    else: 
        print(f"No match for {abr}")
df2
22/60:
df = genomictests.query('thl_CodeId.isna()')["thl_CodeId"]
df2 = pd.DataFrame(df["HUSLab_abr"])
df
22/61:
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
df["thl_CodeId"]
22/62:
df = genomictests.query('thl_CodeId.isna()')
df2 = pd.DataFrame(df["HUSLab_abr"])
thl
22/63:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_CodeId=lambda x: x["thl_CodeId"].str.replace(" ",""))
    
df2 = pd.DataFrame(df["HUSLab_abr"])
thl
22/64:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_CodeId=lambda x: x["thl_CodeId"].str.replace(" ",""))
)
df2 = pd.DataFrame(df["HUSLab_abr"])
thl
22/65:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_CodeId=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df2 = pd.DataFrame(df["HUSLab_abr"])
thl
22/66:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df2 = pd.DataFrame(df["HUSLab_abr"])
thl
22/67:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df2
22/68:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df
22/69:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df = (
    df
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
22/70:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df = (
    df
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
df
22/71:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ",""))
)
df = (
    df.filter(["HUSLab_abr", "HUSLab_code"])
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
df
22/72:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ","").upper())
)
df = (
    df.filter(["HUSLab_abr", "HUSLab_code"])
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
df
22/73:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ","").str.upper())
)
df = (
    df.filter(["HUSLab_abr", "HUSLab_code"])
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
df
22/74:
df = genomictests.query('thl_CodeId.isna()')
thl = (
    thl_koodistopalvelu.copy()
    .assign(thl_Abbreviation=lambda x: x["thl_Abbreviation"].str.replace(" ","").str.upper())
)
df = (
    df.filter(["HUSLab_abr", "HUSLab_code"])
    .assign(HUSLab_abr=lambda x: x["HUSLab_abr"].str.upper())
    .merge(thl.filter(["thl_CodeId", "thl_Abbreviation", "thl_ShortName", "thl_LongName"]),
        how="left", left_on="HUSLab_abr", right_on="thl_Abbreviation")
)
df
23/1:
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

%matplotlib inline
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint

import sys
sys.path.append("../FinRL")

import itertools
23/2:
## install required packages
!pip install swig
!pip install wrds
!pip install pyportfolioopt
## install finrl library
!pip install -q condacolab
import condacolab
condacolab.install()
!apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
23/3:
## install required packages
!pip install swig
!pip install wrds
!pip install pyportfolioopt
## install finrl library
!pip install -q condacolab
#import condacolab
#condacolab.install()
!apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
24/1:
## install finrl library
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
24/2:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
24/3:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
25/1:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
25/2:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
27/1:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
27/2: !pip install yfinance
27/3: import yfinance as yf
27/4: import yfinance
27/5: !pip install yfinance
27/6: import yfinance
27/7: pip install yfinance --upgrade --no-cache-dir
27/8: pip install pathlib
27/9: pip install ruamel-yaml
28/1: pip install yfinance --upgrade --no-cache-dir
28/2:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
28/3:
## install finrl library
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
29/1:
import pandas as pd
import yfinance as yf
29/2: dir(yf)
29/3:
obj = yf.Ticker('goog')
obj
29/4: dir(obj)
29/5:
goog = yf.Ticker('goog')
data = goog.history()
data.head()
29/6:
data = goog.history(interval='1m', start='2022-01-03', end='2022-01-10')
data.head()
29/7:
data = goog.history(interval='1m', start='2023-10-01', end='2023-01-08')
data.head()
29/8:
data = goog.history(interval='1m', start='2023-10-01', end='2023-10-08')
data.head()
29/9:
data = yf.download(['GOOG','META'], period='1mo')
data.head()
29/10:
dhr = yf.Ticker('DHR')
info = dhr.info
info.keys()
29/11: info['sector']
29/12: dhr.earnings
29/13: dhr.get_financials()
29/14:
pnl = dhr.financials
bs = dhr.balancesheet
cf = dhr.cashflow
fs = pd.concat([pnl,bs,cf])
fs
29/15: fs.T
29/16:
fb = yf.Ticker('fb')
meta = yf.Ticker('meta')
fb.get_cashflow()
29/17: meta.get_cashflow()
30/1:
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools
30/2:
## install finrl library
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
30/3: !conda install -c https://conda.anaconda.org/kne pybox2d
30/4: !pip install box2d-py
31/1:
# Import required libraries
import dash
import dash_core_components as dcc
import dash_html_components as html

# Initialize the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "NYC"},
                ],
                "layout": {"title": "Bar Chart Example"},
            },
        ),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
31/2: !conda install dash dash-core-components dash-html-components
32/1:
# Import required libraries
import dash
import dash_core_components as dcc
import dash_html_components as html

# Initialize the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "NYC"},
                ],
                "layout": {"title": "Bar Chart Example"},
            },
        ),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/2:
# Import required libraries
import dash
from dash import dcc
from dash import html

# Initialize the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "NYC"},
                ],
                "layout": {"title": "Bar Chart Example"},
            },
        ),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/3:
# Import required libraries
import dash
from dash import dcc
from dash import html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        dcc.Graph(id='graph')
    ]
)

# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return {
        'data': [
            {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
        ],
        'layout': {'title': "Bar Chart Example"},
    }

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/4:
# Import required libraries
import dash
from dash import dcc, html

# Initialize the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {"x": [1, 2, 3], "y": [2, 4, 5], "type": "bar", "name": "NYC"},
                ],
                "layout": {"title": "Bar Chart Example"},
            },
        ),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/5:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(id='bar-graph'),
                dcc.Graph(id='scatter-graph')
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {'title': "Bar Chart Example"},
        },
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/6:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                ),
                dcc.Graph(id='scatter-graph')
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {'title': "Bar Chart Example"},
        },
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/7:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    id='scatter-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {'title': "Bar Chart Example"},
        },
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/8:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    id='scatter-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {'title': "Bar Chart Example"},
        },
        {
            'data': [
                {'x': df['A'], 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/9:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    id='scatter-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {
                'title': "Bar Chart Example",
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': value}
            },
        },
        {
            'data': [
                {'x': df['A'], 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/10:
# Import required libraries
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the app
app = dash.Dash(__name__)

# Create a DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 1, 5, 4, 3],
    'C': [5, 3, 1, 2, 4]
})

# Define the app
app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(children="Dash: A web application framework for Python."),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': i, 'value': i} for i in df.columns
            ],
            value='A'
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='bar-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    id='scatter-graph',
                    style={'width': '50%', 'display': 'inline-block'}
                    )
            ],
            style={'display': 'flex'}
        )
    ]
)

# Define callback to update graphs
@app.callback(
    [Output('bar-graph', 'figure'),
     Output('scatter-graph', 'figure')],
    [Input('dropdown', 'value')]
)
def update_figure(value):
    return [
        {
            'data': [
                {'x': df.index, 'y': df[value], 'type': 'bar', 'name': value},
            ],
            'layout': {
                'title': "Bar Chart Example",
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': value}
            },
        },
        {
            'data': [
                {'x': df['A'], 'y': df[value], 'type': 'scatter', 'mode': 'markers', 'name': value},
            ],
            'layout': {'title': "Scatter Plot Example"},
        }
    ]

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
32/11:
import sys
print(sys.version)
33/1:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'concept_id', 'concept_name']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['concept_id_df1'] == merged_df['concept_id_df2']) &
        (merged_df['concept_name_df1'] == merged_df['concept_name_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['concept_id_df1'] != merged_df['concept_id_df2']) |
        (merged_df['concept_name_df1'] != merged_df['concept_name_df2'])
    ]
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/2:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'concept_id', 'concept_name']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['concept_id_df1'] == merged_df['concept_id_df2']) &
        (merged_df['concept_name_df1'] == merged_df['concept_name_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['concept_id_df1'] != merged_df['concept_id_df2']) |
        (merged_df['concept_name_df1'] != merged_df['concept_name_df2'])
    ]
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/3: df1 = pd.read_csv("ICD10fi.usagi.csv")
33/4: df1
33/5:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")
33/6: same, different = compare_dataframes(df1, df2)
33/7:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")

head(df1. 5)
33/8:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")

head(df1, 5)
33/9:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")

df1.head(2)
33/10:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['concept_id_df1'] == merged_df['concept_id_df2']) &
        (merged_df['concept_name_df1'] == merged_df['concept_name_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['concept_id_df1'] != merged_df['concept_id_df2']) |
        (merged_df['concept_name_df1'] != merged_df['concept_name_df2'])
    ]
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/11:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")

df1.head(2)
33/12: same, different = compare_dataframes(df1, df2)
33/13:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['concept_id_df1'] == merged_df['concept_id_df2']) &
        (merged_df['concept_name_df1'] == merged_df['concept_name_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['concept_id_df1'] != merged_df['concept_id_df2']) |
        (merged_df['concept_name_df1'] != merged_df['concept_name_df2'])
    ]
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/14: same, different = compare_dataframes(df1, df2)
33/15:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['conceptId_df1'] == merged_df['conceptId_df2']) &
        (merged_df['conceptName_df1'] == merged_df['conceptName_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['conceptId_df1'] != merged_df['conceptId_df2']) |
        (merged_df['conceptName_df1'] != merged_df['conceptName_df2'])
    ]
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/16: same, different = compare_dataframes(df1, df2)
33/17: same
33/18: different
33/19:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['conceptId_df1'] == merged_df['conceptId_df2']) &
        (merged_df['conceptName_df1'] == merged_df['conceptName_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['conceptId_df1'] != merged_df['conceptId_df2']) |
        (merged_df['conceptName_df1'] != merged_df['conceptName_df2'])
    ]
    
    # Add highlight columns to indicate differences
    different_rows['sourceName_diff'] = different_rows['sourceName_df1'] != different_rows['sourceName_df2']
    different_rows['ADD_INFO:sourceName_fi_diff'] = different_rows['ADD_INFO:sourceName_fi_df1'] != different_rows['ADD_INFO:sourceName_fi_df2']
    different_rows['conceptId_diff'] = different_rows['conceptId_df1'] != different_rows['conceptId_df2']
    different_rows['conceptName_diff'] = different_rows['conceptName_df1'] != different_rows['conceptName_df2']
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/20: same, different = compare_dataframes(df1, df2)
33/21:
import pandas as pd

def compare_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode'
    merged_df = pd.merge(df1_selected, df2_selected, on='sourceCode', suffixes=('_df1', '_df2'))
    
    # Find rows where all columns are the same
    same_rows = merged_df[
        (merged_df['sourceName_df1'] == merged_df['sourceName_df2']) &
        (merged_df['ADD_INFO:sourceName_fi_df1'] == merged_df['ADD_INFO:sourceName_fi_df2']) &
        (merged_df['conceptId_df1'] == merged_df['conceptId_df2']) &
        (merged_df['conceptName_df1'] == merged_df['conceptName_df2'])
    ]
    
    # Find rows where any column values differ
    different_rows = merged_df[
        (merged_df['sourceName_df1'] != merged_df['sourceName_df2']) |
        (merged_df['ADD_INFO:sourceName_fi_df1'] != merged_df['ADD_INFO:sourceName_fi_df2']) |
        (merged_df['conceptId_df1'] != merged_df['conceptId_df2']) |
        (merged_df['conceptName_df1'] != merged_df['conceptName_df2'])
    ]
    
    # Add highlight columns to indicate differences
    different_rows['sourceName_diff'] = different_rows['sourceName_df1'] != different_rows['sourceName_df2']
    different_rows['ADD_INFO:sourceName_fi_diff'] = different_rows['ADD_INFO:sourceName_fi_df1'] != different_rows['ADD_INFO:sourceName_fi_df2']
    different_rows['conceptId_diff'] = different_rows['conceptId_df1'] != different_rows['conceptId_df2']
    different_rows['conceptName_diff'] = different_rows['conceptName_df1'] != different_rows['conceptName_df2']
    
    return same_rows, different_rows

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes(df1, df2)
33/22:
df1 = pd.read_csv("ICD10fi.usagi.csv")
df2 = pd.read_csv("ICD10fi.fixedEF.usagi.csv")

df1.head(2)
33/23: same, different = compare_dataframes(df1, df2)
33/24: different
33/25:
import pandas as pd

def compare_dataframes_mult(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Group by 'sourceCode'
    df1_grouped = df1_selected.groupby('sourceCode')
    df2_grouped = df2_selected.groupby('sourceCode')
    
    same_rows_list = []
    different_rows_list = []
    
    # Iterate over each group in df1
    for source_code, group1 in df1_grouped:
        if source_code in df2_grouped.groups:
            group2 = df2_grouped.get_group(source_code)
            
            # Merge the groups on 'concept_id'
            merged_group = pd.merge(group1, group2, on='conceptId', suffixes=('_df1', '_df2'))
            
            # Find rows where all columns are the same
            same_rows = merged_group[
                (merged_group['sourceName_df1'] == merged_group['sourceName_df2']) &
                (merged_group['ADD_INFO:sourceName_fi_df1'] == merged_group['ADD_INFO:sourceName_fi_df2']) &
                (merged_group['conceptName_df1'] == merged_group['conceptName_df2'])
            ]
            
            # Find rows where any column values differ
            different_rows = merged_group[
                (merged_group['sourceName_df1'] != merged_group['sourceName_df2']) |
                (merged_group['ADD_INFO:sourceName_fi_df1'] != merged_group['ADD_INFO:sourceName_fi_df2']) |
                (merged_group['conceptName_df1'] != merged_group['conceptName_df2'])
            ]
            
            same_rows_list.append(same_rows)
            different_rows_list.append(different_rows)
    
    # Concatenate the results
    same_rows_df = pd.concat(same_rows_list)
    different_rows_df = pd.concat(different_rows_list)
    
    return same_rows_df, different_rows_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# same, different = compare_dataframes_mult(df1, df2)
33/26: same, different = compare_dataframes_mult(df1, df2)
33/27: same
33/28: different
33/29:
def merge_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'concept_id', 'concept_name']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode' and 'concept_id'
    merged_df = pd.merge(df1_selected, df2_selected, on=['sourceCode', 'concept_id'], suffixes=('_df1', '_df2'))
    
    return merged_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# merged = merge_dataframes(df1, df2
33/30: merged = merge_dataframes(df1, df2)
33/31:
def merge_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode' and 'concept_id'
    merged_df = pd.merge(df1_selected, df2_selected, on=['sourceCode', 'conceptId'], suffixes=('_df1', '_df2'))
    
    return merged_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# merged = merge_dataframes(df1, df2)
33/32: merged = merge_dataframes(df1, df2)
33/33: merged
33/34: df1
33/35:
def merge_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode' and 'concept_id'
    merged_df = pd.merge(df1_selected, df2_selected, on=['sourceCode', 'conceptId'], suffixes=('_df1', '_df2'), how='outer')
    
    return merged_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# merged = merge_dataframes(df1, df2)
33/36: merged = merge_dataframes(df1, df2)
33/37: df1
33/38:
def merge_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode' and 'concept_id'
    merged_df = pd.merge(df1_selected, df2_selected, on=['sourceCode', 'conceptId'], suffixes=('_df1', '_df2'), how='outer')
    
    return merged_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# merged = merge_dataframes(df1, df2)
33/39: merged = merge_dataframes(df1, df2)
33/40: df1
33/41: merged
33/42:
# Example df1
df1 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'C'],
    'sourceName': ['Source A', 'Source B', 'Source C'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde C'],
    'concept_id': [1, 2, None],
    'concept_name': ['Concept 1', 'Concept 2', 'Concept C']
})

# Example df2
df2 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'D'],
    'sourceName': ['Source A', 'Source B', 'Source D'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde D'],
    'concept_id': [1, None, 4],
    'concept_name': ['Concept 1', 'Concept B', 'Concept 4']
})
33/43:
# Example df1
df1 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'C'],
    'sourceName': ['Source A', 'Source B', 'Source C'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde C'],
    'concept_id': [1, 2, None],
    'concept_name': ['Concept 1', 'Concept 2', 'Concept C']
})

# Example df2
df2 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'D'],
    'sourceName': ['Source A', 'Source B', 'Source D'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde D'],
    'concept_id': [1, None, 4],
    'concept_name': ['Concept 1', 'Concept B', 'Concept 4']
})

merged = merge_dataframes_keep_all(df1, df2)
print(merged)
33/44:
# Example df1
df1 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'C'],
    'sourceName': ['Source A', 'Source B', 'Source C'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde C'],
    'concept_id': [1, 2, None],
    'concept_name': ['Concept 1', 'Concept 2', 'Concept C']
})

# Example df2
df2 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'D'],
    'sourceName': ['Source A', 'Source B', 'Source D'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde D'],
    'concept_id': [1, None, 4],
    'concept_name': ['Concept 1', 'Concept B', 'Concept 4']
})

merged = merge_dataframes(df1, df2)
print(merged)
33/45:
def merge_dataframes(df1, df2):
    # Select the relevant columns
    columns = ['sourceCode', 'sourceName', 'ADD_INFO:sourceName_fi', 'conceptId', 'conceptName']
    df1_selected = df1[columns]
    df2_selected = df2[columns]
    
    # Merge the dataframes on 'sourceCode' and 'concept_id'
    merged_df = pd.merge(df1_selected, df2_selected, on=['sourceCode', 'conceptId'], suffixes=('_df1', '_df2'), how='outer')
    
    return merged_df

# Example usage:
# df1 = pd.DataFrame({...})
# df2 = pd.DataFrame({...})
# merged = merge_dataframes(df1, df2)
33/46:
# Example df1
df1 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'C'],
    'sourceName': ['Source A', 'Source B', 'Source C'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde C'],
    'conceptId': [1, 2, None],
    'conceptName': ['Concept 1', 'Concept 2', 'Concept C']
})

# Example df2
df2 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'D'],
    'sourceName': ['Source A', 'Source B', 'Source D'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde D'],
    'conceptId': [1, None, 4],
    'conceptName': ['Concept 1', 'Concept B', 'Concept 4']
})

merged = merge_dataframes(df1, df2)
print(merged)
33/47:
# Example df1
df1 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'C'],
    'sourceName': ['Source A', 'Source B', 'Source C'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde C'],
    'conceptId': [1, 2, None],
    'conceptName': ['Concept 1', 'Concept 2', 'Concept C']
})

# Example df2
df2 = pd.DataFrame({
    'sourceCode': ['A', 'B', 'D'],
    'sourceName': ['Source A', 'Source B', 'Source D'],
    'ADD_INFO:sourceName_fi': ['Lähde A', 'Lähde B', 'Lähde D'],
    'conceptId': [1, None, 4],
    'conceptName': ['Concept 1', 'Concept B', 'Concept 4']
})

print(df1)
print(df2)

merged = merge_dataframes(df1, df2)
print(merged)
36/1: !ls
36/2: !dir
36/3: import pandas as pd
36/4: concept = pd.read_csv("CONCEPT.csv")
36/5:
concept = pd.read_csv("CONCEPT.csv", engine = 'python', error_bad_lines=False)
concept.head()
36/6:
concept = pd.read_csv("CONCEPT.csv", engine = 'python', 
                      error_bad_lines=False, nrows=100)
concept.head()
36/7:
concept = pd.read_csv("CONCEPT.csv", engine = 'python', sep='\t',
                      error_bad_lines=False, nrows=100)
concept.head()
36/8:
concept = pd.read_csv("CONCEPT.csv", engine = 'python' 
                      ,sep='\t'
                      ,error_bad_lines=False 
                      ,nrows=100
                     )
concept.head()
36/9:
concept = pd.read_csv("CONCEPT.csv", engine = 'python' 
                      ,sep='\t'
                      ,error_bad_lines=False 
                      #,nrows=100
                     )
concept.head()
36/10: concept.query("LENGTH(domain_id)>20")
36/11: concept.query("domain_id.str.len() > 20", engine='python')
38/1:
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1('Patient Timeline Viewer'),
    dcc.RadioItems(
        id='data-selector',
        options=[
            {'label': 'Top 10', 'value': 'top10'},
            {'label': 'Custom', 'value': 'custom'},
        ],
        value='top10'
    ),
    html.Div(id='output-selected-data', children=html.P("Nothing here...")),
    html.Button('Click Me', id='button'),
    html.Div(id='button-output')
])

# Callback to handle the radio selection
@app.callback(
    Output('output-selected-data', 'children'),
    Input('data-selector', 'value')
)
def update_output(selection):
    global meas_list, proc_list, drugs_list
    print(f'Clicked {selection}')
    if selection == 'top10':
        #meas_list, proc_list, drugs_list = get_top10data()
        print(f'top10 selected')
    return html.P(f'Selected: {selection}')

# Callback to handle the button click
@app.callback(
    Output('button-output', 'children'),
    Input('button', 'n_clicks')
)
def update_button_output(n_clicks):
    if n_clicks:
        print('OK')
        return html.P('OK')
    return html.P('Button not clicked yet')

if __name__ == '__main__':
    app.run_server(debug=True)
39/1: !pip install -e .
39/2: !pip install -e .
39/3: !pip install -e .
39/4: !pip install .
42/1:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
43/1:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
43/2:
#Load tidy data
print("Loading tidy data")
df_xy = pd.read_csv("data/df_xy_synth_v1.csv")
# IMPUTE nan: -1
df_xy = df_xy.fillna(-1)

# Define function to convert df into 3-D numpy array
def convert_to_3d_df(df):

    # Convert column names to tuples, assuming this "('feature', timepoint)"
    columns = [eval(col) for col in df.columns]
    df.columns = columns
    
    # Extract unique features and timepoints
    features = sorted(list(set([col[0] for col in columns])))
    timepoints = sorted(list(set([col[1] for col in columns])))
    
    # Initialize a 3D numpy array
    n_rows = df.shape[0]
    n_features = len(features)
    n_timepoints = len(timepoints)
    data_3d = np.empty((n_rows, n_features, n_timepoints))
    data_3d.fill(np.nan)
    
    # Map feature names and timepoints to indices
    feature_indices = {feature: i for i, feature in enumerate(features)}
    timepoint_indices = {timepoint: i for i, timepoint in enumerate(timepoints)}
    
    # Fill the 3D array with data from the DataFrame
    for col in columns:
        feature, timepoint = col
        feature_idx = feature_indices[feature]
        timepoint_idx = timepoint_indices[timepoint]
        data_3d[:, feature_idx, timepoint_idx] = df[col]

    # Create a MultiIndex for the columns of the 3D DataFrame
    columns = pd.MultiIndex.from_product([features, timepoints], names=["Feature", "Timepoint"])
    
    # Create the 3D DataFrame
    df_multiindex = pd.DataFrame(data_3d.reshape(n_rows, -1), columns=columns)
    
    return df_multiindex, data_3d

# Convert AML data to multiindex df
df_x, data_3d = convert_to_3d_df(df_xy.iloc[:,3:].fillna(-1))
df_y = df_xy.iloc[:,:3]
display(df_x)
display(df_y)
43/3:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
43/4: !pip install -e .
43/5:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
45/1:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
45/2:
class SurvivalHead(nn.Module):
    """
    Neural Cox proportional hazards model head.
    Outputs hazard ratios instead of binary classification.
    """
    def __init__(self, d_model, n_features, dropout=0.1):
        super().__init__()
        self.risk_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # log hazard ratio
        )

    def forward(self, x):
        # Risk score (log hazard ratio)
        return torch.exp(self.risk_score(x))  # hazard ratio

    
def cox_loss(risk_scores, survival_times, events):
    """
    Negative log partial likelihood for Cox model
    
    Args:
        risk_scores: predicted hazard ratios
        survival_times: time to event/censoring
        events: 1 if event occurred, 0 if censored
    """
    # Sort by survival time
    _, indices = torch.sort(survival_times, descending=True)
    risk_scores = risk_scores[indices]
    events = events[indices]
    
    # Calculate log partial likelihood
    log_risk = torch.log(torch.cumsum(torch.exp(risk_scores), 0))
    likelihood = risk_scores - log_risk
    
    # Mask for events only
    return -torch.mean(likelihood * events)
45/3:
class SurvivalDataset(Dataset):
    def __init__(self, features, survival_times, events):
        self.features = features
        self.times = survival_times    # Time to event/censoring
        self.events = events           # Event indicator (1=death, 0=censored)
45/4:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
print("Libraries loaded")
45/5:
class SurvivalDataset(Dataset):
    def __init__(self, features, survival_times, events):
        self.features = features
        self.times = survival_times    # Time to event/censoring
        self.events = events           # Event indicator (1=death, 0=censored)
45/6:
#Load tidy data
print("Loading tidy data")
df_xy = pd.read_csv("data/df_xy_synth_v1.csv")
# IMPUTE nan: -1
df_xy = df_xy.fillna(-1)

# Define function to convert df into 3-D numpy array
def convert_to_3d_df(df):

    # Convert column names to tuples, assuming this "('feature', timepoint)"
    columns = [eval(col) for col in df.columns]
    df.columns = columns
    
    # Extract unique features and timepoints
    features = sorted(list(set([col[0] for col in columns])))
    timepoints = sorted(list(set([col[1] for col in columns])))
    
    # Initialize a 3D numpy array
    n_rows = df.shape[0]
    n_features = len(features)
    n_timepoints = len(timepoints)
    data_3d = np.empty((n_rows, n_features, n_timepoints))
    data_3d.fill(np.nan)
    
    # Map feature names and timepoints to indices
    feature_indices = {feature: i for i, feature in enumerate(features)}
    timepoint_indices = {timepoint: i for i, timepoint in enumerate(timepoints)}
    
    # Fill the 3D array with data from the DataFrame
    for col in columns:
        feature, timepoint = col
        feature_idx = feature_indices[feature]
        timepoint_idx = timepoint_indices[timepoint]
        data_3d[:, feature_idx, timepoint_idx] = df[col]

    # Create a MultiIndex for the columns of the 3D DataFrame
    columns = pd.MultiIndex.from_product([features, timepoints], names=["Feature", "Timepoint"])
    
    # Create the 3D DataFrame
    df_multiindex = pd.DataFrame(data_3d.reshape(n_rows, -1), columns=columns)
    
    return df_multiindex, data_3d

# Convert AML data to multiindex df
df_x, data_3d = convert_to_3d_df(df_xy.iloc[:,3:].fillna(-1))
df_y = df_xy.iloc[:,:3]
display(df_x)
display(df_y)
45/7:
class LinearCoxPH(nn.Module):
    """
    Classical Cox PH with linear predictor: h(t|x) = h₀(t)exp(βx)
    """
    def __init__(self, n_features):
        super().__init__()
        self.beta = nn.Linear(n_features, 1, bias=False)  # β coefficients
        
    def forward(self, x):
        return torch.exp(self.beta(x))  # exp(βx)
45/8:
class CoxPHModel(nn.Module):
    """
    Classical Cox Proportional Hazards model implemented in PyTorch.
    Learns a linear combination of features to predict hazard ratios.
    """
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Linear hazard prediction
        self.hazard_ratio = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        """
        Compute hazard ratios for each sample.
        
        Args:
            x: Input features [batch_size, n_features]
            
        Returns:
            hazard_ratios: Predicted hazard ratios [batch_size, 1]
        """
        return torch.exp(self.hazard_ratio(x))  # exp(β * x)
45/9:
def cox_loss(hazard_ratios, durations, events):
    """
    Negative log partial likelihood for Cox model.
    
    Args:
        hazard_ratios: Predicted hazard ratios [batch_size, 1]
        durations: Time to event/censoring [batch_size]
        events: Event indicators (1=event, 0=censored) [batch_size]
    
    Returns:
        loss: Negative log partial likelihood
    """
    # Sort all arrays by duration in descending order
    sorted_idx = torch.argsort(durations, descending=True)
    hazard_ratios = hazard_ratios[sorted_idx]
    events = events[sorted_idx]
    
    # Calculate log risk (cumulative hazard)
    log_risk = torch.logcumsumexp(hazard_ratios.flatten(), dim=0)
    
    # Select events that contribute to likelihood
    event_indices = (events == 1).nonzero().flatten()
    
    if len(event_indices) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    # Calculate negative log likelihood
    partial_likelihood = hazard_ratios[event_indices].flatten() - log_risk[event_indices]
    neg_likelihood = -torch.mean(partial_likelihood)
    
    return neg_likelihood
45/10:
def train_cox_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """
    Train Cox proportional hazards model.
    
    Args:
        model: CoxPHModel instance
        train_loader: DataLoader with (features, durations, events)
        val_loader: DataLoader with (features, durations, events)
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for x, durations, events in train_loader:
            x = x.float().to(device)
            durations = durations.float().to(device)
            events = events.float().to(device)
            
            optimizer.zero_grad()
            hazard_ratios = model(x)
            loss = cox_loss(hazard_ratios, durations, events)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, durations, events in val_loader:
                x = x.float().to(device)
                durations = durations.float().to(device)
                events = events.float().to(device)
                
                hazard_ratios = model(x)
                val_loss += cox_loss(hazard_ratios, durations, events).item()
        
        # Log metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cox_model.pt')
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return history
45/11:
from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    """
    Dataset for survival analysis with Cox PH model.
    """
    def __init__(self, features, durations, events):
        self.features = torch.FloatTensor(features)
        self.durations = torch.FloatTensor(durations)
        self.events = torch.FloatTensor(events)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return (self.features[idx], 
                self.durations[idx], 
                self.events[idx])
45/12:
# Test/Example data

# from sksurv.datasets import load_veterans_lung_cancer

# data_x, data_y = load_veterans_lung_cancer()
# data_y

from lifelines.datasets import load_regression_dataset
regression_dataset = load_regression_dataset() # a Pandas DataFrame
regression_dataset.head()
45/13:
# Example usage
from torch.utils.data import DataLoader

# Split data
df_train = regression_dataset.sample(frac=0.8)
df_val = regression_dataset.drop(df_train.index)

# Create datasets
train_dataset = SurvivalDataset(
    features=df_train.drop(columns=['T', 'E']).values,
    durations=df_train['T'].values,
    events=df_train['E'].values
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True
)

val_dataset = SurvivalDataset(
    features=df_val.drop(columns=['T', 'E']).values,
    durations=df_val['T'].values,
    events=df_val['E'].values
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False
)

# # Define model
# n_features = 3
# model = CoxPHModel(n_features=n_features)

# # Train model
# history = train_cox_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     epochs=100
# )

model2 = LinearCoxPH(n_features=n_features)

# Train model
history = train_cox_model(
    model=model2,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
45/14:
# Example usage
from torch.utils.data import DataLoader

# Split data
df_train = regression_dataset.sample(frac=0.8)
df_val = regression_dataset.drop(df_train.index)

# Create datasets
train_dataset = SurvivalDataset(
    features=df_train.drop(columns=['T', 'E']).values,
    durations=df_train['T'].values,
    events=df_train['E'].values
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True
)

val_dataset = SurvivalDataset(
    features=df_val.drop(columns=['T', 'E']).values,
    durations=df_val['T'].values,
    events=df_val['E'].values
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False
)

# # Define model
n_features = 3
# model = CoxPHModel(n_features=n_features)

# # Train model
# history = train_cox_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     epochs=100
# )

model2 = LinearCoxPH(n_features=n_features)

# Train model
history = train_cox_model(
    model=model2,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
45/15:
def get_risk_groups(model, data_loader, device='cpu'):
    """Get risk scores for all patients and split into high/low risk groups"""
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for x, durations, events in data_loader:
            x = x.float().to(device)
            hazard_ratios = model(x)
            # Ensure 1D arrays
            all_risks.append(hazard_ratios.cpu().numpy().flatten())
            all_times.append(durations.numpy().flatten())
            all_events.append(events.numpy().flatten())
    
    # Concatenate all predictions
    risk_scores = np.concatenate(all_risks)
    times = np.concatenate(all_times)
    events = np.concatenate(all_events)
    
    # Split into high/low risk groups using median
    median_risk = np.median(risk_scores)
    high_risk = risk_scores >= median_risk
    
    return risk_scores, times, events, high_risk

from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def plot_risk_stratification(times, events, high_risk, title="Risk Stratification"):
    """Plot Kaplan-Meier curves for high and low risk groups"""
    
    # Initialize KM estimator
    kmf = KaplanMeierFitter()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot high risk group
    mask = high_risk.astype(bool)  # Ensure boolean mask
    kmf.fit(times[mask], events[mask], label='High Risk')
    kmf.plot()
    
    # Plot low risk group
    mask = ~high_risk.astype(bool)  # Ensure boolean mask
    kmf.fit(times[mask], events[mask], label='Low Risk')
    kmf.plot()
    
    # Customize plot
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    
    # Add log-rank test
    from lifelines.statistics import logrank_test
    log_rank = logrank_test(times[high_risk], times[~high_risk],
                           events[high_risk], events[~high_risk])
    plt.text(0.05, 0.05, f'Log-rank p-value: {log_rank.p_value:.3e}',
             transform=plt.gca().transAxes)
    
    return plt.gcf()
45/16: !pip install pycox
45/17: from pycox.models import DeepHitSingle
45/18:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
45/19:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
45/20:
from pycox.models import DeepHitSingle
import torchtupes as tt
45/21:
from pycox.models import DeepHitSingle
import torchtuples as tt
45/22:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=labtrans.cuts)
45/23:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])
45/24:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])
model
45/25:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])

print(model)
45/26:
# Simple test of pycox
in_features = 5 #x_train.shape[1]
num_nodes = [8, 8]
out_features = 10 #labtrans.out_features
batch_norm = True
dropout = 0.2

# Define ANN
print("Creating ANN")
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])

print(model)
model.net
45/27:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)
45/28:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

# Initialize model
model = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=128,
    n_heads=16,
    dropout=0.1
)
45/29:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
model = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)
45/30:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
model = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)
model
45/31:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
net = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)
model
45/32:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
net = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)
net
45/33:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
net = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)
net

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])

print(model)
model.net
45/34:
# now try a ukko net
import importlib
import ukko.core
import ukko.data
importlib.reload(ukko.core)
importlib.reload(ukko.data)

n_features = 5
sequence_length = 12

# Initialize model
net = ukko.core.DualAttentionRegressor(
#model = ukko.core.DualAttentionModelOld(
    n_features=n_features,
    time_steps=sequence_length,
    d_model=8,
    n_heads=4,
    dropout=0.1
)

print("Creating model")
optimizer = tt.optim.Adam(0.005)
model = DeepHitSingle(net, optimizer, duration_index=[])

print(model)
model.net
45/35:
# pycox data example
from pycox.datasets import metabric
45/36:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)
45/37:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
45/38:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

type(x_train)
45/39:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
type(x_train)
45/40:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print("{type(x_train)}: x_train.shape")
45/41:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print("{type(x_train)}: {x_train.shape}")
45/42:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")
45/43:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {y_train_surv.shape}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: {train.shape}")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/44:
import pycox
from pycox.models import DeepHitSingle
import torchtuples as tt
45/45:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.preprocessing.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {y_train_surv.shape}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: {train.shape}")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/46:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.models.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {y_train_surv.shape}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: {train.shape}")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/47:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.models.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {y_train_surv.shape}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: ")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/48:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.models.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: ")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: ")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/49: y_train_surv
45/50:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.models.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {len_y_train_surv}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: ")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
45/51:
# pycox data example
from pycox.datasets import metabric

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

# x data:
cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

print("Type of x data:")
print(f"{type(x_train)}: {x_train.shape}")

# y data:
num_durations = 10
labtrans = pycox.models.LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train))
y_val_surv = labtrans.transform(*get_target(df_val))

print("Type of y data:")
print(f"{type(y_train_surv)}: {len(y_train_surv)}")



train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = tt.tuplefy(x_val, (y_val_surv, x_val))
print("Type of train data:")
print(f"{type(train)}: ")

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)
   1:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
print("Libraries loaded")
   2:
class SurvivalHead(nn.Module):
    """
    Neural Cox proportional hazards model head.
    Outputs hazard ratios instead of binary classification.
    """
    def __init__(self, d_model, n_features, dropout=0.1):
        super().__init__()
        self.risk_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # log hazard ratio
        )

    def forward(self, x):
        # Risk score (log hazard ratio)
        return torch.exp(self.risk_score(x))  # hazard ratio

    
def cox_loss(risk_scores, survival_times, events):
    """
    Negative log partial likelihood for Cox model
    
    Args:
        risk_scores: predicted hazard ratios
        survival_times: time to event/censoring
        events: 1 if event occurred, 0 if censored
    """
    # Sort by survival time
    _, indices = torch.sort(survival_times, descending=True)
    risk_scores = risk_scores[indices]
    events = events[indices]
    
    # Calculate log partial likelihood
    log_risk = torch.log(torch.cumsum(torch.exp(risk_scores), 0))
    likelihood = risk_scores - log_risk
    
    # Mask for events only
    return -torch.mean(likelihood * events)
   3:
class SurvivalDataset(Dataset):
    def __init__(self, features, survival_times, events):
        self.features = features
        self.times = survival_times    # Time to event/censoring
        self.events = events           # Event indicator (1=death, 0=censored)
   4:
import torch
import torch.nn as nn
import math
import ukko 
import importlib
# For preprocessing
print("Loading sklearn")
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
print("Libraries loaded")
   5:
class SurvivalDataset(Dataset):
    def __init__(self, features, survival_times, events):
        self.features = features
        self.times = survival_times    # Time to event/censoring
        self.events = events           # Event indicator (1=death, 0=censored)
   6:
#Load tidy data
print("Loading tidy data")
df_xy = pd.read_csv("data/df_xy_synth_v1.csv")
# IMPUTE nan: -1
df_xy = df_xy.fillna(-1)

# Define function to convert df into 3-D numpy array
def convert_to_3d_df(df):

    # Convert column names to tuples, assuming this "('feature', timepoint)"
    columns = [eval(col) for col in df.columns]
    df.columns = columns
    
    # Extract unique features and timepoints
    features = sorted(list(set([col[0] for col in columns])))
    timepoints = sorted(list(set([col[1] for col in columns])))
    
    # Initialize a 3D numpy array
    n_rows = df.shape[0]
    n_features = len(features)
    n_timepoints = len(timepoints)
    data_3d = np.empty((n_rows, n_features, n_timepoints))
    data_3d.fill(np.nan)
    
    # Map feature names and timepoints to indices
    feature_indices = {feature: i for i, feature in enumerate(features)}
    timepoint_indices = {timepoint: i for i, timepoint in enumerate(timepoints)}
    
    # Fill the 3D array with data from the DataFrame
    for col in columns:
        feature, timepoint = col
        feature_idx = feature_indices[feature]
        timepoint_idx = timepoint_indices[timepoint]
        data_3d[:, feature_idx, timepoint_idx] = df[col]

    # Create a MultiIndex for the columns of the 3D DataFrame
    columns = pd.MultiIndex.from_product([features, timepoints], names=["Feature", "Timepoint"])
    
    # Create the 3D DataFrame
    df_multiindex = pd.DataFrame(data_3d.reshape(n_rows, -1), columns=columns)
    
    return df_multiindex, data_3d

# Convert AML data to multiindex df
df_x, data_3d = convert_to_3d_df(df_xy.iloc[:,3:].fillna(-1))
df_y = df_xy.iloc[:,:3]
display(df_x)
display(df_y)
   7:
class LinearCoxPH(nn.Module):
    """
    Classical Cox PH with linear predictor: h(t|x) = h₀(t)exp(βx)
    """
    def __init__(self, n_features):
        super().__init__()
        self.beta = nn.Linear(n_features, 1, bias=False)  # β coefficients
        
    def forward(self, x):
        return torch.exp(self.beta(x))  # exp(βx)
   8: %history -g
   9: %history -g -f "recovery.py"
