# MIT License

# Copyright(c) 2020 Toni Pacheco

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pathlib
import pandas as pd
import visualization as tree_view
from IPython.display import display


dataset_names = [
        "Breast-Cancer-Wisconsin",
        "COMPAS-ProPublica",
        "FICO",
        "HTRU2",
        "Pima-Diabetes",
        "Seeds"
    ]


def create_dataset_selection(show=True ,no_fico_sa=True):
    import ipywidgets as widgets
    selected_datasets = widgets.Select(
        options=dataset_names,
        value=dataset_names[0],
        description="Datasets",
        disabled=False
    )
    if show:
        display(selected_datasets)
    return selected_datasets


def create_kfold_selection(min_v=1, max_v=10, show=True):
    import ipywidgets as widgets
    select = widgets.IntSlider(
            value=1,
            min=min_v,
            max=max_v,
            step=1,
            description='Fold:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
    if show:
        display(select)
    return select

def create_cplex_linking_selection(show=True):
    import ipywidgets as widgets
    select = widgets.Checkbox(
        value=False,
        description='CPLEX linking',
        disabled=False,
        indent=False
    )
    if show:
        display(select)
    return select

def load_info(dataset, df_train, fn):
    info = {
        'classes': {},
        'filename': fn,
        'colors': None,
    }
    info['features'] = {k:v for k,v in zip(range(len(df_train.columns)-1), df_train.columns[:-1])}
    return info


def load(dataset, fold, F=None, S=None):
    respath = str(pathlib.Path(__file__).parent.absolute()) + '/resources/datasets/'

    if F or S:
        fn = respath+'{}/F{}.S{}/{}.F{}.S{}.train{}.csv'.format(dataset, F, S, dataset, F, S, fold)
        df_train = pd.read_csv(fn)
        df_test = pd.read_csv(respath+'{}/F{}.S{}/{}.F{}.S{}.test{}.csv'.format(dataset, F, S, dataset, F, S, fold))
    else:
        fn = respath+'{}/{}.train{}.csv'.format(dataset, dataset, fold)
        df_train = pd.read_csv(fn)
        df_test = pd.read_csv(respath+'{}/{}.test{}.csv'.format(dataset, dataset, fold)) 

    return df_train, df_test, load_info(dataset, df_train, fn)

