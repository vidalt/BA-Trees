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
import persistence
from IPython.display import display
import ipywidgets as widgets


def create_objective_selection(show=True):
    select = widgets.Dropdown(
        options=[('Depth', 0), ('NbLeaves', 1), ('Depth > NbLeaves', 2), ('Heuristic', 4)],
        value=4,
        description='Objective:',
    )
    if show:
        display(select)
    return select


def create_depth_selection(show=True):
    select = widgets.IntSlider(
            value=3,
            min=2,
            max=5,
            step=1,
            description='Max depth:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
    if show:
        display(select)
    return select


def create_n_trees_selection(show=True):
    select = widgets.IntSlider(
        value=10,
        min=3,
        max=10,
        step=1,
        description='#Trees:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
    )
    if show:
        display(select)
    return select


def load(X, y, dataset, fold, n_trees, F=None, S=None, return_file=False):
    respath = str(pathlib.Path(__file__).parent.absolute()) + '/resources/forests'
    if F or S:
        filename = '{}.F{}.S{}.RF{}.txt'.format(dataset, F, S, fold)
        filename = '{}/{}/F{}.S{}/{}'.format(respath, dataset, F, S, filename)
    else:
        filename = '{}.RF{}.txt'.format(dataset, fold)
        filename = '{}/{}/{}'.format(respath, dataset, filename)

    clf = persistence.classifier_from_file(filename, X, y, pruning=True, num_trees=n_trees)
    if return_file:
        return clf, filename
    return clf
