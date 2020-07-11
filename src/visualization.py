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

import numpy as np
import graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

_colors = [ "{}".format(hex((int(255*c[0])<<16) + 
                           (int(255*c[1])<<8) +
                            int(255*c[2]))).replace('0x', '#')
            for c in plt.get_cmap('Pastel2').colors[:-1]]

def to_dot_format(trees, features={}, classes={},
                  colors = None, simplified=True, gini=False):
    
    if colors is None:
        colors = _colors.copy()
        
    def _get_node_color(info):
        feature = info[2]
        if not simplified and info[5] == 0:
            return "#eeeeee"
        if  feature >= 0:
            return colors[feature % (len(colors))]
        return "#ffffff"
    
    def _fix_text(text_):
        text = text_.replace(':', ' = ')
        
        text = text.replace("= >","&le; ")
        text = text.replace("= <","&ge; ")
        text = text.replace("= =","&ne; ")
        text = text.replace("=","&ne;")
        text = text.replace(">"," &le; ")
        text = text.replace("<"," &ge; ")
        
        return text, text != text_
    
    def _get_node_text(info, values):
        feature = info[2]
        threshold = info[3]
        impurity = info[4]
        n_samples = info[5]
        
        info = '<'
        if feature >= 0:
            ft_text, fixed = _fix_text(features.get(feature, 'X[{}]'.format(feature)))
            if fixed:
                info += '{}<br/>'.format(ft_text, threshold)
            else:
                info += '{} &le; {}<br/>'.format(ft_text, threshold)
        else:
            output = np.argmax(values[0])
            info += '<u>{}</u><br/>'.format(classes.get(output, 'Class {}'.format(output)))
            
        if not simplified:
            if gini:
                info += 'gini = {:.3f}<br/>'.format(impurity)
            info += 'samples = {}<br/>'.format(n_samples)

        return info + '>'
    
    

    dot_content = 'digraph Tree {\n'
    dot_content += '\tnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n'
    dot_content += '\tedge [fontname=helvetica] ;\n'

    offset = 0
    for tree in trees:
        dot_content += '\n'
        
        d = tree.__getstate__()
        for idx, node_info in enumerate(d['nodes']):
            dot_content += '\t{} [label={}, fillcolor=\"{}\"] ;\n'.format(idx+offset,
                _get_node_text(node_info, d['values'][idx]),
                _get_node_color(node_info),
            )

        dot_content += '\n'

        for idx, node_info in enumerate(d['nodes']):
            true_lbl =  '[labeldistance=2.5, labelangle=45, headlabel="True"]' if idx==0  else ''
            false_lbl = '[labeldistance=2.5, labelangle=-45, headlabel="False"]' if idx==0 else ''
            if node_info[0] >= 0:
                dot_content += '\t{} -> {} {}\n '.format(idx+offset, node_info[0]+offset, true_lbl)
            if node_info[1] >= 0:   
                dot_content += '\t{} -> {} {}\n '.format(idx+offset, node_info[1]+offset, false_lbl)
        
        offset += d['node_count']
        
    return dot_content + '}'

def create_graph(tree, features={}, classes={},
               colors = None,  simplified=True, gini=False):
    
    if colors is None:
        colors = _colors.copy()
    
    dot_data = to_dot_format(tree, features, classes, colors, simplified, gini)
    graph = graphviz.Source(dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data)  
    return graph


def tree_map(ax, tree, classes, features, fixed_values, 
            colors=None, all_limits=None):
    
    if colors is None:
        colors = _colors.copy()
        
    def deep_first_squares(rects, cur, nodes, values, fixed_values, limits,
                      min1=None, max1=None, min2=None, max2=None):
    
        if nodes[cur][2] == -1:
            min1 = min1 if min1 is not None else limits[axis_keys[0]]['min']
            max1 = max1 if max1 is not None else limits[axis_keys[0]]['max']
            min2 = min2 if min2 is not None else limits[axis_keys[1]]['min']
            max2 = max2 if max2 is not None else limits[axis_keys[1]]['max']

            pred = np.argmax(values[cur][0])
            rects.append(
                patches.Rectangle((min1,min2), max1-min1, max2-min2,linewidth=1, edgecolor='k',
                                  facecolor=colors[pred % (len(colors))])
            )
        else:
            val2 = nodes[cur][3]
            if nodes[cur][2] in fixed_values:
                val1 = fixed_values[nodes[cur][2]]
                deep_first_squares(
                    rects, nodes[cur][0] if val1 <= val2 else nodes[cur][1],
                    nodes, values, fixed_values, limits,
                    min1, max1, min2, max2
                )
            else:
                if nodes[cur][2] == axis_keys[0]:
                    deep_first_squares(
                        rects, nodes[cur][0],
                        nodes, values, fixed_values, limits,
                        min1, val2, min2, max2
                    )
                    deep_first_squares(
                        rects, nodes[cur][1],
                        nodes, values, fixed_values, limits,
                        val2, max1, min2, max2
                    )
                else:
                    deep_first_squares(
                        rects, nodes[cur][0],
                        nodes, values, fixed_values, limits,
                        min1, max1, min2, val2
                    )
                    deep_first_squares(
                        rects, nodes[cur][1],
                        nodes, values, fixed_values, limits,
                        min1, max1, val2, max2
                    )
    
    
    #compute limits
    if all_limits is None:
        all_limits = {k: {'min': 999, 'max': -999} for k in features}
    nodes = tree.__getstate__()['nodes']
    values = tree.__getstate__()['values']
    for info in nodes:
        if info[2] in features:
            all_limits[info[2]]['min'] = min([all_limits[info[2]]['min'], int(info[3]-1)])
            all_limits[info[2]]['max'] = max([all_limits[info[2]]['max'], int(info[3]+1)])
            
    variable_features = set([x for x in features.keys() if x not in fixed_values])
    assert(len(variable_features)==2)
    
    limits = {k: all_limits[k] for k in variable_features}
    axis_keys = list(limits.keys())
    axis_keys, limits
                        
    #plotting
    rects = []
    deep_first_squares(rects, 0, nodes, values, fixed_values, limits)
    for r in rects[::-1]:
        ax.add_patch(r)

    custom_labels = list(classes.values())
    custom_lines = [Line2D([0], [0], color=colors[c%len(colors)], lw=4)
                   for c in classes]

    ax.legend(custom_lines, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel(features[axis_keys[0]])
    plt.ylabel(features[axis_keys[1]])

    ax.set_xlim(limits[axis_keys[0]]['min'], limits[axis_keys[0]]['max'])
    ax.set_ylim(limits[axis_keys[1]]['min'], limits[axis_keys[1]]['max'])
