#! /usr/bin/env python3
import ruamel.yaml as yaml
import numpy as np
import itertools
import os
from collections import OrderedDict
import jinja2
from anytree.importer import DictImporter
from anytree.render import RenderTree as RT
from anytree.iterators import LevelOrderIter, PostOrderIter
from sklearn.model_selection import ParameterGrid
import re

import argparse
parser=argparse.ArgumentParser()

parser.add_argument("--template_folder",default="./template",help="template folder location",dest='template')
parser.add_argument("--params_file",default="experiment_params.yaml",help="manipulations for experiment file",dest='params')
parser.add_argument("--output_grid",default="grid",help='location to output grid')


    # setup parameter grid function:
def pg(x):
    '''return parameter grid as list'''
    return list(ParameterGrid(x))

    # flatten lists
def flatten(x):
    '''flatten nested lists'''
    def fl(x):
        for i in x:
            if isinstance(i, list):
                yield from fl(i)
            else:
                yield i

    return list(fl(x))

def multireplace(string, replacements):
    substrs = sorted(replacements, key=len, reverse=True)
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    out=regexp.sub(lambda match: replacements[match.group(0)], string)
    return out

#</editor-fold>




def is_simple_list(x):
    '''determine if list contains additional dictionaries. return tuple (true/false, child or flatlist)'''
    for i in flatten(x):
        if not isinstance(i, (str, bytes, int, float)):
            return (False, x)
    # need to convert numbers to strings to maintain compatibility w/yaml.
    out=[str(i) for i in flatten(x)]
    return (True, out)

def format_dict(x, id=None):
    base = {'children': list(), 'id': id, 'attr': {}}
    for k, v in x.items():
        if isinstance(v,str):
            try:
                v=eval(v)
                print(v)
            except:
                v=[v]
        if isinstance(v, list):
            simple_list,list_items = is_simple_list(v)
            if simple_list:
                base['attr'][k] = list_items
            else:
                for i in list_items:
                    base.get('children').append(format_dict(i, id=k))
        else:
            base.get('children').append({k:v})
    return base

def generate_params(x):
    tree=format_dict(x,id=None)
    DI=DictImporter()
    tree=DI.import_(tree)
    PI = PostOrderIter(tree)
    for i in PI:
        if i.parent:
            if i.id in i.parent.attr.keys():
                i.parent.attr[i.id].append(pg(i.attr))
            else:
                i.parent.attr.update(**{i.id:pg(i.attr)})
            i.parent.attr.update(**{i.id:flatten(i.parent.attr[i.id])})
    return pg(tree.attr)

def make_paths(gridpath,x):
    str_dict = {"'": "", " ": "_","{": "","}": "",":": "",",": ""}
    return os.path.join(gridpath,multireplace(str(x),str_dict))

def mkgrid(x="experiment_params.yaml",grid_folder='grid'):
    thedict=OrderedDict(yaml.load(open(x,'r')))
    grid=generate_params(thedict['grid'])
    extra_params={k: v for k, v in thedict.items() if k != 'grid'}
    for i in grid:
        i.update(path=make_paths(grid_folder,i))
        i.update(extra_params)
    return grid



def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)




def render_template_folder(manipulation, output_location='grid', template_folder='template'):
    os.makedirs(output_location, exist_ok=True)
    ignore_files = ['.DS_Store']
    files=[i for i in os.listdir(template_folder) if i not in ignore_files]
    for i in files:
        output=os.path.join(output_location,i)
        ignore_files=[".DS_Store"]
        if os.path.isdir("./template/" + i):
            print("isdir")
            continue
        try:
            with open(output,'w') as f:
                f.write(render("./template/" + i, manipulation))
        except Exception as e:
            print("error on {}: {}".format(i,e))



if __name__=="__main__":
    args = parser.parse_args()
    grid=mkgrid(args.params,grid_folder=args.output_grid)
    for i in grid:
        print(i)
        render_template_folder(i,i['path'],template_folder=args.template)