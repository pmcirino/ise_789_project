import numpy as np
from pathlib import Path
import os
root = Path('.')
ACTION_PATH = '/Users/paul/Documents/789_Project/ise_789_project/output_files/actions'

VALUE_PATH = '/Users/paul/Documents/789_Project/ise_789_project/output_files/values'

def save_results(name, results):
    action_path = os.path.join(ACTION_PATH, f'{name}_actions.npy')
    value_path = os.path.join(VALUE_PATH, f'{name}_values.npy')
    print(value_path)
    np.save(action_path, np.array(results['policy']))
    np.save(value_path, np.array(results['values']))

def open_actions(name):
    actions = np.load(os.path.join(ACTION_PATH, f'{name}'))
    return actions

def open_values(name):
    values = np.load(os.path.join(VALUE_PATH, f'{name}'))
    return values