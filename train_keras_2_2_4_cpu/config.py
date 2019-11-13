import os
from sacred import Experiment

ex = Experiment()

source_filepaths = [
    "train.py"
]

for fpath in source_filepaths:
    if os.path.exists(fpath):
        ex.add_source_file(fpath)

@ex.config
def ex_config():
    input_shape = (150, 150, 3)
