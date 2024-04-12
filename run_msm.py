from typing import *
from pathlib import Path
import gc
from natsort import natsorted

from func_build_msm import *

gc.enable()

traj_paths = Path('/exports/eddie/scratch/s2135271/chignolin/').rglob('*.xtc')
traj_paths = natsorted([str(p) for p in traj_paths])
top_path = '/exports/eddie/scratch/s2135271/chignolin/protein.pdb'
hp_table = pd.read_hdf('data/hpsample_full.h5')

study_name = 'chignolin'
# save_dir = Path(f'/exports/csce/eddie/chem/groups/Mey/Ryan/msm_data/{study_name}/')
save_dir = Path(f'{study_name}/')

run_study(hp_table=hp_table, traj_paths=traj_paths, top_path=top_path, study_name=study_name, save_dir=save_dir)
