from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import mdtraj as md
import pyemma as pm
from tqdm import tqdm
from addict import Dict as Adict
import time
import gc
from pathlib import Path

from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator


def run_study(hp_table, traj_paths, top_path, study_name, save_dir):

    print('No of hp trials: ', len(hp_table))

    for _, hps in tqdm(hp_table.iterrows(), total=len(hp_table)):
        hp_dict = Adict(hps.to_dict())
        print(hp_dict)

        ftrajs_all = featurizer(hp_dict, traj_paths, top_path)
        trial_dir = save_dir/f'{hp_dict.hp_id}'
        trial_dir.mkdir(parents=True, exist_ok=True)

        bootstrap_hp_trial(hp_dict, ftrajs_all, study_name, save_dir)

    return None


def featurizer(hp_dict: Dict, traj_paths: List[str], top_path: str) -> List[np.ndarray]:
    if hp_dict['feature__value'] == 'dihedrals':
        assert hp_dict['dihedrals__which'] == 'all'
        def f(traj: md.Trajectory, **kwargs) -> np.ndarray:
            _, phi = md.compute_phi(traj)
            _, psi = md.compute_psi(traj)
            _, chi1 = md.compute_chi1(traj)
            _, chi2 = md.compute_chi2(traj)
            _, chi3 = md.compute_chi3(traj)
            _, chi4 = md.compute_chi4(traj)
            _, chi5 = md.compute_chi5(traj)
            ftraj = np.concatenate([phi, psi, chi1, chi2, chi3, chi4, chi5], axis=1)
            ftraj = np.concatenate([np.cos(ftraj), np.sin(ftraj)], axis=1)
            return ftraj
    elif hp_dict['feature__value'] == 'distances':
        def f(traj: md.Trajectory, **kwargs):
            scheme = kwargs['distances__scheme']
            transform = kwargs['distances__transform']
            centre = kwargs['distances__centre']
            steepness = kwargs['distances__steepness']
            ftraj, _ = md.compute_contacts(traj, scheme=scheme)
            if transform=='logistic':
                ftraj = 1.0/(1 + np.exp(-steepness*(ftraj - centre)))
            return ftraj
    else:
        raise ValueError
    ftrajs = []
    for traj_path in traj_paths:
        traj = md.load(traj_path, top=top_path)
        ftrajs.append(f(traj, **hp_dict))
    return ftrajs


def bootstrap_hp_trial(hp_dict, ftrajs_all, study_name, save_dir):
    start_time = time.time()

    n_boot = hp_dict.n__boot
    rng = np.random.default_rng(hp_dict.seed)

    ftraj_lens = [x.shape[0] for x in ftrajs_all]

    if n_boot == 1:
        ftraj_ixs = [np.arange(len(ftraj_lens))]
    else:
        ftraj_ixs = [_bootstrap(ftraj_lens, rng) for _ in range(n_boot)]

    for i, ix in tqdm(enumerate(ftraj_ixs), total=n_boot):
        print('\nBootstrap: ', i)
        f_kmeans = save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_kmeans_centers.npy'
        f_tmat = save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_msm_tmat.npy'
        f_ix = save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_traj_indices.npy'
        if f_kmeans.is_file() and f_tmat.is_file() and f_ix.is_file(): 
            print('Already exist. Continue')
            continue

        np.save(f_ix, ix)
        ftrajs = [ftrajs_all[i] for i in ix]
        _estimate_msm(hp_dict, ftrajs, i, study_name, save_dir)

    print('Time elapsed: ', time.time() - start_time)

    return None


def _bootstrap(ftrajs: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    probs = np.array([x.shape[0] for x in ftrajs])
    probs = probs/np.sum(probs)
    ix = np.arange(len(ftrajs))
    new_ix = rng.choice(ix,size=len(ftrajs), p=probs, replace=True)
    return [ftrajs[i] for i in new_ix], new_ix


def _tica(hp_dict: Dict, ftrajs: List[np.ndarray]):
    lag = hp_dict.tica__lag
    stride = hp_dict.tica__stride
    dim = hp_dict.tica__dim
    tica__kinetic_map = hp_dict.tica__kinetic_map

    tica_mod = pm.coordinates.tica(ftrajs, lag=lag, stride=stride, dim=dim, kinetic_map=tica__kinetic_map)
    ttrajs = tica_mod.get_output()

    return ttrajs, tica_mod


def _kmeans(hp_dict: Dict, ttrajs: List[np.ndarray], seed: int):

    n_clusters = hp_dict.cluster__k
    stride = hp_dict.cluster__stride
    max_iter = hp_dict.cluster__max_iter

    kmeans_mod = pm.coordinates.cluster_kmeans(ttrajs, k=n_clusters, max_iter=max_iter, stride=stride, fixed_seed=seed)
    dtrajs = kmeans_mod.dtrajs

    return dtrajs, kmeans_mod


def _estimate_msm(hp_dict, ftrajs, i, study_name, save_dir):
    print('Estimating bootstrap: ', i)
    n_score = 20
    columns = ['bs'] + ['is_sparse'] + [f't{i+2}' for i in range(n_score)] \
            + [f'gap_{i+2}' for i in range(n_score)] \
            + [f'vamp2_{i+2}' for i in range(n_score)] \
            + [f'vamp2eq_{i+2}' for i in range(n_score)]

    try:
        ttrajs, tica_mod = _tica(hp_dict, ftrajs)
        dtrajs, kmeans_mod = _kmeans(hp_dict, ttrajs, hp_dict.seed)
    except:
        print('TICA/Kmeans failed -- skip')
        results = [f'{i}'].extend([np.nan]*(len(columns)-1))
        return None 
    
    count_mod = TransitionCountEstimator(lagtime=hp_dict.markov__lag, count_mode='sliding').fit_fetch(dtrajs)
    msm_mod = MaximumLikelihoodMSM(reversible=True).fit_fetch(count_mod)

    results = []
    results.append(f'{i}')
    results.append(msm_mod.transition_matrix.shape[0] != hp_dict.cluster__k)
    results.extend(msm_mod.timescales()[n_score])
    results.extend(msm_mod.timescales()[0:n_score]/msm_mod.timescales()[1:n_score+1])
    results.extend([msm_mod.score(dtrajs, r=2, dim=i+1) for i in range(n_score)])
    results.extend([sum(msm_mod.eigenvalues(i+2)**2) for i in range(n_score)])

    print('Saving results')
    data = pd.DataFrame({k:v for k,v in zip(columns, results)}, index=[0])
    data.to_hdf(save_dir/f'{study_name}.h5', key=f'result_raw', mode='a', format='table', append=True, data_columns=True)
    np.save(save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_kmeans_centers.npy', kmeans_mod.clustercenters)
    np.save(save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_msm_tmat.npy', msm_mod.transition_matrix)
    
    gc.collect()

    return None
