import numpy as np
import copy
import pickle
import os

from ase import neighborlist


def align(s1, s2):
    """
    given two ASE atoms objects, `s1` and `s2`,
    aligns the positions of `s2` to `s1`.

    basic alignment using mass, does not use centers of charge
    """

    s1c = copy.deepcopy(s1)
    s2c = copy.deepcopy(s2)

    r1 = s1c.get_positions()
    r2 = s2c.get_positions()

    # shift the CM to the origin
    cm1 = np.mean(r1, axis=0)
    r1 -= cm1

    cm2 = np.mean(r2, axis=0)
    r2 -= cm2

    Ra = np.linalg.pinv(r1.T @ r2) @ (r1.T @ r1)
    U, S, Vh = np.linalg.svd(Ra)
    Ri = U @ np.diag([1, 1, -1]) @ Vh

    Rb = U @ Vh

    if np.sum((r1 - r2 @ Ri) ** 2) < np.sum((r1 - r2 @ Rb) ** 2):
        R = Ri
    else:
        R = Rb

    s2c.set_positions(r2 @ R + cm2)
    return s2c


def process_dataset(t_list, sf, df):
    with open(sf, "rb") as f:
        ase_dict = pickle.load(f)

    # handle per state values first
    distances = {}
    connectivity = {}
    for s_id, a in ase_dict.items():
        distances[s_id] = np.array(a.get_all_distances()).astype(np.float32)
        nl = neighborlist.build_neighbor_list(a)
        connectivity[s_id] = np.array(nl.get_connectivity_matrix(sparse=False))

    # calculate alignment for transitions
    aligned_positions = {}
    t_ase_dict = {}
    avgBondDelta = {}
    for t in t_list:
        s1id, s2id = t

        s1a = ase_dict[s1id]
        s2a = ase_dict[s2id]

        s2c = align(s1a, s2a)
        t_ase_dict[t] = (s1a, s2c)
        aligned_positions[t] = (
            s1a.get_positions().astype(np.float32),
            s2c.get_positions().astype(np.float32),
        )

        bd = np.abs(distances[s2id] - distances[s1id])
        avgBondDelta[t] = np.mean(bd * connectivity[s1id], axis=1).astype(np.float32)

    with open(f"{df}/distances.pickle", "wb") as f:
        pickle.dump(distances, f)

    with open(f"{df}/aligned_positions.pickle", "wb") as f:
        pickle.dump(aligned_positions, f)

    with open(f"{df}/t_ase_dict.pickle", "wb") as f:
        pickle.dump(t_ase_dict, f)

    scalarsf = f"{df}/scalars"
    if not os.path.exists(scalarsf):
        os.mkdir(scalarsf)

    with open(f"{scalarsf}/absAvgBonds.pickle", "wb") as f:
        pickle.dump(avgBondDelta, f)

    del distances
    del connectivity
    del aligned_positions
    del t_ase_dict
    del avgBondDelta
