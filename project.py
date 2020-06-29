from flow import FlowProject
import signac
import flow
# import pairing
import matplotlib.pyplot as plt
import mbuild as mb
import mdtraj as md
from mtools.pairing import chunks
from scipy import stats
import numpy as np
import pickle
from foyer import Forcefield
from scipy.optimize import curve_fit
# from get_mol2 import GetSolv, GetIL
# from util.decorators import job_chdir
from pkg_resources import resource_filename
from mtools.gromacs.gromacs import make_comtrj
from mtools.post_process import calc_msd
# from ramtools.conductivity import calc_conductivity
from mtools.post_process import calc_density
from multiprocessing import Pool
from scipy.special import gamma
import os
# import environment
import itertools as it
import gzip
import parmed as pmd
import shutil
# import gafffoyer
# import antefoyer
# from simtk.unit import *


def _pairing_func(x, a, b):
    """Stretched exponential function for fitting pairing data"""
    y = np.exp(-1 * b * x ** a)
    return y


def workspace_command(cmd):
    """Simple command to always go to the workspace directory"""
    return ' && '.join([
        'cd {job.ws}',
        cmd if not isinstance(cmd, list) else ' && '.join(cmd),
        'cd ..',
    ])


def _run_overall(trj, mol):
    D, MSD, x_fit, y_fit = calc_msd(trj)
    return D, MSD


def _save_overall(job, mol, trj, MSD):
    np.savetxt(os.path.join(job.workspace(), 'msd-{}-overall_2.txt'.format(mol)),
               np.transpose(np.vstack([trj.time, MSD])),
               header='# Time (ps)\tMSD (nm^2)')

    fig, ax = plt.subplots()
    ax.plot(trj.time, MSD)
    ax.set_xlabel('Simulation time (ps)')
    ax.set_ylabel('MSD (nm^2)')
    fig.savefig(os.path.join(job.workspace(),
                             'msd-{}-overall_2.pdf'.format(mol)))


def _run_multiple(trj, mol):
    D_pop = list()
    for start_frame in np.linspace(0, 1001, num=200, dtype=np.int):
        end_frame = start_frame + 200
        if end_frame < 1001:
            chunk = trj[start_frame:end_frame]
            print('\t\t\t...frame {} to {}'.format(start_frame, end_frame))
            try:
                D_pop.append(calc_msd(chunk)[0])
            except TypeError:
                import pdb
                pdb.set_trace()
        else:
            continue
    D_bar = np.mean(D_pop)
    D_std = np.std(D_pop)
    return D_bar, D_std


init_file = 'gaff.gro'
em_file = 'em.gro'
nvt_file = 'nvt.gro'
npt_file = 'npt.gro'
sample_file = 'sample.gro'
unwrapped_file = 'sample_unwrapped.xtc'
msd_file = 'msd-all-overall_2.txt'
pair_file = 'direct-matrices-cation-anion.pkl.gz'
pair_fit_file = 'matrix-pairs-solvent-anion.txt'
tau_file = 'tau.txt'
rdf_file = 'rdf-solvent-solvent.txt'
all_directs_file = 'all-directs-solvent-cation.pkl.gz'
all_indirects_file = 'all-indirects-solvent-cation.pkl'
cn_file = 'cn-cation-anion-2.txt'


class Project(FlowProject):
    pass


@Project.label
def initialized(job):
    return job.isfile(init_file)


@Project.label
def minimized(job):
    return job.isfile(em_file)


@Project.label
def nvt_equilibrated(job):
    return job.isfile(nvt_file)


@Project.label
def npt_equilibrated(job):
    return job.isfile(npt_file)


@Project.label
def sampled(job):
    return job.isfile(sample_file)


@Project.label
def prepared(job):
    return job.isfile(unwrapped_file)


@Project.label
def msd_done(job):
    return job.isfile(msd_file)


@Project.label
def pair_done(job):
    return job.isfile(pair_file)


@Project.label
def pair_fit_done(job):
    return job.isfile(pair_fit_file)


@Project.label
def directs_done(job):
    return job.isfile(all_directs_file)


@Project.label
def indirects_done(job):
    return job.isfile(all_indirects_file)


@Project.label
def tau_done(job):
    return job.isfile(tau_file)


@Project.label
def rdf_done(job):
    return job.isfile(rdf_file)

# @Project.label
# def cn_done(job):
#    return job.isfile(cn_file)


@Project.operation
@Project.post.isfile(init_file)
def initialize(job):
    wd = os.getcwd()
    print(wd)
    with job:
        if job.statepoint()['Density']:
            n_anion = job.statepoint()['n_anion']
            n_cation = job.statepoint()['n_cation']
            anion = job.statepoint()['anion']
            cation = job.statepoint()['cation']
            density = job.statepoint()['density']
            # Load in mol2 files as mb.Compound
            cation = mb.load(
                wd+'/util/' + str(job.statepoint()['cation'])+'.mol2')
            cation.name = job.statepoint()['cation']
            anion = mb.load(
                wd + '/util/' + str(job.statepoint()['anion']) + '.mol2')
            anion.name = job.statepoint()['anion']

            an_ff = Forcefield(wd+'/util/'+job.statepoint()
                               ['an_forcefield'] + '.xml')
            cat_ff = Forcefield(wd+'/util/'+job.statepoint()
                                ['cat_forcefield'] + '.xml')

            system = mb.fill_box(compound=[cation, anion],
                                 n_compounds=[n_cation, n_anion],
                                 density=density)

            cation = mb.Compound()
            anion = mb.Compound()
            for child in system.children:
                if child.name == job.statepoint()['cation']:
                    cation.add(mb.clone(child))
                elif child.name == job.statepoint()['anion']:
                    anion.add(mb.clone(child))

            catPM = cat_ff.apply(cation, residues=[job.statepoint()['cation']])
            anPM = an_ff.apply(
                anion, residues=[job.statepoint()['anion']])

            scale = job.statepoint()['charge_scale']
            if scale != 1.0:
                print("Scaling charges ... ")
                for atom in anPM.atoms:
                    atom.charge *= scale
                for atom in catPM.atoms:
                    atom.charge *= scale

            systemPM = catPM + anPM
            systemPM.save('init.gro', combine='all', overwrite=True)
            systemPM.save('init.top', combine='all', overwrite=True)


@Project.operation
@Project.pre.isfile(init_file)
@Project.post.isfile(em_file)
@flow.cmd
def em(job):
    return _gromacs_str('em', 'init', 'init', job)


@Project.operation
@Project.pre.isfile(em_file)
@Project.post.isfile(nvt_file)
@flow.cmd
def nvt(job):
    return _gromacs_str('nvt', 'em', 'init', job)


@Project.operation
@Project.pre.isfile(nvt_file)
@Project.post.isfile(npt_file)
@flow.cmd
def npt(job):
    return _gromacs_str('npt', 'nvt', 'init', job)


@Project.operation
@Project.pre.isfile(npt_file)
@Project.post.isfile(sample_file)
@flow.cmd
def sample(job):
    return _gromacs_str('sample', 'npt', 'init', job)


@Project.operation
@Project.pre.isfile(sample_file)
@Project.post.isfile(unwrapped_file)
def prepare(job):
    # if job.get_id() == '41fd6198b7f5675f9ecd034ce7c5af73':
    #    pass
    # else:
    trr_file = os.path.join(job.workspace(), 'sample.trr')
    xtc_file = os.path.join(job.workspace(), 'sample.xtc')
    gro_file = os.path.join(job.workspace(), 'sample.gro')
    tpr_file = os.path.join(job.workspace(), 'sample.tpr')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        unwrapped_trj = os.path.join(job.workspace(),
                                     'sample_unwrapped.xtc')
        # if not os.path.isfile(unwrapped_trj):
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc nojump'.format(
            xtc_file, unwrapped_trj, tpr_file))
        res_trj = os.path.join(job.ws, 'sample_res.xtc')
        com_trj = os.path.join(job.ws, 'sample_com.xtc')
        unwrapped_com_trj = os.path.join(job.ws, 'sample_com_unwrapped.xtc')
        # if not os.path.isfile(res_trj):
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -pbc res'.format(
            xtc_file, res_trj, tpr_file))
        # if os.path.isfile(res_trj) and not os.path.isfile(com_trj):
        trj = md.load(res_trj, top=gro_file)
        comtrj = make_comtrj(trj)
        comtrj.save_xtc(com_trj)
        comtrj[-1].save_gro(os.path.join(job.workspace(),
                                         'com.gro'))
        print('made comtrj ...')
        # if os.path.isfile(com_trj) and not os.path.isfile(unwrapped_com_trj)    :
        os.system('gmx trjconv -f {0} -o {1} -pbc nojump'.format(
            com_trj, unwrapped_com_trj))


@Project.operation
@Project.pre.isfile(unwrapped_file)
@Project.post.isfile(msd_file)
def run_msd(job):
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'sample.gro')
    trj_file = os.path.join(job.workspace(),
                            'sample_unwrapped.xtc')
    trj = md.load(trj_file, top=top_file)
    selections = {'all': trj.top.select('all'),
                  # 'ion' : trj.top.select('resname li tf2n'),
                  'cation': trj.top.select("resname li"),
                  'anion': trj.top.select("resname tf2n"),
                  'chloroform': trj.top.select('resname chlor'),
                  'ch3cn': trj.top.select('resname ch3cn')
                  }

    for mol, indices in selections.items():
        print('\tConsidering {}'.format(mol))
        if indices.size == 0:
            print('{} does not exist in this statepoint'.format(mol))
            continue
        print(mol)
        sliced = trj.atom_slice(indices)
        D, MSD = _run_overall(sliced, mol)
        job.document['D_' + mol + '_overall_2'] = D
        _save_overall(job, mol, sliced, MSD)

        sliced = trj.atom_slice(indices)
        D_bar, D_std = _run_multiple(sliced, mol)
        job.document['D_' + mol + '_bar_2'] = D_bar
        job.document['D_' + mol + '_std_2'] = D_std


@Project.operation
@Project.pre.isfile(msd_file)
@Project.post.isfile(pair_file)
def run_pair(job):
    print('hey')
    combinations = [['cation', 'anion']]
    for combo in combinations:
        if os.path.exists(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl.gz'.format(combo[0], combo[1]))):
            continue
        else:
            print('Loading trj {}'.format(job))
            trj_file = os.path.join(job.workspace(), 'sample.xtc')
            top_file = os.path.join(job.workspace(), 'sample.gro')
            trj = md.load(trj_file, top=top_file)
            anion = job.statepoint()['anion']
            cation = 'li'
            sliced = trj.topology.select(f'resname {cation} {anion}')
            distance = 0.8

            trj_slice = trj.atom_slice(sliced)
            trj_slice = trj_slice[:-1]
            direct_results = []
            print('Analyzing trj {}'.format(job))

            chunk_size = 500
            for chunk in chunks(range(trj_slice.n_frames), chunk_size):  # 500
                trj_chunk = trj_slice[chunk]
                first = make_comtrj(trj_chunk[0])
                first_direct = pairing.pairing._generate_direct_correlation(
                    first, cutoff=distance)

                # Math to figure out frame assignments for processors
                proc_frames = (len(chunk)-1) / 16
                remain = (trj_chunk.n_frames-1) % 16
                index = (trj_chunk.n_frames-1) // 16
                starts = np.empty(16)
                ends = np.empty(16)
                i = 1
                j = index+1
                for x in range(16):
                    starts[x] = i
                    if x < remain:
                        j += 1
                        i += 1
                    ends[x] = j
                    i += index
                    j += index
                starts = [int(start) for start in starts]
                ends = [int(end) for end in ends]
                params = [trj_chunk[i:j] for i, j in zip(starts, ends)]

                print('Checking direct')
                with Pool() as pool:
                    directs = pool.starmap(pairing.check_pairs, zip(params,
                                                                    it.repeat(distance), it.repeat(first_direct)))
                directs[0].insert(0, first_direct)
                directs = np.asarray(directs)
                direct_results.append(directs)

                print("saving now")

                with open(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl'.format(
                        combo[0], combo[1])), 'wb') as f:
                    pickle.dump(direct_results, f)

                with open(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl'.format(
                    combo[0], combo[1])), 'rb') as f_in, gzip.open(os.path.join(job.workspace(),
                                                                                'direct-matrices-{}-{}.pkl.gz'.format(combo[0], combo[1])), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print("saved")


@Project.operation
@Project.pre.isfile(pair_file)
@Project.post.isfile(pair_fit_file)
def run_pairing_fit_matrix(job):
    print(job.get_id())
    combinations = [['cation', 'anion']]
    for combo in combinations:
        print(combo)
        direct_results = []
        if os.path.exists(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl.gz'.format(combo[0], combo[1]))):
            with gzip.open(os.path.join(job.workspace(), 'direct-matrices-{}-{}.pkl.gz'.format(combo[0], combo[1])), 'rb') as f:
                direct_results = pickle.load(f)
            frames = 10000
            chunk_size = 500
            overall_pairs = []
            for chunk in direct_results:
                for proc in chunk:
                    for matrix in proc:
                        pairs = []
                        for row in matrix:
                            N = len(row)
                            count = len(np.where(row == 1)[0])
                            pairs.append(count)
                        pairs = np.sum(pairs)
                        pairs = (pairs - N) / 2
                        overall_pairs.append(pairs)

            ratio_list = []
            for i, pair in enumerate(overall_pairs):
                if i % chunk_size == 0:
                    divisor = pair
                    ratio_list.append(1)
                else:
                    if pair == 0:
                        ratio_list.append(0)
                    else:
                        pair_ratio = pair / divisor
                        ratio_list.append(pair_ratio)
            new_ratio = []
            i = 0
            for j in range(chunk_size, frames, chunk_size):
                x = ratio_list[i:j]
                new_ratio.append(x)
                i = j

            # mean = np.mean(new_ratio, axis=0)
            mean = np.mean(new_ratio[:12], axis=0)
            time_interval = [(frame * 1) for frame in range(chunk_size)]
            time_interval = np.asarray(time_interval)
            popt, pcov = curve_fit(_pairing_func, time_interval, mean)
            fit = _pairing_func(time_interval, *popt)

            np.savetxt(os.path.join(job.workspace(),
                                    'matrix-pairs-{}-{}.txt'.format(combo[0], combo[1])),
                       np.column_stack((mean, time_interval, fit)),
                       header='y = np.exp(-1 * b * x ** a) \n' +
                       str(popt[0]) + ' ' + str(popt[1]))

            job.document['pairing_fit_a_matrix_{}_{}'.format(
                combo[0], combo[1])] = popt[0]
            job.document['pairing_fit_b_matrix_{}_{}'.format(
                combo[0], combo[1])] = popt[1]


@Project.operation
# @Project.pre.isfile(pair_fit_file)
def run_tau(job):
    combinations = [['cation', 'anion']]
    for combo in combinations:
        if 'pairing_fit_a_matrix_{}_{}'.format(combo[0], combo[1]) in job.document:
            a = job.document['pairing_fit_a_matrix_{}_{}'.format(
                combo[0], combo[1])]
            b = job.document['pairing_fit_b_matrix_{}_{}'.format(
                combo[0], combo[1])]
            tau_pair = gamma(1 / a) * np.power(b, (-1 / a)) / a

            with open(os.path.join(job.workspace(), 'tau_{}_{}.txt'.format(combo[0], combo[1])), 'w') as f:
                f.write(str(tau_pair))
            print('saving')

            job.document['tau_pair_matrix_{}_{}'.format(
                combo[0], combo[1])] = tau_pair


@Project.operation
@Project.pre.isfile(msd_file)
def run_rdf(job):
    print('Loading trj {}'.format(job))
    if os.path.exists(os.path.join(job.workspace(), 'com.gro')):
        top_file = os.path.join(job.workspace(), 'com.gro')
        trj_file = os.path.join(job.workspace(), 'sample_com.xtc')
        trj = md.load(trj_file, top=top_file, stride=10)

        selections = dict()
        selections['cation'] = trj.topology.select('name li')
        selections['anion'] = trj.topology.select(
            'resname {}'.format(job.statepoint()['anion']))
        selections['acn'] = trj.topology.select('resname ch3cn')
        selections['chlor'] = trj.topology.select('resname chlor')
        selections['all'] = trj.topology.select('all')

        combos = [('cation', 'anion'),
                  ('cation', 'cation'),
                  ('anion', 'anion'),
                  ('acn', 'anion'),
                  ('acn', 'cation'),
                  ('chlor', 'anion'),
                  ('chlor', 'cation')]
        for combo in combos:
            fig, ax = plt.subplots()
            print('running rdf between {0} ({1}) and\t{2} ({3})\t...'.format(combo[0],
                                                                             len(
                                                                                 selections[combo[0]]),
                                                                             combo[1],
                                                                             len(selections[combo[1]])))
            r, g_r = md.compute_rdf(trj, pairs=trj.topology.select_pairs(
                selections[combo[0]], selections[combo[1]]), r_range=((0.0, 2.0)))

            data = np.vstack([r, g_r])
            np.savetxt(os.path.join(job.workspace(),
                                    'rdf-{}-{}.txt'.format(combo[0], combo[1])),
                       np.transpose(np.vstack([r, g_r])),
                       header='# r (nm)\tg(r)')
            ax.plot(r, g_r)
            plt.xlabel('r (nm)')
            plt.ylabel('G(r)')
            plt.savefig(os.path.join(job.workspace(),
                                     f'rdf-{combo[0]}-{combo[1]}.pdf'))
            print(' ... done\n')


@Project.operation
@Project.pre.isfile(msd_file)
def run_cond(job):
    if 'D_cation_bar_2' in job.document().keys():
        top_file = os.path.join(job.workspace(), 'sample.gro')
        trj_file = os.path.join(job.workspace(),
                                'sample_unwrapped.xtc')
        trj = md.load(trj_file, top=top_file)
        cation = trj.topology.select('name Li')
        cation_msd = job.document()['D_cation_bar_2']
        anion_msd = job.document()['D_anion_bar_2']
        volume = float(np.mean(trj.unitcell_volumes))*1e-27
        N = len(cation)
        T = job.sp['T']

        conductivity = calc_conductivity(N, volume, cation_msd, anion_msd, T=T)
        print(conductivity)
        job.document['ne_conductivity'] = conductivity
        print('Conductivity calculated')


@Project.operation
@Project.pre.isfile(msd_file)
def run_eh_cond(job):
    print(job.get_id())
    top_file = os.path.join(job.workspace(), 'com.gro')
    trj_file = os.path.join(job.workspace(), 'sample_com_unwrapped.xtc')
    trj_frame = md.load_frame(trj_file, top=top_file, index=0)

    trj_ion = trj_frame.atom_slice(trj_frame.top.select('resname li {}'.format(
        job.statepoint()['anion'])))
    charges = get_charges(trj_ion, job.statepoint()['anion'])
    new_charges = list()
    for charge in charges:
        if charge != 1:
            if charge > 0:
                charge = 1
            else:
                charge = -1
            new_charges.append(charge)

    chunk = 200
    running_avg = np.zeros(chunk)
    for i, trj in enumerate(md.iterload(trj_file, top=top_file, chunk=chunk, skip=100)):
        if i == 0:
            trj_time = trj.time
        if trj.n_frames != chunk:
            continue
        trj = trj.atom_slice(trj.top.select('resname li {}'.format(
            job.statepoint()['anion'])))
        M = dipole_moments_md(trj, new_charges)
        running_avg += [np.linalg.norm((M[i] - M[0]))
                        ** 2 for i in range(len(M))]

        x = (trj_time - trj_time[0]).reshape(-1)
        y = running_avg / i

    slope, intercept, r_value, p_value, std_error = stats.linregress(
        x, y)

    kB = 1.38e-23 * joule / kelvin
    V = np.mean(trj_frame.unitcell_volumes, axis=0) * nanometer ** 3
    T = job.statepoint()['T'] * kelvin

    sigma = slope * (elementary_charge * nanometer) ** 2 / \
        picosecond / (6 * V * kB * T)
    seimens = seconds ** 3 * ampere ** 2 / (kilogram * meter ** 2)
    sigma = sigma.in_units_of(seimens / meter)
    # print(sigma)
    # print(job.document()['ne_conductivity'])
    # print(job.document()['eh_conductivity'])

    job.document['eh_conductivity'] = sigma / sigma.unit


@Project.operation
@Project.pre.isfile(msd_file)
@Project.post.isfile(all_directs_file)
def run_directs(job):
    if job.get_id() in ['1ad289cbe7a639f71461aa6038f16f94', '509a76782f2eda70bfe5c3619485b689']:
        trj_file = os.path.join(job.workspace(), 'sample.xtc')
    else:
        trj_file = os.path.join(job.workspace(), 'sample.xtc')
    top_file = os.path.join(job.workspace(), 'init.gro')
    trj = md.load(trj_file, top=top_file)
    combinations = [['solvent', 'cation']]
    #               ['cation','anion']]
    #                ['anion', 'anion'],
    #                ['cation', 'cation'],
    #                ['solvent', 'solvent']] # ['ion','ion']]
    for combo in combinations:
        print('Loading trj {}'.format(job))
        anion = job.statepoint()['anion']
        cation = job.statepoint()['cation']
        if combo == ['solvent', 'solvent']:
            sliced = trj.topology.select(
                'not resname {} {}'.format(cation, anion))
            if job.sp['solvent'] == 'ch3cn':
                distance = 0.68
            else:
                distance = 0.48
        elif combo == ['cation', 'cation']:
            sliced = trj.topology.select(
                'resname {} {}'.format(cation, cation))
            distance = 0.43
        elif combo == ['anion', 'anion']:
            sliced = trj.topology.select('resname {} {}'.format(anion, anion))
            if job.sp['anion'] == 'tf2n':
                distance = 1.25
            else:
                distance = 0.8
        elif combo == ['cation', 'anion']:
            sliced = trj.topology.select('resname {} {}'.format(cation, anion))
            if job.sp['anion'] == 'tf2n':
                # distance = 0.55
                distance = {'li-li': 0.48, 'tf2n-tf2n': 1.25,
                            'li-tf2n': 0.55, 'tf2n-li': 0.55}
            else:
                # distance = 0.5
                distance = {'li-li': 0.48, 'fsi-fsi': 0.8,
                            'li-fsi': 0.5, 'fsi-li': 0.5}
        elif combo == ['solvent', 'cation']:
            sliced = trj.topology.select('not resname {}'.format(anion))
            if job.sp['solvent'] == 'ch3cn':
                distance = {'li-li': 0.48, 'li-ch3cn': 0.3,
                            'ch3cn-li': 0.3, 'ch3cn-ch3cn': 0.68}
            else:
                distance = {'li-li': 0.48, 'li-RES': 0.28,
                            'RES-li': 0.28, 'RES-RES': 0.45}
        # sliced = trj.topology.select('resname ch3cn')
        trj_slice = trj.atom_slice(sliced)
        trj_slice = trj_slice[:-1]
        index = trj_slice.n_frames / 16
        starts = np.empty(16)
        ends = np.empty(16)
        i = 0
        j = index
        for x in range(16):
            starts[x] = i
            ends[x] = j
            i += index
            j += index
        starts = [int(start) for start in starts]
        ends = [int(end) for end in ends]
        params = [trj_slice[i:j:10] for i, j in zip(starts, ends)]
        results = []

        with Pool() as pool:
            directs = pool.starmap(
                pairing.mult_frames_direct, zip(params, it.repeat(distance)))
        directs = np.asarray(directs)

        with open(os.path.join(job.workspace(), 'all-directs-{}-{}.pkl'.format(
                combo[0], combo[1])), 'wb') as f:
            pickle.dump(directs, f)

        with open(os.path.join(job.workspace(), 'all-directs-{}-{}.pkl'.format(
            combo[0], combo[1])), 'rb') as f_in, gzip.open(os.path.join(job.workspace(),
                                                                        'all-directs-{}-{}.pkl.gz'.format(combo[0], combo[1])), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


@Project.operation
@Project.pre.isfile(all_directs_file)
@Project.post.isfile(all_indirects_file)
def run_indirects(job):
    combinations = [['solvent', 'cation']]
    # combinations = [['cation','cation'],
    #                ['anion', 'anion'],
    #                ['cation', 'anion'],
    #                ['solvent', 'solvent']] # ['ion','ion']]
    print(job.get_id())
    for combo in combinations:
        with gzip.open(os.path.join(job.workspace(), 'all-directs-{}-{}.pkl.gz'.format(combo[0], combo[1])), 'rb') as f:
            direct = pickle.load(f)

        with Pool() as pool:
            indirects = pool.map(pairing.calc_indirect, direct)
            reducs = pool.map(pairing.calc_reduc, indirects)

        with open(os.path.join(job.workspace(), 'all-indirects-{}-{}.pkl'.format(combo[0], combo[1])), 'wb') as f:
            pickle.dump(indirects, f)

        with open(os.path.join(job.workspace(), 'all-indirects-{}-{}.pkl'.format(combo[0], combo[1])), 'rb') as f_in, gzip.open(os.path.join(job.workspace(), 'all-indirects-{}-{}.pkl.gz'.format(combo[0], combo[1])), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with open(os.path.join(job.workspace(), 'all-reducs-{}-{}.pkl'.format(combo[0], combo[1])), 'wb') as f:
            pickle.dump(reducs, f)


@Project.operation
# @Project.post.isfile(rho_file)
def run_rho(job):
    print('Loading trj {}'.format(job))
    top_file = os.path.join(job.workspace(), 'sample.gro')
    trj_file = os.path.join(job.workspace(), 'sample.xtc')
    trj = md.load(trj_file, top=top_file)

    # Compute density in kg * m ^ -3
    rho = calc_density(trj)

    job.document['rho'] = float(np.mean(rho))

    # Compute and store volume in nm ^ -3
    job.document['volume'] = float(np.mean(trj.unitcell_volumes))


@Project.operation
def set_charge_type(job):
    job.sp.setdefault('charge_type', 'am1bcc')


@Project.operation
@Project.pre.isfile(rdf_file)
@Project.post.isfile(cn_file)
def run_cn(job):
    combinations = [['solvent', 'solvent'],
                    ['cation', 'cation'],
                    ['anion', 'anion'],
                    ['solvent', 'cation'],
                    ['cation', 'anion'],
                    ['solvent', 'anion']]
    for combo in combinations:
        r, g_r = np.loadtxt(os.path.join(job.workspace(),
                                         'rdf-{}-{}.txt'.format(
            combo[0], combo[1]))).T
        # if combo == ['anion', 'anion']:
        if 'anion' in combo:
            if 'cation' in combo:
                if job.sp['anion'] == 'fsi':
                    chunk = np.where((r > 0.3) & (r < 0.8))
                else:
                    chunk = np.where((r > 0.45) & (r < 0.8))
            else:
                if job.sp['anion'] == 'fsi':
                    chunk = np.where((r > 0.5) & (r < 0.85))
                else:
                    chunk = np.where((r > 0.75) & (r < 1.3))
        elif combo == ['solvent', 'solvent']:
            if job.sp['solvent'] == 'spce':
                chunk = np.where((r > 0.3) & (r < 0.55))
            else:
                chunk = np.where((r > 0.4) & (r < 0.8))
        else:
            chunk = np.where((r > 0.3) & (r < 0.8))
        g_r_chunk = g_r[chunk]
        r_chunk = r[chunk]
        if combo == ['solvent', 'solvent']:
            rho = (200*job.sp['concentration']) / job.document['volume']
        elif combo == ['cation', 'cation'] or combo == ['anion', 'anion']:
            rho = (200) / job.document['volume']
        elif combo == ['solvent', 'cation'] or combo == ['solvent', 'anion']:
            rho = ((200*job.sp['concentration'])+200) / job.document['volume']
        elif combo == ['cation', 'anion']:
            rho = (400 / job.document['volume'])

        N = [np.trapz(4 * rho * np.pi * g_r[:i] * r[:i] ** 2, r[:i], r)
             for i in range(len(r))]

        # Store CN near r = 0.8
        index = np.where(g_r == np.amin(g_r_chunk))
        print('combo is {}'.format(combo))
        print('g_r is {}'.format(g_r[index]))
        print('r is {}'.format(r[index]))
        # index = r[maxr]
        # if combo == 'solvent-solvent':
        #    if job.sp['solvent'] == 'spce':
        #        index = np.argwhere(r > 0.31)[0]
        #    else:
        #        index = np.argwhere(r > 0.5)[0]
        # else:
        #    if job.sp['anion'] == 'fsi':
        #        index = np.argwhere(r > 0.4)[0]
        #    else:
        #        index = np.argwhere(r > 0.5)[0]

        # if combo == 'solvent-solvent':
        #    job.document['cn_solvent_solvent'] = N[int(index)]
        # elif combo == 'cation-anion':
        #    job.document['cn_cation_anion'] = N[int(index)]
        job.document['cn_{}_{}'.format(combo[0], combo[1])] = N[int(index[0])]

        # Save entire CN
        np.savetxt(os.path.join(job.workspace(),
                                'cn-{}-{}.txt'.format(combo[0], combo[1])),
                   np.transpose(np.vstack([r, N])),
                   header='# r (nm)\tCN(r)')


def _gromacs_str(op_name, gro_name, sys_name, job):
    """Helper function, returns grompp command string for operation """
    if op_name == 'em':
        mdp = signac.get_project().fn('src/util/mdp_files/{}.mdp'.format(op_name))
        cmd = (
            'gmx grompp -f {mdp} -c gaff.gro -p gaff.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    else:
        mdp = signac.get_project().fn(
            'src/util/mdp_files/{}-{}.mdp'.format(op_name, job.sp.T))
        cmd = (
            'gmx grompp -f {mdp} -c {gro}.gro -p gaff.top -o {op}.tpr --maxwarn 1 && gmx mdrun -deffnm {op} -ntmpi 1')
    return workspace_command(cmd.format(mdp=mdp, op=op_name, gro=gro_name, sys=sys_name))


def get_charges(trj, anion):
    charges = np.zeros(shape=(trj.n_atoms))

    for i, atom in enumerate(trj.top.atoms):
        if anion == 'fsi':
            if atom.name == 'fsi':
                charges[i] = -0.6
            elif atom.name == 'li':
                charges[i] = 0.6
        else:
            if atom.name == 'tf2n':
                charges[i] = -0.8
            elif atom.name == 'li':
                charges[i] = 0.8
    return charges


def dipole_moments_md(traj, charges):
    local_indices = np.array([(a.index, a.residue.atom(0).index)
                              for a in traj.top.atoms], dtype='int32')
    local_displacements = md.compute_displacements(
        traj, local_indices, periodic=False)

    molecule_indices = np.array([(a.residue.atom(0).index, 0)
                                 for a in traj.top.atoms], dtype='int32')
    molecule_displacements = md.compute_displacements(
        traj, molecule_indices, periodic=False)

    xyz = local_displacements + molecule_displacements

    moments = xyz.transpose(0, 2, 1).dot(charges)

    return moments


if __name__ == '__main__':
    Project().main()
