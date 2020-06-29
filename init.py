import logging
import argparse

import signac


def main(args):
    project = signac.init_project('IonicLiquids')
    statepoints_init = []

    cations = ['emim', 'bmim', 'hmim']  # We'll probably add more
    anions = ['tfsi', 'fsi', 'nf', 'tf', 'tsac', 'tfsam', 'beti']
    cat_ffs = ['lopes', 'kpl', 'ngkpl']
    charge_scales = [1.0]

    for anion in anions:
        for cation in cations:
            if cation == 'emim':
                if anion == 'tfsi':
                    density = 1517
                    an_ffs = ['lopes_flour', 'lopes', 'kpl', 'ngkpl']
                elif anion == 'fsi':
                    density = 1455
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'nf':
                    density = 1576
                    an_ffs = ['lopes_flour']
                elif anion == 'tf':
                    density = 1387
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'tsac':
                    density = 1481
                    an_ffs = ['lopes_flour']
                elif anion == 'tfsam':
                    density = 1360
                    an_ffs = ['lopes_flour']
                elif anion == 'beti':
                    density = 1570
                    an_ffs = ['lopes_flour']
            if cation == 'bmim':
                if anion == 'tfsi':  # done
                    density = 1438
                    an_ffs = ['lopes_flour', 'lopes', 'kpl', 'ngkpl']
                elif anion == 'fsi':  # done
                    density = 1355
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'nf':
                    density = None
                    an_ffs = ['lopes_flour']
                elif anion == 'tf':  # done
                    density = 1296.6
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'tsac':
                    density = None
                    an_ffs = ['lopes_flour']
                elif anion == 'tfsam':
                    density = 1360
                    an_ffs = ['lopes_flour']
                elif anion == 'beti':  # done
                    density = 1510
                    an_ffs = ['lopes_flour']
            if cation == 'hmim':
                if anion == 'tfsi':  # done
                    density = 1360
                    an_ffs = ['lopes_flour', 'lopes', 'kpl', 'ngkpl']
                elif anion == 'fsi':  # guess
                    density = None
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'nf':
                    density = None
                    an_ffs = ['lopes_flour']
                elif anion == 'tf':  # done
                    density = 1234.9
                    an_ffs = ['lopes_flour', 'lopes']
                elif anion == 'tsac':
                    density = None
                    an_ffs = ['lopes_flour']
                elif anion == 'tfsam':
                    density = None
                    an_ffs = ['lopes_flour']
                elif anion == 'beti':  # done
                    density = 1420
                    an_ffs = ['lopes_flour']
            for cat_ff in cat_ffs:
                for an_ff in an_ffs:
                    for charge_scale in charge_scales:
                        statepoint = dict(
                            anion=anion,
                            cation=cation,
                            T=303,
                            n_anion=500,
                            n_cation=500,
                            cat_forcefield=cat_ff,
                            an_forcefield=an_ff,
                            density=density,
                            charge_scale=charge_scale
                        )
                    project.open_job(statepoint).init()
                    statepoints_init.append(statepoint)

    # Writing statepoints to hash table as a backup
    project.write_statepoints(statepoints_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Initialize the data space.")
    parser.add_argument(
        '-n', '--num-replicas',
        type=int,
        default=1,
        help="Initialize multiple replications.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
