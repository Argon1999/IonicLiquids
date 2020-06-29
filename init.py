import logging
import argparse

import signac


def main(args):
    project = signac.init_project('IonicLiquids')
    statepoints_init = []

    temperatures = [303]
    cations = ['emim', 'bmim', 'hmim'] #We'll probably add more
    anions = ['tfsi','fsi','nf','tf','tsac','tfsam', 'beti']
    forcefields = ['lopes_flour', 'lopes', 'kpl','ngkpl']
    charge_scales = [1.0]

    for anion in anions:
        for cation in cations:
             if anion == 'tfsi':
                density = 1517
            elif anion == 'fsi':
                density = 1455
            elif anion == 'nf':
                density = 1576
            elif anion == 'tf':
                density = 1387
            elif anion == 'tsac':
                density = 1481
            elif anion == 'tfsam':
                density = 1360
            elif anion == 'beti':
                density = 1570


            for ff in forcefields:
                for charge_scale in charge_scales:
                    statepoint = dict(
                                anion=anion,
                                cation =cation,
                                T= 303,
                                n_anion = 500,
                                n_cation = 500,
                                forcefield = ff
                                density = density
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
