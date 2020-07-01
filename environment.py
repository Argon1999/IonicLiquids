"""Configuration of the project enviroment. 
 
The environments defined in this module can be auto-detected. 
This helps to define environment specific behaviour in heterogenous 
environments. 
""" 
import flow 
 
__all__ = ['CoriEnvironment']
 
class CoriEnvironment(flow.environment.SlurmEnvironment): 
    hostname_pattern = 'cori' 
    template = 'cori.sh'
    cores_per_node = 32  

    @classmethod
    def add_args(cls, parser):
        super(flow.environment.SlurmEnvironment, cls).add_args(parser)
        parser.add_argument(
               '-w', "--walltime", type=float, help="Walltime"
            )
