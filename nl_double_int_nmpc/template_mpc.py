
import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc

def template_mpc(model, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 10
    mpc.settings.t_step = 1
    mpc.settings.store_full_solution =True

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = model.aux['terminal_cost']
    lterm = model.aux['stage_cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    max_x = np.array([[10.0], [10.0]])

    mpc.bounds['lower','_x','x'] = -max_x
    mpc.bounds['upper','_x','x'] =  max_x

    mpc.bounds['lower','_u','u'] = -2
    mpc.bounds['upper','_u','u'] =  2


    mpc.setup()

    return mpc
