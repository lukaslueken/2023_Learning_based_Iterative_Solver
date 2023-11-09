"""
Model modified from:
https://doi.org/10.1016/j.sysconle.2007.06.013 Lazar et al. 2008
"""


import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete'

    model = do_mpc.model.Model(model_type, symvar_type)
    
    _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
    
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    A = np.array([[ 1,  1],
                  [ 0,  1]])

    B = np.array([[0.5],
                  [ 1 ]])

    F = np.array([[0.025],
                  [0.025]])

    Q = np.array([[0.8, 0],
                  [0, 0.8]])

    R = np.array([[0.1]])

    stage_cost = _x.T@Q@_x + _u.T@R@_u
    terminal_cost = _x.T@Q@_x
    model.set_expression(expr_name='stage_cost', expr=stage_cost)
    model.set_expression(expr_name='terminal_cost', expr=terminal_cost)

    x_next = A@_x + B@_u + F@_x.T@_x
    model.set_rhs('x', x_next)

    model.setup()

    return model
