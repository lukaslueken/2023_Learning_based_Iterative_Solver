"""
NLP-"Handler" Class
This class transforms the do_mpc NLP formulation to standard form and provides symbolic expressions and functions.
Furthermore, it provides the functions necessary to evaluate the KKT conditions, the Fischer-Burmeister function and the corresponding derivatives as well as Jacobian-vector products and vector-Jacobian products.
"""


# imports
import casadi as ca
import copy
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any


# NLP Handler Class
class NLPHandler():
    def __init__(self, mpc):
        self._get_do_mpc_nlp(mpc)
        self._remove_unused_sym_vars()
        self.transform_nlp_to_standard_full()
        self.setup_sym_expressions()

    def _get_do_mpc_nlp(self,mpc):
        """
        Warning:
            Not part of the public API.

        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': ca.vertcat(mpc.opt_x), 'f': mpc.nlp_obj, 'g': mpc.nlp_cons, 'p': ca.vertcat(mpc.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = mpc.nlp_cons_lb
        nlp_bounds['ubg'] = mpc.nlp_cons_ub
        nlp_bounds['lbx'] = ca.vertcat(mpc._lb_opt_x)
        nlp_bounds['ubx'] = ca.vertcat(mpc._ub_opt_x)
        # return nlp, nlp_bounds
        self.nlp, self.nlp_bounds = nlp, nlp_bounds

    def _detect_undetermined_sym_var(self, var: str ="x") -> Tuple[np.ndarray,np.ndarray]:         
        # symbolic expressions
        var_sym = self.nlp[var]        
        # objective function
        f_sym = self.nlp["f"]
        # constraints
        g_sym = self.nlp["g"]

        # boolean expressions on wether a symbolic is contained in the objective function f or the constraints g
        map_f_var = map(lambda x: ca.depends_on(f_sym,x),ca.vertsplit(var_sym))
        map_g_var = map(lambda x: ca.depends_on(g_sym,x),ca.vertsplit(var_sym))

        # combined boolean expressions as list for each symbolic variable in var_sym
        dep_list = [f_dep or g_dep for f_dep,g_dep in zip(map_f_var,map_g_var)]

        # indices of undetermined and determined symbolic variables
        undet_sym_idx = np.where(np.logical_not(dep_list))[0]
        det_sym_idx = np.where(dep_list)[0]

        return undet_sym_idx, det_sym_idx

    def _remove_unused_sym_vars(self):
        """
        Warning:
            Not part of the public API.

        Reduces the NLP by removing symbolic variables 
        for x and p that are not contained in the objective function or the constraints.

        """
        # detect undetermined symbolic variables
        undet_opt_x_idx, det_opt_x_idx = self._detect_undetermined_sym_var("x")
        undet_opt_p_idx, det_opt_p_idx = self._detect_undetermined_sym_var("p")
        
        # copy nlp and nlp_bounds
        nlp_red = self.nlp.copy()
        nlp_bounds_red = self.nlp_bounds.copy()

        # adapt nlp
        nlp_red["x"] = self.nlp["x"][det_opt_x_idx]
        nlp_red["p"] = self.nlp["p"][det_opt_p_idx]

        # adapt nlp_bounds
        nlp_bounds_red["lbx"] = self.nlp_bounds["lbx"][det_opt_x_idx]
        nlp_bounds_red["ubx"] = self.nlp_bounds["ubx"][det_opt_x_idx]

        det_sym_idx_dict = {"opt_x":det_opt_x_idx, "opt_p":det_opt_p_idx}
        undet_sym_idx_dict = {"opt_x":undet_opt_x_idx, "opt_p":undet_opt_p_idx}

        N_vars_to_remove = len(undet_sym_idx_dict["opt_x"])+len(undet_sym_idx_dict["opt_p"])
        if N_vars_to_remove > 0:
            self.nlp_unreduced, self.nlp_bounds_unreduced = self.nlp, self.nlp_bounds
            self.nlp, self.nlp_bounds = nlp_red, nlp_bounds_red
            self.det_sym_idx_dict, self.undet_sym_idx_dict = det_sym_idx_dict, undet_sym_idx_dict
            self.reduced_nlp = True
        else:
            self.reduced_nlp = False
            print("NLP formulation does not contain unused variables.")

    def transform_nlp_to_standard_full(self):
        """This transformation does not need any information about the current solution of the problem (e.g. active set).
        It only needs the bounds and the standard form of the problem.

        [g_nl,g_x] --> [g_nl_ubg,g_x_ubx,g_nl_lbg,g_x_lbx] + [h_nl, h_x]
        short: upper bounds before lower bounds; inequalities before equalities

        """
        # constraints:
        # 2x (ng+nx) (introduce inequality constraintes of form g(x,p)<=0)
        # 1x nh (introduce equality constraints of form h(x,p)=0)

        # 1.1 extract symbolic expressions
        x_sym = self.nlp['x']
        p_sym = self.nlp['p']
        f_sym = self.nlp['f']
        g_sym = self.nlp['g']

        # 1.2 extract bounds
        lbg = np.array(self.nlp_bounds['lbg'])
        ubg = np.array(self.nlp_bounds['ubg'])
        
        if "lbx" in self.nlp_bounds.keys():
            lbx = np.array(self.nlp_bounds['lbx'])

        if "ubx" in self.nlp_bounds.keys():
            ubx = np.array(self.nlp_bounds['ubx'])

        # 2. detect presence of equality constraints
        # 2.0 check wether linear state constraints are present
        lin_state_constraints_bool = ("lbx" in self.nlp_bounds.keys() and "ubx" in self.nlp_bounds.keys())

        # 2.1 nonlinear constraints (g_lb < g(x,p) < g_ub)
        g_equalities = (np.array(lbg) == np.array(ubg))
        zip_nl_cons = zip(ca.vertsplit(g_equalities), ca.vertsplit(g_sym), ca.vertsplit(ubg))
        nl_equality_sym = ca.vertcat(*[g_sym - lb for condition, g_sym, lb in zip_nl_cons if condition])
        nl_equality_positions =  list(np.argwhere(g_equalities)[:,0])


        # 2.2 state constraints (x_lb < x < x_ub)
        if lin_state_constraints_bool:
            x_equalities = (np.array(lbx) == np.array(ubx))
            zip_lin_cons = zip(ca.vertsplit(x_equalities), ca.vertsplit(x_sym), ca.vertsplit(ubx))
            lin_equality_sym = ca.vertcat(*[x_sym - lb for condition, x_sym, lb in zip_lin_cons if condition])
            lin_equality_positions =  list(np.argwhere(x_equalities)[:,0])

        # 2.3 create full equality constraint
        if lin_state_constraints_bool:
            equality_constraints_sym = ca.vertcat(nl_equality_sym, lin_equality_sym)
        else:
            equality_constraints_sym = nl_equality_sym

        # 3. create full inequality constraint
        g_inequalities = ~np.logical_or(g_equalities, np.logical_and(np.isinf(lbg), np.isinf(ubg))) # consider unconstrained
        nu_nl_inequality_positions =  list(np.argwhere(g_inequalities)[:,0])

        # 3.1 nonlinear constraints (g_lb <= g(x,p) <= g_ub)
        nl_inequality_lbg = []
        nl_inequality_ubg = []
        for idx, (lbg_el,ubg_el) in enumerate(zip(lbg,ubg)):
            if lbg_el==ubg_el:
                continue
            if not np.isinf(lbg_el):
                nl_inequality_lbg.append(idx)
            if not np.isinf(ubg_el):
                nl_inequality_ubg.append(idx)

        nl_inequality_ubg_sym = g_sym[nl_inequality_ubg] - ubg[nl_inequality_ubg]
        nl_inequality_lbg_sym = lbg[nl_inequality_lbg] - g_sym[nl_inequality_lbg]

        tmp_list = list(set(nl_inequality_ubg + nl_inequality_lbg))
        tmp_list.sort()
        assert nu_nl_inequality_positions == tmp_list
        # assert nu_nl_inequality_positions == list(set(nl_inequality_ubg + nl_inequality_lbg))
        
        # 3.2 state constraints (x_lb <= x <= x_ub)
        if lin_state_constraints_bool:
            x_inequalities = ~np.logical_or(x_equalities, np.logical_and(np.isinf(lbx), np.isinf(ubx))) # consider unconstrained
            nu_lin_inequality_positions =  list(np.argwhere(x_inequalities)[:,0])

            nu_lin_inequality_lbx = []
            nu_lin_inequality_ubx = []
            for idx, (lbx_el,ubx_el) in enumerate(zip(lbx,ubx)):
                if lbx_el==ubx_el:
                    continue
                if not np.isinf(lbx_el):
                    nu_lin_inequality_lbx.append(idx)
                if not np.isinf(ubx_el):
                    nu_lin_inequality_ubx.append(idx)

            lin_inequality_ubx_sym = x_sym[nu_lin_inequality_ubx] - ubx[nu_lin_inequality_ubx]
            lin_inequality_lbx_sym = lbx[nu_lin_inequality_lbx] - x_sym[nu_lin_inequality_lbx]

            assert nu_lin_inequality_positions == list(set(nu_lin_inequality_ubx + nu_lin_inequality_lbx))

        # 3.3 create full inequality constraint
        if lin_state_constraints_bool:
            inequality_constraints_sym = ca.vertcat(nl_inequality_ubg_sym, lin_inequality_ubx_sym, nl_inequality_lbg_sym, lin_inequality_lbx_sym)
        else:
            inequality_constraints_sym = ca.vertcat(nl_inequality_ubg_sym, nl_inequality_lbg_sym)

        # 4. create full nlp
        nlp_standard_full_dict = {"f":f_sym, "x":x_sym, "p":p_sym, "g":inequality_constraints_sym, "h":equality_constraints_sym}
        
        if lin_state_constraints_bool:
            transformation_positions = {"nl_eq_pos":nl_equality_positions, "lin_eq_pos":lin_equality_positions, "nl_ineq_pos_ubg":nl_inequality_ubg, "nl_ineq_pos_lbg":nl_inequality_lbg, "lin_ineq_pos_ubx":nu_lin_inequality_ubx, "lin_ineq_pos_lbx":nu_lin_inequality_lbx}
        else:
            transformation_positions = {"nl_eq_pos":nl_equality_positions, "nl_ineq_pos_ubg":nl_inequality_ubg, "nl_ineq_pos_lbg":nl_inequality_lbg}


        # 5. change bounds
        n_g = inequality_constraints_sym.shape[0]
        n_h = equality_constraints_sym.shape[0]

        # lbg_full_standard = np.concatenate((-np.inf*np.ones(n_g), np.zeros(n_h)),axis=0)
        # ubg_full_standard = np.concatenate((np.zeros(n_g), np.zeros(n_h)),axis=0)

        # nlp_standard_full_bounds = {"lbg":lbg_full_standard, "ubg":ubg_full_standard}

        print("NLP transformed: \n")
        print("[g_nl,g_x] --> [g_nl_ubg,g_x_ubx,g_nl_lbg,g_x_lbx] + [h_nl, h_x]")

        # return nlp_standard_full_dict, transformation_positions
        self.nlp_standard_full_dict, self.trans_indices = nlp_standard_full_dict, transformation_positions

    def extract_numeric_primal_dual_sol(self,nlp_sol):

        """function to extract numerical values of lagrange multipliers based on solution of original problem
        """

        n_g = nlp_sol["lam_g"].shape[0]
        n_x = nlp_sol["lam_x"].shape[0]
        
        lag_mul = ca.vertcat(nlp_sol["lam_g"],nlp_sol["lam_x"])
        
        eq_idx = copy.deepcopy(self.trans_indices["nl_eq_pos"])
        ub_idx = copy.deepcopy(self.trans_indices["nl_ineq_pos_ubg"])
        lb_idx = copy.deepcopy(self.trans_indices["nl_ineq_pos_lbg"])

        if "lin_eq_pos" in self.trans_indices.keys():
            if len(self.trans_indices["lin_eq_pos"])>0:
                eq_idx += [idx+n_g for idx in self.trans_indices["lin_eq_pos"]]
            if len(self.trans_indices["lin_ineq_pos_ubx"])>0:
                ub_idx += [idx+n_g for idx in self.trans_indices["lin_ineq_pos_ubx"]]
            if len(self.trans_indices["lin_ineq_pos_lbx"])>0:
                lb_idx += [idx+n_g for idx in self.trans_indices["lin_ineq_pos_lbx"]]
            
        nu = []
        lam_ub = []
        lam_lb = []

        for idx in range(n_g+n_x):
            
            if idx in eq_idx:
                nu.append(lag_mul[idx])

            if (idx in ub_idx) & (idx in lb_idx):
                if lag_mul[idx]<=0:
                    lam_lb.append(-lag_mul[idx]) # IMPORTANT: Transformation of Lagrange multipliers, sign reversel
                    lam_ub.append(0.0)
                elif lag_mul[idx]>0:
                    lam_ub.append(lag_mul[idx])
                    lam_lb.append(0.0)
                else:
                    raise ValueError("The extraction of the solution of the original problem failed.")
                continue

            if (idx in ub_idx) & (idx not in lb_idx):
                lam_ub.append(lag_mul[idx])
                continue

            if (idx not in ub_idx) & (idx in lb_idx):
                lam_lb.append(-lag_mul[idx])
                continue
        
        # IMPORTANT: Upper Bounds before Lower Bounds; inequalities before equalities
        return ca.vertcat(nlp_sol["x"],*lam_ub,*lam_lb,*nu)

    def get_mpc_sol(self,mpc):
        nlp_sol = {}
        nlp_sol["x"] = ca.vertcat(mpc.opt_x_num)
        nlp_sol["x_unscaled"] = ca.vertcat(mpc.opt_x_num_unscaled)
        nlp_sol["g"] = ca.vertcat(mpc.opt_g_num)
        nlp_sol["lam_g"] = ca.vertcat(mpc.lam_g_num)
        nlp_sol["lam_x"] = ca.vertcat(mpc.lam_x_num)
        p_num = ca.vertcat(mpc.opt_p_num)
        return nlp_sol, p_num

    def _reduce_nlp_solution_to_determined(self, nlp_sol, p_num): 
        assert self.reduced_nlp, "NLP is not reduced."
        # adapt nlp_sol
        nlp_sol_red = nlp_sol.copy()
        nlp_sol_red["x"] = nlp_sol["x"][self.det_sym_idx_dict["opt_x"]]
        nlp_sol_red["lam_x"] = nlp_sol["lam_x"][self.det_sym_idx_dict["opt_x"]] 
        p_num = p_num[self.det_sym_idx_dict["opt_p"]]

        return nlp_sol_red, p_num

    def get_reduced_primal_dual_sol(self,nlp_sol,p_num):
        nlp_sol_red, p_num = self._reduce_nlp_solution_to_determined(nlp_sol, p_num)
        z_num = self.extract_numeric_primal_dual_sol(nlp_sol_red)
        return z_num, p_num

    def setup_sym_expressions(self):
        x_sym = self.nlp_standard_full_dict["x"]
        f_sym = self.nlp_standard_full_dict["f"]
        p_sym = self.nlp_standard_full_dict["p"]
        g_sym = self.nlp_standard_full_dict["g"]
        h_sym = self.nlp_standard_full_dict["h"]

        n_x = x_sym.numel()
        n_g = g_sym.numel() # number of inequality constraints
        n_h = h_sym.numel() # number of equality constraints
        n_z = n_x + n_g + n_h
        n_p = p_sym.numel()

        self.n_x = n_x
        self.n_g = n_g
        self.n_h = n_h
        self.n_z = n_z
        self.n_p = n_p
        
        nu_sym = ca.SX.sym("nu_sym", n_h)
        lambda_sym = ca.SX.sym("lambda_sym", n_g)
        z_sym = ca.vertcat(x_sym,lambda_sym,nu_sym)

        self.nlp_standard_full_dict["nu"] = nu_sym
        self.nlp_standard_full_dict["lambda"] = lambda_sym
        self.nlp_standard_full_dict["z"] = z_sym

        # Lagrangian
        L_sym = f_sym + lambda_sym.T @ g_sym + nu_sym.T @ h_sym
        self.L_func = ca.Function("L",[z_sym,p_sym],[L_sym],["z","p"],["L"])

        # Stationarity
        dL_dx_sym = ca.gradient(L_sym,x_sym)
        self.dL_dx_func = ca.Function("dL_dx",[z_sym,p_sym],[dL_dx_sym],["z","p"],["dL_dx"])

        # Constraints
        self.h_func = ca.Function("h",[x_sym,p_sym],[h_sym],["z","p"],["dL_dnu"])
        self.g_func = ca.Function("g",[x_sym,p_sym],[g_sym],["z","p"],["g"])


        # Jacobians
        df_dx = ca.jacobian(f_sym,x_sym)
        dh_dx = ca.jacobian(h_sym,x_sym)
        dg_dx = ca.jacobian(g_sym,x_sym)
        self.df_dx_func = ca.Function("df_dx",[x_sym,p_sym],[df_dx],["z","p"],["df_dx"])
        self.dh_dx_func = ca.Function("dh_dx",[x_sym,p_sym],[dh_dx],["z","p"],["dh_dx"])
        self.dg_dx_func = ca.Function("dg_dx",[x_sym,p_sym],[dg_dx],["z","p"],["dg_dx"])

        # Bound Functions (primal and dual feasibility of inequalities)
        g_bound_sym = ca.fmax(0,g_sym)  # g<=0 --> lambda>=0
        self.g_bound_func = ca.Function("g_tilde",[z_sym,p_sym],[g_bound_sym],["z","p"],["dL_dlambda"])

        lambda_bound_sym = ca.fmin(0,lambda_sym)
        self.lambda_bound_func = ca.Function("lambda_tilde",[z_sym,p_sym],[lambda_bound_sym],["z","p"],["lambda_bound"])

        # Complementary Slackness
        comp_slack_sym = lambda_sym*g_sym
        self.comp_slack_func = ca.Function("comp_slack",[z_sym,p_sym],[comp_slack_sym],["z","p"],["comp_slack"])

        # KKT Conditions
        KKT_sym = ca.vertcat(dL_dx_sym,h_sym,g_bound_sym,lambda_bound_sym,comp_slack_sym)
        self.KKT_func = ca.Function("KKT",[z_sym,p_sym],[KKT_sym],["z","p"],["KKT"])

        #### FISCHER-BURMEISTER FUNCTION ####
        eps_sym = ca.SX.sym("eps_sym",1)
        fb_sym = lambda_sym - g_sym - ca.sqrt(lambda_sym**2 + g_sym**2 + eps_sym**2)
        
        # Dfb_Dlambda_sym = ca.jacobian(fb_sym,lambda_sym)
        # self.Dfb_Dlambda_func = ca.Function("Dfb_Dlambda",[z_sym,p_sym,eps_sym],[Dfb_Dlambda_sym],["z","p","eps"],["Dfb_Dlambda"])

        self.fb_func = ca.Function("fb",[z_sym,p_sym,eps_sym],[fb_sym],["z","p","eps"],["fb"])

        F_FB_sym = ca.vertcat(dL_dx_sym,h_sym,fb_sym)
        self.F_FB_func = ca.Function("F_FB",[z_sym,p_sym,eps_sym],[F_FB_sym],["z","p","eps"],["F_FB"])

        #### FISCHER-BURMEISTER Derivatives ####
        # DF_FB_sym = ca.jacobian(F_FB_sym,z_sym)
        # self.DF_FB_func = ca.Function("DF_FB",[z_sym,p_sym,eps_sym],[DF_FB_sym],["z","p","eps"],["DF_FB"])
       
        # VJP/JVP Vector
        v_sym = ca.SX.sym("v_sym",n_z)

        # JVP
        jvp_sym = ca.jtimes(F_FB_sym,z_sym,v_sym,False)
        self.jvp_func = ca.Function("jvp",[z_sym,p_sym,eps_sym,v_sym],[jvp_sym],["z","p","eps","v"],["jvp"])

        # VJP
        vjp_sym = ca.jtimes(F_FB_sym,z_sym,v_sym,True)
        self.vjp_func = ca.Function("vjp",[z_sym,p_sym,eps_sym,v_sym],[vjp_sym],["z","p","eps","v"],["vjp"])
        
        dz_sym = ca.SX.sym("dz_sym",n_z) # Solver Step
        V_k_sym = 0.5 * F_FB_sym.T @ F_FB_sym # Loss at iteration k
        g_k_sym = ca.jtimes(F_FB_sym,z_sym,dz_sym,False) # JVP of Gradient of F_FB and Step
        F_tilde_sym = F_FB_sym + g_k_sym # (= Fk+DFk*dzk) F_tilde at iteration k = Linear Prediction of KKT Error at iteration k+1
        V_tilde_sym = 0.5 * F_tilde_sym.T @ F_tilde_sym # ||Fk+DFk*dzk|| - Norm: Linear Prediction of Loss at iteration k+1
        
        d_k_sym = ca.jtimes(F_FB_sym,z_sym,F_tilde_sym,True) # (= F_tilde*DF) VJP of Gradient of F_FB and Linear Prediction of KKT Error at iteration k+1
        
        self.F_tilde_func = ca.Function("F_tilde",[z_sym,p_sym,eps_sym,dz_sym],[F_tilde_sym],["z","p","eps","dz"],["F_tilde"])
        self.V_k_func = ca.Function("V_k",[z_sym,p_sym,eps_sym],[V_k_sym],["z","p","eps"],["V_k"])
        self.V_tilde_func = ca.Function("V_tilde",[z_sym,p_sym,eps_sym,dz_sym],[V_tilde_sym],["z","p","eps","dz"],["V_tilde"])
        self.d_k_func = ca.Function("d_k",[z_sym,p_sym,eps_sym,dz_sym],[d_k_sym],["z","p","eps","dz"],["d_k"])
        self.g_k_func = ca.Function("g_k",[z_sym,p_sym,eps_sym,dz_sym],[g_k_sym],["z","p","eps","dz"],["g_k"])


    def setup_batch_functions(self,N,N_threads=1):
        """
        Description:
        This function maps the "normal" casadi functions to batched versions using N_threads threads in parallel.
        Caution! Choosing to many threads can lead to cpu overload.
        """
        self.vjp_batch_func = self.vjp_func.map(N,"thread",N_threads)
        
        self.KKT_batch_func = self.KKT_func.map(N,"thread",N_threads)
        self.F_FB_batch_func = self.F_FB_func.map(N,"thread",N_threads)
        self.F_tilde_batch_func = self.F_tilde_func.map(N,"thread",N_threads)

        self.V_k_batch_func = self.V_k_func.map(N,"thread",N_threads)
        self.V_tilde_batch_func = self.V_tilde_func.map(N,"thread",N_threads)

        self.d_k_batch_func = self.d_k_func.map(N,"thread",N_threads)

        self.DF_FB_batch_func = self.DF_FB_func.map(N,"thread",N_threads)



