import numpy as np
import xarray as xr
from scipy import constants as c
from scipy.interpolate import make_splrep
import warnings
from scipy.optimize import minimize

# Module for the base spectral mixture model

### Define the linear mixing spectral model ###
## Nonlinear optimization is performed with respect to a simple L^2 loss

class SpectralMixtureModel():
    # Goal of the model is to infer
    # 1) n_fire of T_i (temperatures)
    # 2) n_fire of p_i (fire area fractions)
    # 3) n_bkg of p_j (fire area fractions)

    # There is a need to remove the reflected component of the land surface
    # Either from direct sunlight or indirect illumination by the sky

    def __init__(
        self,
        n_fire,
        n_bkg,
        bkg_spectra_lis, # List of spectra of len n_bkg
        # All spectra should be spectral radiances following lambd
        SI=True, # Toggle bkg_spectra units from SI to AVIRIS standard
    ):
        # Design this class to be able to hold arbitrary number of endmembers
        # But for actual purposes, have only one to two end members
        self.n_fire = n_fire
        self.n_bkg = n_bkg
        
        if len(bkg_spectra_lis) != n_bkg:
            raise ValueError("n_bkg must be the same as the length of bkg_spectra_lis!")

        if SI:
            self.bkg_spectra_lis = bkg_spectra_lis
            bkg_spectra_lis_mod = bkg_spectra_lis
        else:
            print('Converting bkg_spectra to SI...')
            bkg_spectra_lis_mod = [(wv, spectra*1e7) for (wv, spectra) in bkg_spectra_lis]
            self.bkg_spectra_lis = bkg_spectra_lis_mod

        # Create spline functions for the spectra
        bkg_spectra_splines = []
        for lambd, spectra in bkg_spectra_lis_mod:
            spline = make_splrep(
                lambd,
                spectra,
                k=1,
                s=0
            )
            bkg_spectra_splines.append(spline)

        self.bkg_spectra_splines = bkg_spectra_splines
    
    def get_fire_spectra(self, lambd, T_tup):
        result_list = list()
        for T in T_tup:
            spectra = _planck(T, lambd)
            result_list.append(spectra)
        return result_list
    
    def get_bkg_spectra(self, lambd):
        result_list = list()
        for spline in self.bkg_spectra_splines:
            result_list.append(spline(lambd))
        return result_list

    def total_radiance(self, lambd, T_tup, T_fracs, bkg_fracs):
        # noise is given as an absolute radiance value for 1 std
        fire_spectra = self.get_fire_spectra(lambd, T_tup)
        bkg_spectra = self.get_bkg_spectra(lambd)

        result = np.zeros_like(lambd)
        for frac, spectra in zip(T_fracs, fire_spectra):
            result += frac * spectra

        for frac, spectra in zip(bkg_fracs, bkg_spectra):
            result += frac * spectra
        
        # Unit conversion
        # Output is in SI (W per m per m^2 per sr)
        # Want to convert to uW per nm per cm^2 per sr for consistency with AVIRIS
        # Multiply by 1e6 / (1e9 * 1e4) -> divide by 1e7
        return result*1e-7

def _planck(T, lambd):
    # Convert lambd from nanometres to metres
    lambd_ = lambd * 1e-9
    top = 2 * c.h * c.c**2
    bottom = lambd_**5 * (np.exp((c.h * c.c)/(lambd_ * c.k * T)) - 1)
    return top/bottom
def _planck(T, lambd):
    # Convert lambd from nanometres to metres
    lambd_ = lambd * 1e-9
    top = 2 * c.h * c.c**2
    bottom = lambd_**5 * (np.exp((c.h * c.c)/(lambd_ * c.k * T)) - 1)
    return top/bottom


### Function definitions for the parameter estimation ###

def return_loss_unconstrained(model, lambd, target):
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    # Designed for BFGS/unconstrained methods
    # The parameter vector is an ndarray of shape (2*n_fire + n_bkg-1,)
    # Arguments are organized in the following order
    # [T_i, p_i_fire, p*_j_bkg]
    # Note that in order to satisfy the p_i_fire + p_j_bkg = 1 constraint,
    # The last p_j_bkg parameter is omitted and calculated from 1 - all

    def loss(params):
        # Unpact the parameters and normalize them appropriately
        T_tup = tuple(params[:n_fire]*1000) # 1000 K scale
        T_fracs = tuple(params[n_fire:2*n_fire])
        # Enforcing land fraction constraint
        if len(params) >= 2*n_fire:
            bkg_fracs = tuple(params[2*n_fire:]) + (1 - np.sum(params[n_fire:]),)
        else:
            bkg_fracs = (1 - np.sum(params[n_fire:]),)

        prediction = model.total_radiance(lambd, T_tup, T_fracs, bkg_fracs)
        diff = target - prediction
        return np.sum(diff**2) # L2 norm

    return loss

def return_loss_constrained(model, lambd, target):
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    # Designed for Methods that accept separate constraints
    # The parameter vector is an ndarray of shape (2*n_fire + n_bkg,)

    def loss(params):
        # Unpact the parameters and normalize them appropriately
        T_tup = tuple(params[:n_fire]*1000) # 1000 K scale
        T_fracs = tuple(params[n_fire:2*n_fire])
        bkg_fracs = tuple(params[2*n_fire:])
        prediction = model.total_radiance(lambd, T_tup, T_fracs, bkg_fracs)
        diff = target - prediction
        return np.sum(diff**2) # L2 norm

    return loss

# Retrieve the parameters from the result
def retrieve_params_unconstrained(result, model):
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    params = result.x
    T_tup = params[:n_fire] * 1000
    T_frac = params[n_fire:2*n_fire]
    if len(params) >= 2*n_fire:
        bkg_fracs = tuple(params[2*n_fire:]) + (1 - np.sum(params[n_fire:]),)
    else:
        bkg_fracs = (1 - np.sum(params[n_fire:]),)
    print("The fire temperatures in K are: ", T_tup)
    print("The fire fractions are: ", T_frac)
    print("The background fractions are: ", bkg_fracs)
    return T_tup, T_frac, bkg_fracs

def retrieve_params_constrained(result, model):
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    params = result.x
    T_tup = params[:n_fire] * 1000
    T_frac = params[n_fire:2*n_fire]
    bkg_fracs = params[2*n_fire:]
    print("The fire temperatures in K are: ", T_tup)
    print("The fire fractions are: ", T_frac)
    print("The background fractions are: ", bkg_fracs)
    return T_tup, T_frac, bkg_fracs


# Simple wrapper function for the parameter estimation
def estimate_params_unconstrained(model, lambd, target, x0, method='L-BFGS-B'):
    loss_func = return_loss_unconstrained(model, lambd, target)
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    bounds = [(0, None)] * n_fire + [(0, 1)] * (n_bkg + n_fire - 1)
    # Check length of x0
    if len(x0) != (2*n_fire + n_bkg - 1):
        raise ValueError("Length of x0 needs to be (2*n_fire + n_bkg - 1)!")
        
    result = minimize(
        fun = loss_func,
        x0 = x0,
        bounds = bounds, # Bounds from land fractions
        method = method
    )
    T_tup, T_frac, bkg_fracs = retrieve_params_unconstrained(result, model)
    return result, [T_tup, T_frac, bkg_fracs]

# Simple wrapper function for the parameter estimation
def estimate_params_constrained(model, lambd, target, x0, method='SLSQP'):
    loss_func = return_loss_constrained(model, lambd, target)
    n_fire = model.n_fire
    n_bkg = model.n_bkg
    bounds = [(0, None)] * n_fire + [(0, 1)] * (n_bkg + n_fire)
    # Check length of x0
    if len(x0) != (2*n_fire + n_bkg):
        raise ValueError("Length of x0 needs to be (2*n_fire + n_bkg)!")
        
    result = minimize(
        fun = loss_func,
        x0 = x0,
        bounds = bounds, # Bounds from land fractions
        method = method,
        constraints = [{
            'type' : 'eq',
            'fun': lambda x: np.sum(x[n_fire:]) - 1,
        }],
    )
    T_tup, T_frac, bkg_fracs = retrieve_params_constrained(result, model)
    return result, [T_tup, T_frac, bkg_fracs]