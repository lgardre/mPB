import numpy
import sys
import types

from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.constants
import LG_mPB_libs.mpb.mpb_types
import LG_mPB_libs.mpb.physics_properties



def bjerrumLength(x: numpy.ndarray, T: float, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> numpy.ndarray:
    """
    Computes the Bjerrum length in nm.
    
    The Bjerrum length is the distance at which the electrostatic interaction energy is equals to the thermal energy.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the Bjerrum Length is computed.
    T : float
        Temperature of the system in K.
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the Bjerrum length in nm.
    """

    bjerrumLengthTimesEpsilonR = LG_mPB_libs.constants.E**2 * LG_mPB_libs.mpb.physics_properties.beta(T) / (4.0 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO)
    
    return bjerrumLengthTimesEpsilonR / epsilon(x, params) if epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR else bjerrumLengthTimesEpsilonR * epsilon(x, params)


def debyeLength(x: numpy.ndarray, T: float, cSalt: float, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> float:
    """
    Computes the Debye length in nm.
    
    The Debye length is the range of the electric potential screening.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the Bjerrum Length is computed.
    T : float
        Temperature of the system in K.
    cSalt : float
        Salt concentration in the water in nm⁻³.
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the Debye length in nm.
    """
    
    return 1.0 / numpy.sqrt(8.0 * numpy.pi * bjerrumLength(x, T, epsilon, params, epsilonFormat) * cSalt)


def gouyChapmanLength(x: numpy.ndarray, T: float, sigma: float, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> float:
    """
    Computes the Gouy-Chapman length in nm.
    
    The Gouy-Chapman length is the length at which the electrostatic interaction energy between an ion and a charged surface is equals to the thermal energy.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the Bjerrum Length is computed.
    T : float
        Temperature of the system in K.
    sigma : float
        Surface charge in C⋅nm⁻² (SI : A⋅s⋅nm⁻²).
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the Gouy-Chapman length in nm.
    """
    
    return LG_mPB_libs.constants.E / (2.0 * numpy.pi * bjerrumLength(x, T, epsilon, params, epsilonFormat) * numpy.abs(sigma))
