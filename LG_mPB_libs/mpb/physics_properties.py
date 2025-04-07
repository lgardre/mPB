"""
This modules defines functions that compute physics properties of a given system.
"""

import numpy
import sys
import types

from enum import Enum
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.constants
import LG_mPB_libs.mpb.mpb_types



def beta(T: float) -> float:
    """
    Returns the thermodynamic beta defined as 1 over the Boltzmann constant times the temperature.
    
    Parameters
    ----------
    T : float
        The temperature in K.
    """
    
    return 1.0 / (LG_mPB_libs.constants.KB * T)

    
def W(x: numpy.ndarray, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], ionType: LG_mPB_libs.mpb.mpb_types.IonType, epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> numpy.ndarray:
    """
    Computes the Born energy.
    
    The Born energy represents the ion solvatation energy: the energy needed to make a hole of given radius and relative dielectric permittivity in the water.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the Born Energy is computed.
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    ionType : LG_mPB_libs.mpb.mpb_types.IonType
        Either a Cation or an Anion.
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the Born energy in kg⁻¹⋅nm⁻²⋅s⁻².
    
    Raises
    ------
    ValueError
        If the ion type is not anion nor cation.
    """
    
    rIon = 0.0
    if ionType == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
        rIon = params["rCation"]
    elif ionType == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
        rIon = params["rAnion"]
    else:
        raise ValueError("Ion type must be either an anion or a cation to determine its radius.")
    
    return LG_mPB_libs.constants.E**2 / (8.0 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO * rIon) * ((epsilon(x, params) if epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED else 1.0 / epsilon(x, params)) - (1.0 / params["epsilonR_bulkWater"]))


def rhoAnions(x: numpy.ndarray, T: float, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], phi: numpy.ndarray, epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> numpy.ndarray:
    """
    Computes the anionic charge density.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    T : float
        The temperature in K.
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    phi: numpy.ndarray
        The array containing the electric potential values in V⋅nm⁻¹.
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the anionic charge density in e⋅nm⁻³ (SI : 1.602⋅10⁻¹⁹ A⋅s⋅nm⁻³)
    """
    
    return - params["cSalt"] * numpy.exp(phi - beta(T) * W(x, epsilon, params, LG_mPB_libs.mpb.mpb_types.IonType.ANION, epsilonFormat))


def rhoCations(x: numpy.ndarray, T: float, epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray], params: dict[str, float], phi: numpy.ndarray, epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat) -> numpy.ndarray:
    """
    Computes the cationic charge density.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    T : float
        The temperature in K.
    epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    phi: numpy.ndarray
        The array containing the electric potential values in V⋅nm⁻¹.
    epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        Specifies the format of the water permittivity function: whether it returns the regular dielectric permittivity or the inverse of its perpendicular component.
    
    Returns
    -------
    numpy.ndarray
        The array containing the cationic charge density in e⋅nm⁻³ (SI : 1.602⋅10⁻¹⁹ A⋅s⋅nm⁻³)
    """
    
    return params["cSalt"] * numpy.exp(- phi - beta(T) * W(x, epsilon, params, LG_mPB_libs.mpb.mpb_types.IonType.CATION, epsilonFormat))
