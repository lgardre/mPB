"""
This module gives models to represent the initial reduced electric potential.
"""

import numpy
import sys
import types

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.mpb.uncalibrated_values



def phiInitGuess_OnePlateau_OneExponential(x: numpy.ndarray) -> numpy.ndarray:
    """
    Returns an initial guess for the the reduced electric potential modelled by a plateau and an exponential, and its numerical derivative.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the electric potential initial guess and its derivative are computed.
    
    Returns
    -------
    numpy.ndarray
        The array contaning the values of the electric potential initial guess (first column) and its derivative (second column).
    """
    
    phi_initGuess = numpy.zeros((2, x.size))

    for i in range(0, x.size):
        if x[i] <= LG_mPB_libs.mpb.uncalibrated_values.Z_HEADS:
            phi_initGuess[0][i] = -5
        else:
            phi_initGuess[0][i] = -5 * numpy.exp(-(x[i] - LG_mPB_libs.mpb.uncalibrated_values.Z_HEADS))

    phi_initGuess[1] = numpy.append([0], numpy.diff(phi_initGuess[0] / (x[1] - x[0])))
    
    return phi_initGuess


def phiInitGuess_TwoDecreasingExponentials_OnePlateau(x: numpy.ndarray) -> numpy.ndarray:
    """
    Returns an initial guess for the the reduced electric potential modelled by a plateau and two exponentials, and its numerical derivative.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the electric potential initial guess and its derivative are computed.
    
    Returns
    -------
    numpy.ndarray
        The array contaning the values of the electric potential initial guess (first column) and its derivative (second column).
    """
    
    phi_initGuess = numpy.zeros((2, x.size))

    for i in range(0, x.size):
        if x[i] <= LG_mPB_libs.mpb.uncalibrated_values.Z_TAILS:
            phi_initGuess[0][i] = -numpy.exp((x[i]) - (LG_mPB_libs.mpb.uncalibrated_values.Z_TAILS - numpy.log(5)))
        elif x[i] >= LG_mPB_libs.mpb.uncalibrated_values.Z_TAILS and x[i] <= LG_mPB_libs.mpb.uncalibrated_values.Z_HEADS:
            phi_initGuess[0][i] = -5
        elif x[i] > LG_mPB_libs.mpb.uncalibrated_values.Z_HEADS:
            phi_initGuess[0][i] = -5 * numpy.exp(-(x[i] - LG_mPB_libs.mpb.uncalibrated_values.Z_HEADS))

    phi_initGuess[1] = numpy.append([0], numpy.diff(phi_initGuess[0] / (x[1] - x[0])))
    
    return phi_initGuess


def phiInitGuess_Sinusoidal(x: numpy.ndarray) -> numpy.ndarray:
    """
    Returns an initial guess for the reduced electric potential modelled by a sinusoid, and its mathematical derivative.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the electric potential initial guess and its derivative are computed.
    
    Returns
    -------
    numpy.ndarray
        The array contaning the values of the electric potential initial guess (first column) and its derivative (second column).
    """
    
    phi_initGuess = numpy.zeros((2, x.size))
    phi_initGuess[0] = - numpy.sin((2 * numpy.pi) / x[-1] * x)
    phi_initGuess[1] = - ((2 * numpy.pi) / x[-1]) * numpy.cos((2 * numpy.pi) / x[-1] * x)
    
    return phi_initGuess
