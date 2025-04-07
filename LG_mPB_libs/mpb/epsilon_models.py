"""
This module gives models to represent the dielectric permittivity, the perpendicular component of its inverse, and the associated derivatives.
"""

import numpy
import re
import types



###############################
# BUILT-IN MODELS FOR EPSILON #
###############################

#-------------#
# One Sigmoid #
#-------------#

def epsilon_Sigmoidal(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the water dielectric permittivity with a sigmoid.
    
    Model parameters:
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        z1: Abscissa of the heads/tails interface. In practice, this is the abscissa of the sigmoid inflection point.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        A dictionnary containing all the parameters for the function epsilon().
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return params["epsilonR_tails"] + (params["epsilonR_plateau"] - params["epsilonR_tails"]) / (1.0 + numpy.exp(- params["K"] * (z - params["z1"])))


def epsilonDerivative_Sigmoidal(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the water dielectric permittivity curve modelled with a sigmoid.
    
    Model parameters:
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        z1: Abscissa of the heads/tails interface. In practice, this is the abscissa of the sigmoid inflection point.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return params["K"] * (params["epsilonR_plateau"] - params["epsilonR_tails"]) * numpy.exp(- params["K"] * (z - params["z1"])) / (1.0 + numpy.exp(- params["K"] * (z - params["z1"])))**2


#-------------------------------------------------#
# One Generalized Logistic Function, One Gaussian #
#-------------------------------------------------#

def epsilon_OneGeneralizedLogistic_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the water dielectric permittivity with a generalized logistic function and a 3σ-gaussian (99.7% of the values are in the interval).
    
    Model parameters:
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
        z1: Parameter of the generalized logistic function (controls the horizontal offset).
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        K: Parameter of the generalized logistic function (controls the slope).
        nu: Parameter of the generalized logistic function (controls the asymmetry).
        C: Parameter of the gaussian (controls its maxima).
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (params["epsilonR_tails"]
            + (params["epsilonR_plateau"] - params["epsilonR_tails"]) / (1.0 + numpy.exp(- params["K"] * (z - params["z1"])))**(1.0 / params["nu"])
            + params["C"] / ((params["z3"] - params["z2"]) / 6.0 * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2)))


def epsilonDerivative_OneGeneralizedLogistic_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the water dielectric permittivity curve modelled with a generalized logistic function and a 3σ-gaussian (99.7% of the values are in the interval).
    
    Model parameters:
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
        z1: Parameter of the generalized logistic function (controls the horizontal offset).
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        K: Parameter of the generalized logistic function (controls the slope).
        nu: Parameter of the generalized logistic function (controls the asymmetry).
        C: Parameter of the gaussian (controls its maxima).
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return (params["K"] * (params["epsilonR_plateau"] - params["epsilonR_tails"]) * numpy.exp(- params["K"] * (z - params["z1"])) / params["nu"] * (1.0 + numpy.exp(- params["K"] * (z - params["z1"])))**(- (1.0 / params["nu"] + 1.0))
            + (params["C"] * (params["z2"] + params["z3"] - 2.0 * z) / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**3 * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2)))


#---------------------------#
# Discrete Cosine Transform #
#---------------------------#

def epsilon_DiscreteCosineTransform(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the water dielectric permittivity with a Cosine Series using the Discrete Cosine Transform.
    
    Model parameters:
        dct0: First coefficient of the Discrete Cosine Transform
        dct1: Second coefficient of the Discrete Cosine Transform
        ...
        dctN: Nth coefficient of the Discrete Cosine Transform, with N being strictly inferior to the length of the z array.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.

    Raises
    ------
    IndexError
        If the index of a Discrete Cosine Transform coefficient is not between 0 and len(z).
    """

    N = len(z)
    epsilon = []
    
    pattern = re.compile("^dct([0-9]+)$")
    dctCoefficients = numpy.zeros(N)

    for param in params:
        match = pattern.match(param)
        if match:
            index = int(match.group(1))
            
            if index < 0 or index >= N:
                raise IndexError(f"The parameter '{param}' has an incorrect index (it should be between 0 and {N} in this case).")
            
            dctCoefficients[index] = params[param]
    
    for k in range(0, N):
        s = 0.0
        
        for n in range(1, N):
            s += dctCoefficients[n] * numpy.cos((2 * k + 1) * n * numpy.pi / (2 * N))
        
        s *= 2
        s += dctCoefficients[0]
        s /= (2 * N)
        
        epsilon.append(s)
        
    return numpy.asarray(epsilon)

def epsilonDerivative_DiscreteCosineTransform(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a Cosine Series using the Discrete Cosine Transform.
    
    Model parameters:
        dct0: First coefficient of the Discrete Cosine Transform
        dct1: Second coefficient of the Discrete Cosine Transform
        ...
        dctN: Nth coefficient of the Discrete Cosine Transform, with N being strictly inferior to the length of the z array.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    N = len(z)
    
    pattern = re.compile("^dct([0-9]+)$")
    dctCoefficients = numpy.zeros(N)

    for param in params:
        match = pattern.match(param)
        if match:
            index = int(match.group(1))
            
            if index < 0 or index >= N:
                raise IndexError(f"The parameter '{param}' has an incorrect index (it should be between 0 and {N} in this case).")
            
            dctCoefficients[index] = params[param]
    
    # We first rebuild epsilon...
    epsilon = []
    
    for k in range(0, N):
        s = 0.0
        
        for n in range(1, N):
            s += dctCoefficients[n] * numpy.cos((2 * k + 1) * n * numpy.pi / (2 * N))
        
        s *= 2
        s += dctCoefficients[0]
        s /= (2 * N)
        
        epsilon.append(s)
    
    # ...because we compute its numerical derivative
    epsilonDerivative = numpy.ndarray( ( 0, ) )
    epsilonDerivative = numpy.append(epsilonDerivative, 0.0)
    
    for i in range(1, N - 1):
        dz_backard = z[i - 1] - z[i]
        dz_forward = z[i] - z[i + 1]
        epsilonDerivative = numpy.append(epsilonDerivative, (epsilon[i + 1] - epsilon[i - 1]) / (0.5 * (dz_backard + dz_forward)))
        
    epsilonDerivative = numpy.append(epsilonDerivative, 0.0)
    
    return epsilonDerivative
    


#############################################################################
# BUILT-IN MODELS FOR THE INVERSE OF THE PERPENDICULAR COMPONENT OF EPSILON #
#############################################################################

#---------------------------#
# One Sigmoid, One Gaussian #
#---------------------------#

def epsilonPerpInv_Sigmoidal(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and a 3σ-gaussian (99.7% of the values are in the interval).
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (1.0 / params["epsilonR_plateau"] + (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) / (1.0 + numpy.exp(params["K"] * (z - params["z1"]))))


def epsilonPerpInvDerivative_Sigmoidal(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and a gaussian.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return ((1.0 / params["epsilonR_plateau"] - 1.0 / params["epsilonR_tails"]) * params["K"] * numpy.exp(params["K"] * (z - params["z1"])) / ((1.0 + numpy.exp(params["K"] * (z - params["z1"])))**2))


#---------------------------#
# One Sigmoid, One Gaussian #
#---------------------------#

def epsilonPerpInv_OneSigmoid_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and a 3σ-gaussian (99.7% of the values are in the interval).
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (1.0 / params["epsilonR_plateau"]
            + (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) / (1.0 + numpy.exp(params["K"] * (z - params["z1"])))
            + params["C"] / ((params["z3"] - params["z2"]) / 6.0 * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2)))


def epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and a gaussian.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return ((1.0 / params["epsilonR_plateau"] - 1.0 / params["epsilonR_tails"]) * params["K"] * numpy.exp(params["K"] * (z - params["z1"])) / ((1.0 + numpy.exp(params["K"] * (z - params["z1"])))**2)
           + (params["C"] * (params["z2"] + params["z3"] - 2.0 * z) / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**3 * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2)))


def epsilonPerpInvPartialDerivatives_OneSigmoid_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> dict[str, numpy.ndarray]:
    """
    Returns the error on the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and a gaussian.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    dict[str, numpy.ndarray]
        The dictionnary containing the partial derivatives of epsilon for each model parameter.
    """
    
    partialDerivatives = {}
    
    partialDerivatives["z1"] = (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) * params["K"] * numpy.exp(params["K"] * (z - params["z1"])) / ((1.0 + numpy.exp(params["K"] * (z - params["z1"])))**2)
    
    partialDerivatives["z2"] = 6.0 * params["C"] / (numpy.sqrt(2.0 * numpy.pi) * (params["z3"] - params["z2"])**2) * numpy.exp(- 18.0 * ((z - (params["z2"] + params["z3"]) / 2.0) / (params["z3"] - params["z2"]))**2) * (1.0 - 36.0 * (z - (params["z2"] + params["z3"]) / 2.0) * (z - params["z3"]) / ((params["z3"] - params["z2"])**2))
    
    partialDerivatives["z3"] = - 6.0 * params["C"] / (numpy.sqrt(2.0 * numpy.pi) * (params["z3"] - params["z2"])**2) * numpy.exp(- 18.0 * ((z - (params["z2"] + params["z3"]) / 2.0) / (params["z3"] - params["z2"]))**2) * (1.0 + 36.0 * (z - (params["z2"] + params["z3"]) / 2.0) * (params["z2"] - z) / ((params["z3"] - params["z2"])**2))
    
    partialDerivatives["C"] = 6.0 / (numpy.sqrt(2.0 * numpy.pi) * (params["z3"] - params["z2"])) * numpy.exp(- 18.0 * ((z - (params["z2"] + params["z3"]) / 2.0) / (params["z3"] - params["z2"]))**2)
    
    partialDerivatives["K"] = - (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) * (z - params["z1"]) * numpy.exp(params["K"] * (z - params["z1"])) / ((1.0 + numpy.exp(params["K"] * (z - params["z1"])))**2)
    
    partialDerivatives["epsilonR_tails"] = - 1.0 / ((params["epsilonR_tails"]**2) * (1.0 + numpy.exp(params["K"] * (z - params["z1"]))))
    
    partialDerivatives["epsilonR_plateau"] = 1.0 / (params["epsilonR_plateau"]**2) * (1.0 / (1.0 + numpy.exp(params["K"] * (z - params["z1"]))) - 1.0)
    
    return partialDerivatives


#----------------------------#
# One Sigmoid, Two Gaussians #
#----------------------------#

def epsilonPerpInv_OneSigmoid_TwoGaussians(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and two 3σ-gaussians (99.7% of the values are in the interval).
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the first gaussian interval.
        z3: Upper limit of the first gaussian interval.
        z4: Lower limit of the second gaussian interval.
        z5: Upper limit of the second gaussian interval.
        C1: Parameter of the first gaussian (controls its maxima).
        C2: Parameter of the second gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (1.0 / params["epsilonR_plateau"]
            + (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) / (1.0 + numpy.exp(params["K"] * (z - params["z1"])))
            + params["C1"] / ((params["z3"] - params["z2"]) / 6.0 * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2))
            + params["C2"] / ((params["z5"] - params["z4"]) / 6.0 * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (z - (params["z4"] + params["z5"]) / 2.0)**2 / (2.0 * ((params["z5"] - params["z4"]) / 6.0)**2)))


def epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and two gaussians.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the first gaussian interval.
        z3: Upper limit of the first gaussian interval.
        z4: Lower limit of the second gaussian interval.
        z5: Upper limit of the second gaussian interval.
        C1: Parameter of the first gaussian (controls its maxima).
        C2: Parameter of the second gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return ((1.0 / params["epsilonR_plateau"] - 1.0 / params["epsilonR_tails"]) * params["K"] * numpy.exp(params["K"] * (z - params["z1"])) / ((1.0 + numpy.exp(params["K"] * (z - params["z1"])))**2)
           + (params["C1"] * (params["z2"] + params["z3"] - 2.0 * z) / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**3 * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(- (z - (params["z2"] + params["z3"]) / 2.0)**2 / (2.0 * ((params["z3"] - params["z2"]) / 6.0)**2))
           + (params["C2"] * (params["z4"] + params["z5"] - 2.0 * z) / (2.0 * ((params["z5"] - params["z4"]) / 6.0)**3 * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(- (z - (params["z4"] + params["z5"]) / 2.0)**2 / (2.0 * ((params["z5"] - params["z4"]) / 6.0)**2)))


#--------------#
# Two Sigmoids #
#--------------#

def epsilonPerpInv_TwoSigmoids(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with two sigmoids.
    
    Model parameters:
        z1: Abscissa of the first sigmoid inflection point.
        z2: Abscissa of the second sigmoid inflection point.
        C: Maximum value of the second sigmoid, in ratio of the first sigmoid maximum value (must be between 0 and 1).
        K1: Parameter of the first sigmoid (slope = K/4 at the inflection point).
        K2: Parameter of the second sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (1.0 / params["epsilonR_plateau"]
            + (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) * ((1.0 - params["C"]) / (1.0 + numpy.exp(params["K1"] * (z - params["z1"])))
                                                                                   + params["C"] / (1.0 + numpy.exp(params["K2"] * (z - params["z2"])))))
def epsilonPerpInvDerivative_TwoSigmoids(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with two sigmoids.
    
    Model parameters:
        z1: Abscissa of the first sigmoid inflection point.
        z2: Abscissa of the second sigmoid inflection point.
        C: Maximum value of the second sigmoid, in ratio of the first sigmoid maximum value (must be between 0 and 1).
        K1: Parameter of the first sigmoid (slope = K/4 at the inflection point).
        K2: Parameter of the second sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return ((1.0 / params["epsilonR_plateau"] - 1.0 / params["epsilonR_tails"]) * (((1.0 - params["C"]) * params["K1"] * numpy.exp(params["K1"] * (z - params["z1"]))) / ((1 + numpy.exp(params["K1"] * (z - params["z1"])))**2)
                                                                                 + (params["C"] * params["K2"] * numpy.exp(params["K2"] * (z - params["z2"]))) / ((1 + numpy.exp(params["K2"] * (z - params["z2"])))**2)))


#----------------------------#
# Two Sigmoids, One Gaussian #
#----------------------------#

def epsilonPerpInv_TwoSigmoids_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with two sigmoids and a 3σ-gaussian (99.7% of the values are in the interval).
    
    Model parameters:
        z1: Abscissa of the first sigmoid inflection point.
        z2: Abscissa of the second sigmoid inflection point.
        z3: Lower limit of the gaussian interval.
        z4: Upper limit of the gaussian interval.
        C1: Maximum value of the second sigmoid, in ratio of the first sigmoid maximum value (must be between 0 and 1).
        C2: Parameter of the gaussian (controls its maxima).
        K1: Parameter of the first sigmoid (slope = K/4 at the inflection point).
        K2: Parameter of the second sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    return (1.0 / params["epsilonR_plateau"]
            + (1.0 / params["epsilonR_tails"] - 1.0 / params["epsilonR_plateau"]) * ((1.0 - params["C1"]) / (1.0 + numpy.exp(params["K1"] * (z - params["z1"])))
                                                                                   + params["C1"] / (1.0 + numpy.exp(params["K2"] * (z - params["z2"]))))
            + params["C2"] / ((params["z4"] - params["z3"]) / 6.0 * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (z - (params["z3"] + params["z4"]) / 2.0)**2 / (2.0 * ((params["z4"] - params["z3"]) / 6.0)**2)))

def epsilonPerpInvDerivative_TwoSigmoids_OneGaussian(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with two sigmoids and one gaussian.
    
    Model parameters:
        z1: Abscissa of the first sigmoid inflection point.
        z2: Abscissa of the second sigmoid inflection point.
        z3: Lower limit of the gaussian interval.
        z4: Upper limit of the gaussian interval.
        C1: Maximum value of the second sigmoid, in ratio of the first sigmoid maximum value (must be between 0 and 1).
        C2: Parameter of the gaussian (controls its maxima).
        K1: Parameter of the first sigmoid (slope = K/4 at the inflection point).
        K2: Parameter of the second sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    return ((1.0 / params["epsilonR_plateau"] - 1.0 / params["epsilonR_tails"]) * (((1.0 - params["C1"]) * params["K1"] * numpy.exp(params["K1"] * (z - params["z1"]))) / ((1 + numpy.exp(params["K1"] * (z - params["z1"])))**2)
                                                                                 + (params["C1"] * params["K2"] * numpy.exp(params["K2"] * (z - params["z2"]))) / ((1 + numpy.exp(params["K2"] * (z - params["z2"])))**2))
            + (params["C2"] * (params["z3"] + params["z4"] - 2.0 * z) / (2.0 * ((params["z4"] - params["z3"]) / 6.0)**3 * numpy.sqrt(2.0 * numpy.pi))) * numpy.exp(- (z - (params["z3"] + params["z4"]) / 2.0)**2 / (2.0 * ((params["z4"] - params["z3"]) / 6.0)**2)))



#############################################################################################
# BUILT-IN MODELS FOR THE INVERSE OF THE PERPENDICULAR COMPONENT OF EPSILON WITH CORRECTION #
#############################################################################################

def epsilonPerpInv_OneSigmoid_OneGaussian_withPolynomialCorrectionOnEpsilon(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and a 3σ-gaussian (99.7% of the values are in the interval).
    Corrects the function by adding a third order polynomial so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)

    box_halfLength = z2 - z1
    Z = z - z1
    
    slope1 = - epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z1, params) / epsilonPerpInv_OneSigmoid_OneGaussian(z1, params)**2
    slope2 = - epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z2, params) / epsilonPerpInv_OneSigmoid_OneGaussian(z2, params)**2
    
    correctionPolynomial = - (slope1 + slope2) / box_halfLength**2 * Z**3 + (2 * slope1 + slope2) / box_halfLength * Z**2 - slope1 * Z
    epsilon_withCorrection = 1.0 / epsilonPerpInv_OneSigmoid_OneGaussian(z, params) + correctionPolynomial

    return 1.0 / epsilon_withCorrection

def epsilonPerpInvDerivative_OneSigmoid_OneGaussian_withPolynomialCorrectionOnEpsilon(z: numpy.ndarray, params: dict[str, float]):
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and a gaussian, considering the function has been corrected by adding a third order polynomial so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)

    box_halfLength = z2 - z1
    Z = z - z1

    slope1 = - epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z1, params) / epsilonPerpInv_OneSigmoid_OneGaussian(z1, params)**2
    slope2 = - epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z2, params) / epsilonPerpInv_OneSigmoid_OneGaussian(z2, params)**2
    
    correctionPolynomial = - (slope1 + slope2) / box_halfLength**2 * Z**3 + (2 * slope1 + slope2) / box_halfLength * Z**2 - slope1 * Z
    correctionPolynomialDerivative = - 3 * (slope1 + slope2) / box_halfLength**2 * Z**2 + 2 * (2 * slope1 + slope2) / box_halfLength * Z - slope1
    
    epsilonInv = epsilonPerpInv_OneSigmoid_OneGaussian(z, params)
    epsilonInvDerivative = epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z, params)

    return (epsilonInvDerivative - epsilonInv**2 * correctionPolynomialDerivative) / (1.0 + correctionPolynomial * epsilonInv)**2


def epsilonPerpInv_OneSigmoid_OneGaussian_withExponentialCorrection(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and a 3σ-gaussian (99.7% of the values are in the interval).
    Corrects the function by adding a exponentials so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)
    
    Z1 = z - z1
    Z2 = z2 - z
    
    K = 50
    
    slope1 = epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z1, params)
    slope2 = epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z2, params)
    
    correction = - slope1 * Z1 * numpy.exp(- K * Z1) + slope2 * Z2 * numpy.exp(- K * Z2)
    
    return epsilonPerpInv_OneSigmoid_OneGaussian(z, params) + correction

def epsilonPerpInvDerivative_OneSigmoid_OneGaussian_withExponentialCorrection(z: numpy.ndarray, params: dict[str, float]):
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and a gaussian, considering the function has been corrected by adding exponentials so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the gaussian interval.
        z3: Upper limit of the gaussian interval.
        C: Parameter of the gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)
    
    Z1 = z - z1
    Z2 = z2 - z
    
    K = 50
    
    slope1 = epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z1, params)
    slope2 = epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z2, params)
    
    correction = - slope1 * (1 - K * Z1) * numpy.exp(- K * Z1) + slope2 * (K * Z2 - 1) * numpy.exp(- K * Z2)
    
    return epsilonPerpInvDerivative_OneSigmoid_OneGaussian(z, params) + correction


def epsilonPerpInv_OneSigmoid_TwoGaussians_withExponentialCorrection(z: numpy.ndarray, params: dict[str, float]) -> numpy.ndarray:
    """
    Models the inverse of the water dielectric permittivity perpendicular component with a sigmoid and two 3σ-gaussians (99.7% of the values are in the interval).
    Corrects the function by adding a exponentials so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the first gaussian interval.
        z3: Upper limit of the first gaussian interval.
        z4: Lower limit of the second gaussian interval.
        z5: Upper limit of the second gaussian interval.
        C1: Parameter of the first gaussian (controls its maxima).
        C2: Parameter of the second gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the derivative of epsilon.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)
    
    Z1 = z - z1
    Z2 = z2 - z
    
    K = 50
    
    slope1 = epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z1, params)
    slope2 = epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z2, params)
    
    correction = - slope1 * Z1 * numpy.exp(- K * Z1) + slope2 * Z2 * numpy.exp(- K * Z2)
    
    return epsilonPerpInv_OneSigmoid_TwoGaussians(z, params) + correction

def epsilonPerpInvDerivative_OneSigmoid_TwoGaussians_withExponentialCorrection(z: numpy.ndarray, params: dict[str, float]):
    """
    Returns the derivative of the inverse of the water dielectric permittivity perpendicular component modelled with a sigmoid and two gaussians, considering the function has been corrected by adding exponentials so that the derivatives are null at the borders.
    
    Model parameters:
        z1: Abscissa of the sigmoid inflection point.
        z2: Lower limit of the first gaussian interval.
        z3: Upper limit of the first gaussian interval.
        z4: Lower limit of the second gaussian interval.
        z5: Upper limit of the second gaussian interval.
        C1: Parameter of the first gaussian (controls its maxima).
        C2: Parameter of the second gaussian (controls its maxima).
        K: Parameter of the sigmoid (slope = K/4 at the inflection point).
        epsilonR_tails: Water dielectric permittivity value in the tails region.
        epsilonR_plateau: Water dielectric permittivity value in the plateau region.
    
    Parameters
    ----------
    x : numpy.ndarray
        The array of abscissa at which the charge density is computed.
    params : dict[str, float]
        The dictionnary containing all the parameters for the model.
    
    Returns
    -------
    numpy.ndarray
        The array containing the values of the model.
    """
    
    z1 = numpy.min(z)
    z2 = numpy.max(z)
    
    Z1 = z - z1
    Z2 = z2 - z
    
    K = 50
    
    slope1 = epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z1, params)
    slope2 = epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z2, params)
    
    correction = - slope1 * (1 - K * Z1) * numpy.exp(- K * Z1) + slope2 * (K * Z2 - 1) * numpy.exp(- K * Z2)
    
    return epsilonPerpInvDerivative_OneSigmoid_TwoGaussians(z, params) + correction
