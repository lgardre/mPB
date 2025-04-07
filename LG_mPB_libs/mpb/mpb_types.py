"""
This modules defines Enum types usefull in this subpackage.
"""

from enum import Enum



class IonType(Enum):
    """ Defines the type of the ion. Type SALT is used when both species are present in the system. """
    
    ANION = 0
    CATION = 1
    SALT = 2


class EpsilonFormat(Enum):
    """ Specifies if the modified Poisson-Boltzmann is solved using a curve model for the dielectric permittivity or the inverse of it's perpendicular component. """
    
    REGULAR = 0
    INVERTED = 1


class SaltFittedData(Enum):
    """
    Determines how to fit the data for the systems that contain salt.
    
    CATIONS_AND_ANIONS: Fits on the concatenation of rhoCations and rhoAnions.
    NORMALIZED_CATIONS_AND_ANIONS: Fits on the concatenation of normalized rhoCations and normalized rhoAnions (the normalization coefficient being their value at the middle of the water, ie. at max z).
    CATIONS_AND_SQRT: Fits on the concatenation of rhoCations and the square root of the product of rhoCations and rhoAnions.
    """
    
    CATIONS_AND_ANIONS = 0
    NORMALIZED_CATIONS_AND_ANIONS = 1
    CATIONS_AND_SQRT = 2
