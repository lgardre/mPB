"""
This module contains all constants usefull in the whole LG_Libs package.
"""

from enum import Enum



class WaterModel(Enum):
    """ The water model used in the simulation. """
    
    TIP3P = "TIP3P"
    OPC = "OPC"


EPSILON_ZERO = 8.85418782E-39
""" Vacuum dielectric permittivity in A²⋅s⁴⋅kg⁻¹⋅nm⁻³. """

KB = 1.380649E-5
""" Boltzmann constant in aJ⋅K⁻¹ (SI : nm²⋅kg⋅s⁻²⋅K⁻¹). """

E = 1.602176634E-19
""" Elementary charge in C (SI : A⋅s) """

H2O_VOLUME = 27E-3
""" Volume of a water molecule in nm⁻³. """