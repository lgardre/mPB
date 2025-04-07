from enum import Enum



class DataType(Enum):
    """
    This Enum class defines any data format that can be loaded within the LG_Libs package.
    For each type, a custom code can be written to load the data.
    """
    
    GMX_DENSITY_CHARGE = 0
    """
    Describes a xvg file created by Gromacs gmx density program, containing charge density data. It is assumed that "-xvg none" has been used (no header in the file).
    
    Column informations:
    
    - Column 0: the axis (typically z) in nm
    - Column 1: the charge density in e⋅nm⁻³
    """
    
    GMX_DENSITY_CHARGE_LEFT_HALF = 1
    """
    Same as GMX_CHARGE_DENSITY but only contains the left half. Usefull for bilayers.
    """
    
    GMX_DENSITY_CHARGE_RIGHT_HALF = 2
    """
    Same as GMX_CHARGE_DENSITY but only contains the right half. Usefull for bilayers.
    """
    
    GMX_ENERGY_BOX_DIMENSIONS = 3
    """
    Describes a xvg file created by Gromacs gmx energy program, containing the length of all sides of a simulation box. It is assumed that "-xvg none" has been used (no header in the file).
    
    Column informations:
    
    - Column 0: the time in ps
    - Column 1: The length of the box in the x direction in nm
    - Column 2: The length of the box in the y direction in nm
    - Column 3: The length of the box in the z direction in nm
    """
    
    LG_ORDER_PARAMETER = 4
    """
    Describes a xvg file created by LG_Analysis library, containing the order parameter for a given chain in a lipid.
    
    Column informations:
    
    - Column 0: the time in ps
    - Column 1: the lipid layer (bottom/top of a bilayer, monolayer of a trilayer)
    - Column 2: the order parameter relative to the normal to the lipid layer
    - Column 3: the x component of the layer's mean vector of all lipids belonging to the layer
    - Column 4: the y component of the layer's mean vector of all lipids belonging to the layer
    - Column 5: the z component of the layer's mean vector of all lipids belonging to the layer
    - Column 6: the mean cosinus of the angle between the layer's normal vector and the mean vector of all lipids belonging to the layer
    - Column 7: the order parameter relative to the mean vector of all lipids belonging to the layer.
    """
    
    LG_EPSILON_R = 5
    """
    Describes a txt file created by LG_Analysis library, containing the dielectric constant obtained from dipoles fluctuations.
    The file only contains this value.
    """
