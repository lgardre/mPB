from enum import Enum



class LayerLocation(Enum):
    """
    This Enum class represents the location of a lipid's layer, with respect to other lipids.
    """
    
    MONOLAYER = 0
    """
    The lipid is part of a monolayer.
    """
    
    TOP = 1
    """
    The lipid is part of a bilayer's top layer.
    """
    
    BOTTOM = 2
    """
    The lipid is part of a bilayer's bottom layer.
    """


class Lipid:
    """
    This class stores usefull informations about a lipid molecule.
    
    Attributes
    ----------
    resid : int
        The residue identifier of the lipid (starts from 1).
    location : LayerLocation
        The location of the layer holding this lipid, with respect to other lipids.
    atoms : dict[str, int]
        A dictionnary holding all atoms of interest. The keys are the names of the atoms, and the associated value is their identifier.
    """
    
    def __init__(self, resid: int) -> None:
        """
        Constructor of the Lipid class.
        
        Parameters
        ----------
        resid : int
            The residue identifier of the lipid (starts from 1).
        """
        
        self._resid = resid
        self._atoms = {}
        self._location = None
        self._headPosition = []
        self._tailsPosition = []
        self._tailsToHeadVector = []
        
        return
