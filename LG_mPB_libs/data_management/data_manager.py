import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LG_mPB_libs.data_management.data_source import DataSource
from LG_mPB_libs.data_management.data_type import DataType



class DataManager:
    """
    This class manages a collection of data (for now it handles only the loading, but it might be extended to handle saving later).
    
    Attributes
    ----------
    _dataSources : dict
        A dictionary containing all the DataSources. The keys are the sources names, choosen when calling AddDataSource() method.
    
    Methods
    -------
    __init__()
        The constructor of the DataManager class.
    __iter__()
        Iterates over the dataSources.
    __getitem__()
        Returns the requested DataSource.
    AddDataSource(name, url, type)
        Adds a new data source to the dictionary.
    Load(reload)
        Loads all sources.
    """
    
    def __init__(self) -> None:
        """
        Constructor of the DataManager class.
        """
        
        self._dataSources = {}
        
        return
    
    
    def __iter__(self) -> type(iter([])):
        """
        Returns the list_iterator associated to the _dataSources list.
        Allows to use "in" keyword on DataManager objetcs.
        
        Returns
        -------
        The iterator of the _dataSources list.
        """
        
        return iter(self._dataSources)
    
    
    def __getitem__(self, key: str) -> DataSource:
        """
        Returns the DataSource named after the key parameter.
        
        Parameters
        ----------
        key : str
            Name of the DataSource.
        
        Returns
        -------
        The DataSource named "key".
        
        Raises
        ------
        KeyError
            If the DataSource does not exist.
        """
        
        if not key in self._dataSources:
            raise KeyError(f"There is no DataSource named \"{key}\" in this DataManager.")
        
        return self._dataSources[key]
    
    
    def addDataSource(self, name: str, url: str, type: DataType) -> None:
        """
        Adds a new source of data to the manager's list.
        
        Parameters
        ----------
        name : str
            The name of the data source in the dictionary.
        url : str
            URL locating the data.
        type : DataType
            The type of data contained in the source. This allows custom loading of the data (see DataSource.Load() method).
        """
        
        dataSource = DataSource(url, type)
        self._dataSources[name] = dataSource
        
        return
    
    
    def load(self, reload: bool = False) -> None:
        """
        Loads all sources.
        
        Calls the load() method on every datasource contained in the manager's dictionary.
        
        Parameters
        ----------
        reload : bool
            Specifies if the data has to be loaded even if it had already been previously.
        """
        
        for dsName in self._dataSources.keys():
            self._dataSources[dsName].load(reload)
        
        return
        
