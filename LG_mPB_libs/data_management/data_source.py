import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LG_mPB_libs.data_management.data_type import DataType



class DataSource:
    """
    This class represents a source of data (typically a text file).
    It handles the loading of the data, which is customizable according to the type of data.
    
    Attributes
    ----------
    url : str
        URL locating the data.
    type : DataType
        The type of data contained in the source. This allows custom loading of the data (see Load() method).
    data : dict[str, list]
        A dictionary containing all the data from the source. The keys are the columns names, defined in the Load() method according to type attribute.
    loaded: bool
        The loading state of the DataSource.
    
    Methods
    -------
    __init__(url, type)
        The constructor of the DataSource class.
    __getitem__(key)
        Returns the data column named after the key parameter.
    load(reload)
        Loads the data.
    """
    
    
    def __init__(self, url: str, type: DataType) -> None:
        """
        Constructor of the DataSource class.
        
        Parameters
        ----------
        url : str
            URL locating the data.
        type : DataType
            The type of data contained in the source. This allows custom loading of the data (see Load() method).
        """
        
        self._url = url
        self._type = type
        self._data = {}
        self._loaded = False
        
        return
    
    
    def __getitem__(self, key: str) -> list:
        """
        Returns the data column named after the key parameter.
        
        Parameters
        ----------
        key : str
            Name of the data column.
        
        Raises
        ------
        KeyError
            If the column does not exist.
        """
        
        if not key in self._data:
            raise KeyError(f"There is no column named \"{key}\" in this DataSource.")
        
        return self._data[key]
    
    
    def load(self, reload: bool = False) -> None:
        """
        Loads the data.
        
        This method MUST BE completed whenever a new type is defined in the DataType Enum so that the new type can be loaded properly.
        
        Parameters
        ----------
        reload : bool
            Specifies if the data has to be loaded even if it had already been previously.
        
        Raises
        ------
        NotImplementedError
            If the DataSource holds a DataType which handling has not yet been implemented in this method.
        """
        
        if not self._loaded or reload:
            self._loaded = True
            
            with open(self._url, 'r') as dataFile:
                if self._type == DataType.GMX_DENSITY_CHARGE or self._type == DataType.GMX_DENSITY_CHARGE_LEFT_HALF or self._type == DataType.GMX_DENSITY_CHARGE_RIGHT_HALF:
                    self._data['z'] = []
                    self._data["rho"] = []
                    self._data['n'] = []
                    
                    for line in dataFile:
                        columns = line.split()
                        self._data['z'].append(float(columns[0]))
                        self._data["rho"].append(float(columns[1]))
                        self._data['n'].append(abs(float(columns[1])))
                    
                    if self._type == DataType.GMX_DENSITY_CHARGE_LEFT_HALF:
                        self._data['z'] = self._data['z'][: len(self._data['z']) // 2]
                        self._data["rho"] = self._data["rho"][: len(self._data["rho"]) // 2]
                        self._data['n'] = self._data['n'][: len(self._data['n']) // 2]
                    
                    elif self._type == DataType.GMX_DENSITY_CHARGE_RIGHT_HALF:
                        self._data['z'] = self._data['z'][len(self._data['z']) // 2 :]
                        self._data["rho"] = self._data["rho"][len(self._data["rho"]) // 2 :]
                        self._data['n'] = self._data['n'][len(self._data['n']) // 2 :]

                elif self._type == DataType.GMX_ENERGY_BOX_DIMENSIONS:
                    self._data['t'] = []
                    self._data['x'] = []
                    self._data['y'] = []
                    self._data['z'] = []
                    self._data["xyArea"] = []
                    self._data["xzArea"] = []
                    self._data["yzArea"] = []
                    self._data["volume"] = []
                    
                    for line in dataFile:
                        columns = line.split()
                        self._data['t'].append(float(columns[0]) / 1000)
                        self._data['x'].append(float(columns[1]))
                        self._data['y'].append(float(columns[2]))
                        self._data['z'].append(float(columns[3]))
                        self._data["xyArea"].append(float(columns[1]) * float(columns[2]))
                        self._data["xzArea"].append(float(columns[1]) * float(columns[3]))
                        self._data["yzArea"].append(float(columns[2]) * float(columns[3]))
                        self._data["volume"].append(float(columns[1]) * float(columns[2]) * float(columns[3]))
                
                elif self._type == DataType.LG_ORDER_PARAMETER:
                    self._data["monolayer"] = {}
                    self._data["bilayer_top"] = {}
                    self._data["bilayer_bottom"] = {}
                    for layer in self._data:
                        self._data[layer]['t'] = []
                        self._data[layer]["orderParam_normal"] = []
                        self._data[layer]["meanVectorLayer"] = []
                        self._data[layer]["meanCosAngle_normal_meanVectorLayer"] = []
                        self._data[layer]["orderParam_meanVectorLayer"] = []
                    
                    for line in dataFile:
                        if not line.startswith('#'):
                            columns = line.split()
                            if columns[1] == "MONO":
                                self._data["monolayer"]['t'].append(float(columns[0]) / 1000)
                                self._data["monolayer"]["orderParam_normal"].append(float(columns[2]))
                                self._data["monolayer"]["meanVectorLayer"].append([ float(columns[3]), float(columns[4]), float(columns[5]) ])
                                self._data["monolayer"]["meanCosAngle_normal_meanVectorLayer"].append(float(columns[6]))
                                self._data["monolayer"]["orderParam_meanVectorLayer"].append(float(columns[7]))
                            elif columns[1] == "TOP":
                                self._data["bilayer_top"]['t'].append(float(columns[0]) / 1000)
                                self._data["bilayer_top"]["orderParam_normal"].append(float(columns[2]))
                                self._data["bilayer_top"]["meanVectorLayer"].append([ float(columns[3]), float(columns[4]), float(columns[5]) ])
                                self._data["bilayer_top"]["meanCosAngle_normal_meanVectorLayer"].append(float(columns[6]))
                                self._data["bilayer_top"]["orderParam_meanVectorLayer"].append(float(columns[7]))
                            else:
                                self._data["bilayer_bottom"]['t'].append(float(columns[0]) / 1000)
                                self._data["bilayer_bottom"]["orderParam_normal"].append(float(columns[2]))
                                self._data["bilayer_bottom"]["meanVectorLayer"].append([ float(columns[3]), float(columns[4]), float(columns[5]) ])
                                self._data["bilayer_bottom"]["meanCosAngle_normal_meanVectorLayer"].append(float(columns[6]))
                                self._data["bilayer_bottom"]["orderParam_meanVectorLayer"].append(float(columns[7]))
                
                elif self._type == DataType.LG_EPSILON_R:
                    self._data["epsilonR"] = float(dataFile.readline())
                
                else:
                    self._loaded = False
                    raise NotImplementedError("The loading of this DataType has not been implemented yet.")
        
        return
    
    
    def selectWindow(self, start: int, end: int, step: int = 1) -> None:
        """
        Restrains the data to the specified window.
        
        Parameters
        ----------
        start : int
            First index of the selection window.
        end : int
            Last index of the selection window.
        step : int
            Step size for the selection window.
        """
        
        for column in self._data:
            if end == -1: end = len(self._data[column])
            
            window = self._data[column][start:end:step]
            self._data[column] = window
        
        return
    
    def selectWindows(self, windows: list[tuple], step: int = 1) -> None:
        """
        Restrains the data to the specified windows.
        
        Please note that the data from all the windows will be concatenated in the same order the windows are defined in the list attribute.
        
        Parameters
        ----------
        windows : list[tuple]
            List of index pairs (startIndex, endIndex) defining all the selection windows.
        step : int
            Step size for all selection windows.
        """
        
        for column in self._data:
            selection = []
            
            for window in windows:
                start = window[0]
                end = window[1] if window[1] != -1 else len(self._data[column])
                
                selection = selection + self._data[column][start:end:step]
        
            self._data[column] = selection
        
        return
