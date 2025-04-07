"""
This module contains tools to plot data from DataManagers or DataSources.
"""

import copy
import matplotlib.pyplot
import numpy
import sys
import types

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.data_management.data_manager
import LG_mPB_libs.tools.stats



#######################
# PLOTS CONFIGURATION #
#######################

SCALING = 3
CM = 1.0 / 2.54
FIGSIZE = (8 * CM * SCALING, 8 * CM * SCALING)
SUPTITLE_FONTSIZE = 32
TITLE_FONTSIZE = 24
LEGEND_FONTSIZE = 24
TICKS_FONTSIZE = 24
AXES_FONTSIZE = 24



##################
# PLOT FUNCTIONS #
##################

def plotChargeDensities(systemName: str, dataManagers: dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]], rootDir: str = '', start: int = 0, end: int = -1, step: int = 1) -> None:
    """
    Plots the charge densities.
    
    Parameters
    ----------
    systemName : str
        The name of the system displayed in the plot title.
    dataManagers : dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]]
        The dictionary that contains all data managers of the system, organized by temperature and barostat type. This function EXPECTS the DataManagers to have a dataSource named "Lipids_ChargeDensity" and a DataSource named "Cations_ChargeDensity". A DataSource named "Anions_ChargeDensity" is optional.
    rootDir : str
        The path to the root directory of the system. If empty, no plot is saved.
    start : int
        The first frame of the plotted data.
    end : int
        The last frame of the plotted data.
    step : int
        The number of frames between each plotted point.
    
    Raises
    ------
    KeyError
        If the "Lipids_ChargeDensity" and/or "Cations_ChargeDensity" DataSources are not found in any of the DataManagers.
    """
    
    toTheEnd = True if end == -1 else False
    
    for temperature in dataManagers:
        for pCouplingType in dataManagers[temperature]:
            
            if not "Lipids_ChargeDensity" in dataManagers[temperature][pCouplingType]:
                raise KeyError(f"No \"Lipids_ChargeDensity\" DataSource found in the DataManager for {temperature} K - {pCouplingType}.")
            
            if not "Cations_ChargeDensity" in dataManagers[temperature][pCouplingType]:
                raise KeyError(f"No \"Cations_ChargeDensity\" DataSource found in the DataManager for {temperature} K - {pCouplingType}.")
            
            if toTheEnd: end = len(dataManagers[temperature][pCouplingType]["Lipids_ChargeDensity"]["rho"])
            axis_z = dataManagers[temperature][pCouplingType]["Lipids_ChargeDensity"]['z'][start:end:step]
            data_lipid = dataManagers[temperature][pCouplingType]["Lipids_ChargeDensity"]["rho"][start:end:step]
            data_cation = dataManagers[temperature][pCouplingType]["Cations_ChargeDensity"]["rho"][start:end:step]
            
            if "Anions_ChargeDensity" in dataManagers[temperature][pCouplingType]:
                data_anion = dataManagers[temperature][pCouplingType]["Anions_ChargeDensity"]["rho"][start:end:step]
            
            figure, subFigure = matplotlib.pyplot.subplots()
            figure.suptitle(f"{systemName}: charge densities", fontsize = 16)
            subFigure.set_title(f"T = {temperature} K - {pCouplingType}")

            subFigure.set_xlabel(r"$z$ (nm)")
            subFigure.set_ylabel(r"$\rho$ ($e$.nm$^{-3}$)")

            subFigure.plot(axis_z, data_lipid, label = "lipids", color = "black")
            subFigure.plot(axis_z, data_cation, label = "cations", color = "red")
            
            if "Anions_ChargeDensity" in dataManagers[temperature][pCouplingType]:
                subFigure.plot(axis_z, data_anion, label = "anions", color = "green")
            
            subFigure.legend(prop = {"size" : 8})

            if rootDir != '':
                figure.savefig(f"{rootDir}/{temperature}K/analyse/{pCouplingType}/density/figures/chargeDensities_{temperature}K_{pCouplingType}.svg", transparent = True)

            matplotlib.pyplot.show()
    
    return

def plotSaltCalibration(systemName: str, dataManagers: dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]], temperature: float, pCouplingType: str, calibrationWindows: list[tuple] = None) -> float:
    """
    Plots the ionic charge densities for the calibration of "Csalt", for a precise system.
    
    Parameters
    ----------
    systemName : str
        The name of the system dysplayed in the plot title.
    dataManagers : dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]]
        The dictionary that contains all data managers of the system, organized by temperature and barostat type. This function EXPECTS the DataManagers to have a dataSource named "Anions_ChargeDensity_Calibration" and a DataSource named "Cations_ChargeDensity_Calibration".
    temperature : float
        The chosen temperature.
    pCouplingType : str
        The chosen barostat type.
    calibrationWindows : list[tuple]
        The data window(s) to use for the calibration. This consists in a list of tuples formed by the start and end indexes of the desired window(s). If none, the function just plots the charge densities.
    
    Returns
    -------
    Returns the mean salt concentration in the selected windows. If calibrationWindows is None, returns numpy.nan.
    """
    
    if not "Cations_ChargeDensity_Calibration" in dataManagers[temperature][pCouplingType]:
        raise KeyError(f"No \"Cations_ChargeDensity_Calibration\" DataSource found in the DataManager for {temperature} K - {pCouplingType}.")
    
    if not "Anions_ChargeDensity_Calibration" in dataManagers[temperature][pCouplingType]:
        raise KeyError(f"No \"Anions_ChargeDensity_Calibration\" DataSource found in the DataManager for {temperature} K - {pCouplingType}.")
    
    
    axis_z = dataManagers[temperature][pCouplingType]["Cations_ChargeDensity_Calibration"]['z']
    data_cations = dataManagers[temperature][pCouplingType]["Cations_ChargeDensity_Calibration"]["rho"]
    data_anions_abs = dataManagers[temperature][pCouplingType]["Anions_ChargeDensity_Calibration"]['n']
    
    data_salt = numpy.sqrt(numpy.asarray(data_cations) * numpy.asarray(data_anions_abs))
    
    figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
    # figure.suptitle(f"{systemName}: calibration", fontsize = SUPTITLE_FONTSIZE)
    subFigure.set_title(f"T = {temperature} K - {pCouplingType}", fontsize = TITLE_FONTSIZE)

    subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
    subFigure.set_ylabel(r"$C$ (nm$^{-3}$)", fontsize = AXES_FONTSIZE)
    subFigure.tick_params(labelsize = TICKS_FONTSIZE)

    subFigure.plot(axis_z, data_cations, label = r"$C_{+}$", color = "red")
    subFigure.plot(axis_z, data_anions_abs, label = r"$C_{-}$", color = "green")
    subFigure.plot(axis_z, data_salt, label = r"$\sqrt{C_{+} \times C_{-}}$", color = "blue")
    
    if calibrationWindows is not None:
        ds_cations_windowed = copy.deepcopy(dataManagers[temperature][pCouplingType]["Cations_ChargeDensity_Calibration"])
        ds_cations_windowed.selectWindows(calibrationWindows)
        data_cations_windowed = ds_cations_windowed["rho"]

        ds_anions_windowed = copy.deepcopy(dataManagers[temperature][pCouplingType]["Anions_ChargeDensity_Calibration"])
        ds_anions_windowed.selectWindows(calibrationWindows)
        data_anions_windowed = ds_anions_windowed['n']
    
        data_salt_windowed = numpy.sqrt(numpy.asarray(data_cations_windowed) * numpy.asarray(data_anions_windowed))
        mean = numpy.mean(data_salt_windowed)
        uncertainty = (numpy.max(data_salt_windowed) - numpy.min(data_salt_windowed)) / 2.0
    
        meanLabel = f"= {mean:.3f} +/- {uncertainty:.3f}"
        meanLabel = r"$C_{salt}$" # + meanLabel
        subFigure.axhline(y = mean, label = meanLabel, color = "black")

        for window in calibrationWindows:
            subFigure.axvspan(axis_z[window[0]], axis_z[window[1]], color = "orange", alpha = 0.1)
    
    subFigure.legend(prop = {"size" : LEGEND_FONTSIZE - 4}) #, bbox_to_anchor = (1, 0.5), loc = "center left")

    matplotlib.pyplot.show()
    
    return ( mean, uncertainty ) if calibrationWindows is not None else ( numpy.nan, numpy.nan )


def plotAreaPerLipid(systemName: str, dataManagers: dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]], lipidPerMonolayer: int = 100, rootDir: str = '', showMeanValue: bool = False, start: int = 0, end: int = -1, step: int = 1, nbSplits: int = 1) -> None:
    """
    Plots the area per lipid.
    
    Parameters
    ----------
    systemName : str
        The name of the system displayed in the plot title.
    dataManagers : dict[str, dict[str, LG_mPB_libs.data_management.data_manager.DataManager]]
        The dictionary that contains all data managers of the system, organized by temperature and barostat type. This function EXPECTS the DataManagers to have a dataSource named "BoxDimensions".
    lipidPerMonolayer : int
        The number of lipid molecules per monolayer.
    rootDir : str
        The path to the root directory of the system. If empty, no plot is saved.
    showMeanValue : bool
        Weather the mean values should be plotted
    start : int
        The first frame of the plotted data.
    end : int
        The last frame of the plotted data.
    step : int
        The number of frames between each plotted point.
    nbSplits : int
        The number of splits of the trajectory for statistical error calculation.
    
    Raises
    ------
    KeyError
        If the "BoxDimensions" DataSource is not found in any of the DataManagers.
    """
    
    toTheEnd = True if end == -1 else False
    
    for temperature in dataManagers:
        for pCouplingType in dataManagers[temperature]:
            
            if not "BoxDimensions" in dataManagers[temperature][pCouplingType]:
                raise KeyError(f"No \"BoxDimensions\" DataSource found in the DataManager for {temperature} K - {pCouplingType}.")
            
            if toTheEnd: end = len(dataManagers[temperature][pCouplingType]["BoxDimensions"]["xyArea"])
            axis_t = dataManagers[temperature][pCouplingType]["BoxDimensions"]['t'][start:end:step]
            data = dataManagers[temperature][pCouplingType]["BoxDimensions"]["xyArea"][start:end:step]

            if lipidPerMonolayer != 100:
                for i in range(0, len(data)): data[i] *= (100.0 / lipidPerMonolayer)
            
            stats = LG_mPB_libs.tools.stats.getStatsAreaPerLipid(dataManagers[temperature][pCouplingType]["BoxDimensions"], temperature, lipidPerMonolayer, start, end, step, nbSplits)
            
            figure, subFigure = matplotlib.pyplot.subplots()
            figure.suptitle(f"{systemName}: Area per Lipid", fontsize = 16)
            subFigure.set_title(f"T = {temperature} K - {pCouplingType}")

            subFigure.set_xlabel(r"$t$ (ns)")
            subFigure.set_ylabel(r"area per lipid (Å$^{2}$)")

            subFigure.plot(axis_t, data, color = "black")

            if showMeanValue:
                subFigure.axhline(y = stats["ApL_mean"], label = f'Mean value: {stats["ApL_mean"]:.2f}', color = "red")
                subFigure.legend(prop = {"size" : 8})

            if rootDir != '':
                figure.savefig(f"{rootDir}/{temperature}K/{pCouplingType}/areaPerLipid/figures/areaPerLipid_{temperature}K_{pCouplingType}.svg", transparent = True)

            matplotlib.pyplot.show()
            
            print(f'mean: {stats["ApL_mean"]} Å²')
            print(f'variance: {stats["ApL_variance"]} Å⁴')
            print(f'standard deviation: {stats["ApL_standardDeviation"]} Å²')
            
            if "ApL_uncertainty" in stats:
                print(f'uncertainty (95% trust interval): {stats["ApL_uncertainty"]} Å²')
            
            print(f'area compressibility: {stats["AC_mean"]} N.m⁻¹')
            
            if "AC_uncertainty" in stats:
                print(f'area compressibility uncertainty (95% trust interval): {stats["AC_uncertainty"]} Å²')
            
    return

def plotOrderParameter(systemName: str, dataSources: dict[str, dict[str, LG_mPB_libs.data_management.data_source.DataSource]], rootDir: str = '', relativeToMeanVectorLayer = False, showMeanValue: bool = False, start: int = 0, end: int = -1, step: int = 1) -> None:
    """
    
    """
    
    toTheEnd = True if end == -1 else False
    
    for temperature in dataSources:
        for pCouplingType in dataSources[temperature]:
            if toTheEnd: end = len(dataSources[temperature][pCouplingType]["bilayer_top"]["orderParam_normal"])
            
            data_monolayer = dataSources[temperature][pCouplingType]["monolayer"]["orderParam_normal"][start:end:step] if not relativeToMeanVectorLayer else dataSources[temperature][pCouplingType]["monolayer"]["orderParam_meanVectorLayer"][start:end:step]
            data_top = dataSources[temperature][pCouplingType]["bilayer_top"]["orderParam_normal"][start:end:step] if not relativeToMeanVectorLayer else dataSources[temperature][pCouplingType]["bilayer_top"]["orderParam_meanVectorLayer"][start:end:step]
            data_bottom = dataSources[temperature][pCouplingType]["bilayer_bottom"]["orderParam_normal"][start:end:step] if not relativeToMeanVectorLayer else dataSources[temperature][pCouplingType]["bilayer_bottom"]["orderParam_meanVectorLayer"][start:end:step]
            
            figure, subFigure = matplotlib.pyplot.subplots()
            figure.suptitle(f"{systemName}: Order Parameter", fontsize = 16)
            subFigure.set_title(f"T = {temperature} K - {pCouplingType}")

            subFigure.set_xlabel(r"$t$ (ns)")
            subFigure.set_ylabel(r"order parameter")

            subFigure.plot(dataSources[temperature][pCouplingType]["monolayer"]['t'][start:end:step], data_monolayer, label = "monolayer", color = "black")
            subFigure.plot(dataSources[temperature][pCouplingType]["bilayer_top"]['t'][start:end:step], data_top, label = "bilayer - top", color = "blue")
            subFigure.plot(dataSources[temperature][pCouplingType]["bilayer_bottom"]['t'][start:end:step], data_bottom, label = "bilayer - bottom", color = "orange")

            if showMeanValue:
                mean_monolayer = numpy.mean(data_monolayer)
                mean_top = numpy.mean(data_top)
                mean_bottom = numpy.mean(data_bottom)
                subFigure.axhline(y = mean_monolayer, label = f"Mean value monolayer: {mean_monolayer:.2f}", color = "red")
                subFigure.axhline(y = mean_top, label = f"Mean value bilayer - top: {mean_top:.2f}", color = "purple")
                subFigure.axhline(y = mean_bottom, label = f"Mean value bilayer - bottom: {mean_bottom:.2f}", color = "darkgreen")
                subFigure.legend(prop = {"size" : 8})

            if rootDir != '':
                figure.savefig(f"{rootDir}/{temperature}K/analyse/{pCouplingType}/orderParameter/figures/orderParameter_{temperature}K_{pCouplingType}.svg", transparent = True)

            matplotlib.pyplot.show()
    
    return

#def plotMSD(dataSource: LG_mPB_libs.data_management.data_source.DataSource, start: int = 0, end: int = -1, step: int = 1, nbSplits: int = 1) -> dict[str, float]:
    
    
    
