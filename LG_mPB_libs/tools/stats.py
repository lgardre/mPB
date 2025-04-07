"""
This module contains tools to compute specific stats on a DataSource.
"""

import numpy
import scipy.stats
import sys
import types

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.constants
import LG_mPB_libs.data_management.data_source



def getStatsAreaPerLipid(dataSource: LG_mPB_libs.data_management.data_source.DataSource, temperature: float, lipidPerMonolayer: int = 100, start: int = 0, end: int = -1, step: int = 1, nbSplits: int = 1) -> dict[str, float]:
    """
    Extracts area per lipid statistics from one precise "box dimensions DataSource".
    
    Parameters
    ----------
    dataSource : LG_mPB_libs.data_management.data_source.DataSource
        The "box dimensions" data source.
    temperature : float
        The temperature (to compute the area compressibility).
    lipidPerMonolayer : int
        The number of lipid molecules per monolayer.
    start : int
        The first frame for the analysis.
    end : int
        The last frame for the analysis.
    step : int
        The frame step for the analysis.
    nbSplits : int
        The number of splits of the trajectory for statistical error calculation.
    """
    
    if end == -1: end = len(dataSource["xyArea"])
    
    data = dataSource["xyArea"][start:end:step]

    if lipidPerMonolayer != 100:
        for i in range(0, len(data)): data[i] *= (100.0 / lipidPerMonolayer)
    
    if nbSplits == 1:
        mean_ApL = numpy.mean(data)
        variance_ApL = numpy.var(data, ddof = 1)
        standardDeviation_ApL = numpy.std(data, ddof = 1)
        areaCompressibility = mean_ApL / variance_ApL * LG_mPB_libs.constants.KB * temperature
    
        statsDict = { "ApL_mean": mean_ApL,
                      "ApL_variance": variance_ApL,
                      "ApL_standardDeviation": standardDeviation_ApL,
                      "AC_mean": areaCompressibility }
    else:
        start += int((end - start) / step) % nbSplits #supprime des frames au début si la division ne tombe pas juste
        nbFramesSplit = int(((end - start) / step) / nbSplits)
        
        means = []
        areaCompressibilities = []
        
        for i in range(0, nbSplits):
            mean = numpy.mean(data[start + nbFramesSplit * i : start + nbFramesSplit * (i + 1)])
            means.append(mean)
            
            variance = numpy.var(data[start + nbFramesSplit * i : start + nbFramesSplit * (i + 1)], ddof = 1)
            areaCompressibilities.append(mean / variance * LG_mPB_libs.constants.KB * temperature)
        
        mean_ApL = numpy.mean(means)
        variance_ApL = numpy.var(means, ddof = 1)
        standardDeviation_ApL = numpy.std(means, ddof = 1)
        uncertainty_ApL = standardDeviation_ApL / numpy.sqrt(nbSplits) * scipy.stats.t.isf(0.05 / 2, nbSplits - 1) # 95% trust interval
        
        mean_AC = numpy.mean(areaCompressibilities)
        variance_AC = numpy.var(areaCompressibilities, ddof = 1)
        standardDeviation_AC = numpy.std(areaCompressibilities, ddof = 1)
        uncertainty_AC = standardDeviation_AC / numpy.sqrt(nbSplits) * scipy.stats.t.isf(0.05 / 2, nbSplits - 1) # 95% trust interval
        
        statsDict = { "ApL_mean": mean_ApL,
                      "ApL_variance": variance_ApL,
                      "ApL_standardDeviation": standardDeviation_ApL,
                      "ApL_uncertainty": uncertainty_ApL,
                      "AC_mean": mean_AC,
                      "AC_variance": variance_AC,
                      "AC_standardDeviation": standardDeviation_AC,
                      "AC_uncertainty": uncertainty_AC }
    
    return statsDict

def getStatsVolume(dataSource: LG_mPB_libs.data_management.data_source.DataSource, start: int = 0, end: int = -1, step: int = 1, nbSplits: int = 1) -> dict[str, float]:
    """
    Extracts volume statistics from one precise "box dimensions DataSource".
    
    Parameters
    ----------
    dataSource : LG_mPB_libs.data_management.data_source.DataSource
        The "box dimensions" data source.
    start : int
        The first frame for the analysis.
    end : int
        The last frame for the analysis.
    step : int
        The frame step for the analysis.
    nbSplits : int
        The number of splits of the trajectory for statistical error calculation.
    """
    
    if end == -1: end = len(dataSource["volume"])
    
    data = dataSource["volume"][start:end:step]
    
    if nbSplits == 1:
        mean = numpy.mean(data)
        variance = numpy.var(data, ddof = 1)
        standardDeviation = numpy.std(data, ddof = 1)
    
        statsDict = { "mean": mean,
                      "variance": variance,
                      "standardDeviation": standardDeviation }
    else:
        start += int((end - start) / step) % nbSplits #supprime des frames au début si la division ne tombe pas juste
        nbFramesSplit = int(((end - start) / step) / nbSplits)
        
        means = []
        
        for i in range(0, nbSplits):
            mean = numpy.mean(data[start + nbFramesSplit * i : start + nbFramesSplit * (i + 1)])
            means.append(mean)
        
        mean = numpy.mean(means)
        variance = numpy.var(means, ddof = 1)
        standardDeviation = numpy.std(means, ddof = 1)
        uncertainty = standardDeviation / numpy.sqrt(nbSplits) * scipy.stats.t.isf(0.05 / 2, nbSplits - 1) # 95% trust interval
        
        statsDict = { "mean": mean,
                      "variance": variance,
                      "standardDeviation": standardDeviation,
                      "uncertainty": uncertainty }
    
    return statsDict
