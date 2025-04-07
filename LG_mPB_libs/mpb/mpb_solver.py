import lmfit
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import scipy.integrate
import sys
import types

from enum import Enum
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.constants
import LG_mPB_libs.mpb.characteristic_lengths
import LG_mPB_libs.mpb.physics_properties
import LG_mPB_libs.mpb.mpb_types
import LG_mPB_libs.mpb.uncalibrated_values



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



####################
# MPB SOLVER CLASS #
####################

class MPBSolver:
    """
    The MPBSolver (Modified Poisson-Boltzmann Solver) class is responsible for solving mPB equation for a defined system.
    
    This class requires the definition of a function that returns the dielectric permittivity of the system (and its derivative), which can be parametrized with help of a dictionnary of parameters (the keys being the names of the parameter). Built-in functions of this kind are already defined in this Python file, but the user is free to define its own model.
    To solve the mPB equation, a lipid charge density profile must be provided to the class (typically obtained from a reference MD simulation). Solving the equation will give an electric potential profile which can then be used to compute ionic charge density profiles.
    This class also serves as a wrapper to optimize the parameters of the given dielectric permittivity model to better fit the ions density profiles obtained in the MD reference simulation.
    
    Attributes
    ----------
    _plotsMainTitleRoot : str
        The name of the system displayed in the plot titles.
    _plotsSubtitle : str
        The text displayed in the plot subtitles.
    _x : numpy.ndarray
        The array of abscissa at which the modified Poisson-Boltzmann equation is solved.
    _rhoLipidsMD : numpy.ndarray
        The charge density of the lipids obtained from the reference MD simulation.
    _rhoAnionsMD : numpy.ndarray
        The charge density of the anions obtained from the reference MD simulation.
    _rhoCationsMD : numpy.ndarray
        The charge density of the cations obtained from the reference MD simulation.
    _epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
    _epsilonDerivative : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
        The derivative of the epsilon function.
    _parameters : lmfit.Parameters
        The list of the fitting parameters, for optimization purpose. Uses the syntax of lmfit library (https://lmfit.github.io/lmfit-py/).
    _parametersDict : dict[str, float]
        The list of the parameters for the calculation of the water dielectric permittivity.
    _phiInitGuess : numpy.ndarray
        Initial guess for the electric potential values in V⋅nm⁻¹.
    _phi : numpy.ndarray
        Electric potential values (in V⋅nm⁻¹) after solving mPB equation.
    _systemIons : LG_mPB_libs.mpb.mpb_types.IonType
        Describes the type of ions present in the water.
    _epsilonFormat : LG_mPB_libs.mpb.mpb_types.EpsilonFormat
        The format of the curve giving the water dielectric permittivity (see LG_mPB_libs.mpb.mpb_types.EpsilonFormat enum documentation).
    _model : lmfit.Model
        The instance of lmfit.Model that does the optimization of the parameters.
    _results : lmfit.model.ModelResult
        Contains the results of the parameters optimization.
    _fittedData : numpy.mdarray
        The data that is actually fitted by the model. It depends on the type of ions in the system.
    _modelFunction : Callable[[numpy.ndarray], numpy.ndarray]
        The function that returns the expected values of the model given an array of abscissa.
    
    Methods
    -------
    __init__(plotsMainTitleRoot, plotsSubtitle, x, rhoLipidsMD, rhoAnionsMD, rhoCationsMD, T, epsilon, epsilonDerivative, calibratedParameters, parameters, phi, systemIons, epsilonFormat):
        Constructor of the MPBSolver Class.
    getODEsystem(x, y):
        Rewrites and returns the modified Poisson-Boltzmann equation as an Ordinary Differential Equations system, which allows to solve it numerically.
    boundaryConditions(self, ya: numpy.ndarray, yb: numpy.ndarray):
        Defines the boundary conditions to solve the modified Poisson-Boltzmann equation under the form of residues. Here, the first derivatives of the unknown are equal to zero at both borders.
    solve_mPB(self, x: numpy.ndarray):
        Solves the Ordinary Differential Equations system that represents the modified Poisson-Boltzmann equation.
    getModelIonicChargeDensity():
        Returns the model for the ionic charge density, with respect to the ions species present in the problem.
    solveOptimized():
        Solves the modified Poisson-Boltzmann equation and the optimal values of the given parameters so that the solution allows to reproduce at best the ionic charge densities model.
    correctEpsilonDerivatives():
        Corrects the epsilon curve by adding a polynomial curve (of order 3) to it, so that the derivative of the sum is null at the borders.
    plotPhi(self, saveURL: str = "", showDerivative: bool = False)
        Plots the electric potential, both initial guess and the numerical solution to the modified Poisson-Boltzmann equation.
    plotCharges():
        Plots all the charge densities of the system, both from the model data and the prediction comming from the solution of the modified Poisson-Boltzmann equation.
    plotEpsilon():
        Plots the dielectric permittivity profile used to solve the modified Poisson-Boltzmann equation.
    plotEpsilonInv():
        Plots the profile of tre inverse of the dielectric permittivity used to solve the modified Poisson-Boltzmann equation.
    plotTotalChargeIntegral():
        Plots the cumulative integral of the total charge density in the system, along the z axis.
    plotPlasmaParameter():
        Plots the plasma parameter value along the z axis.
    getRhoAnions():
        Returns the array containing the charge density prediction for the anions.
    getRhoCations():
        Returns the array containing the charge density prediction for the cations.
    """
    
    def __init__(self,
                 plotsMainTitleRoot: str,
                 plotsSubtitle: str,
                 x: numpy.ndarray,
                 rhoLipidsMD: numpy.ndarray,
                 rhoAnionsMD: numpy.ndarray,
                 rhoCationsMD: numpy.ndarray,
                 T: float,
                 epsilon: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray],
                 epsilonDerivative: Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray],
                 parameters: lmfit.Parameters,
                 phi: numpy.ndarray,
                 systemIons: LG_mPB_libs.mpb.mpb_types.IonType,
                 epsilonFormat: LG_mPB_libs.mpb.mpb_types.EpsilonFormat,
                 saltFittedData: LG_mPB_libs.mpb.mpb_types.SaltFittedData = LG_mPB_libs.mpb.mpb_types.SaltFittedData.NORMALIZED_CATIONS_AND_ANIONS) -> None:
        """
        Constructor of the MPBSolver Class.
        
        Parameters
        ----------
        plotsMainTitleRoot : str
            The name of the system displayed in the plot titles.
        plotsSubtitle : str
            The text displayed in the plot subtitles.
        x : numpy.ndarray
            The array of abscissa at which the modified Poisson-Boltzmann equation is solved.
        rhoLipidsMD : numpy.ndarray
            The charge density of the lipids obtained from the reference MD simulation.
        rhoAnionsMD : numpy.ndarray
            The charge density of the anions obtained from the reference MD simulation.
        rhoCationsMD : numpy.ndarray
            The charge density of the cations obtained from the reference MD simulation.
        T : float
            The temperature in K.
        epsilon : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
           The function that gives the water dielectric permittivity. This function must accept two arguments: the array of abscissa and its parameters dictionary.
        epsilonDerivative : Callable[[numpy.ndarray, dict[str, float]], numpy.ndarray]
            The derivative of the epsilon function.
        parameters : lmfit.Parameters
            The list of the fitting parameters, for optimization purpose. Uses the syntax of lmfit library (https://lmfit.github.io/lmfit-py/).
        phi : numpy.ndarray
            Electric potential values (in V⋅nm⁻¹) after solving mPB equation.
        systemIons : LG_mPB_libs.mpb.mpb_types.IonType
            Describes the type of ions present in the water.
        epsilonFormat : LG_mPB_libs.mpb.mpb_types.EpsilonFormat
            The format of the curve giving the water dielectric permittivity (see LG_mPB_libs.mpb.mpb_types.EpsilonFormat enum documentation).
        """
        
        self._plotsMainTitleRoot = plotsMainTitleRoot
        self._plotsSubtitle = plotsSubtitle
        self._x = x
        self._rhoLipidsMD = rhoLipidsMD
        self._rhoAnionsMD = rhoAnionsMD
        self._rhoCationsMD = rhoCationsMD
        self._T = T
        self._epsilon = epsilon
        self._epsilonDerivative = epsilonDerivative
        self._parameters = parameters
        self._parametersDict = {}
        self._phiInitGuess = phi
        self._phi = phi
        self._systemIons = systemIons
        self._epsilonFormat = epsilonFormat
        self._saltFittedData = saltFittedData
        
        
        if systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
            self._fittedData = rhoAnionsMD
        elif systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
            self._fittedData = rhoCationsMD
        else:
            if saltFittedData == LG_mPB_libs.mpb.mpb_types.SaltFittedData.CATIONS_AND_ANIONS:
                self._fittedData = numpy.concatenate(( rhoCationsMD, rhoAnionsMD ))
            elif saltFittedData == LG_mPB_libs.mpb.mpb_types.SaltFittedData.NORMALIZED_CATIONS_AND_ANIONS:
                self._fittedData = numpy.concatenate(( rhoCationsMD / rhoCationsMD[-1], rhoAnionsMD / numpy.abs(rhoAnionsMD[-1]) ))
            else:
                self._fittedData = numpy.concatenate(( rhoCationsMD, numpy.sqrt(numpy.abs(rhoCationsMD * rhoAnionsMD)) ))
        
        
        # If not declared, we add default calibrated parameters
        if not "epsilonR_tails" in parameters:
            self._parameters.add("epsilonR_tails", value = LG_mPB_libs.mpb.uncalibrated_values.EPSILON_R_TAILS, vary = False)
        
        if not "epsilonR_bulkWater" in parameters:
            self._parameters.add("epsilonR_bulkWater", value = LG_mPB_libs.mpb.uncalibrated_values.EPSILON_R_BULKWATER, vary = False)
        
        if not "epsilonR_plateau" in parameters:
            self._parameters.add("epsilonR_plateau", expr = "epsilonR_bulkWater") # if epsilonR_plateau is not declared, we couple it to epsilonR_bulkWater (usefull for calibration for instance)
        
        if not "cSalt" in parameters:
            self._parameters.add("cSalt", value = LG_mPB_libs.mpb.uncalibrated_values.C_SALT, vary = False)
        
        if not "rCation" in parameters:
            self._parameters.add("rCation", value = LG_mPB_libs.mpb.uncalibrated_values.R_CATION, vary = False)
        
        if not "rAnion" in parameters:
            self._parameters.add("rAnion", value = LG_mPB_libs.mpb.uncalibrated_values.R_ANION, vary = False)
        
        
        # Dynamic creation of the model function.
        s = "def f(self, x"
        for p in self._parameters:
            self._parametersDict[p] = parameters[p].value
            s += f", {p}"
        s += "):\n"
        for p in self._parameters:
            s += f"\tself._parametersDict[\"{p}\"] = {p}\n"
        s += "\tself.solve_mPB(x)\n"
        s += "\treturn self.getModelIonicChargeDensity()\n"
        s += "self._modelFunction = types.MethodType(f, self)\n"

        exec(s)
        
        self._model = lmfit.Model(self._modelFunction)
        self._results = None
        
        return
    
    
    def getODEsystem(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        """
        Rewrites and returns the modified Poisson-Boltzmann equation as an Ordinary Differential Equations system, which allows to solve it numerically.
        
        Parameters
        ----------
        x : numpy.ndarray
            The array of abscissa at which the modified Poisson-Boltzmann equation is solved.
        y : numpy.ndarray
            The array containing the unknown function. First index corresponds to the function, second index corresponds to its first derivative, and so on if needed.
        
        Returns
        -------
        numpy.ndarray
            The Ordinary Differential Equations system defining the modified Poisson-Boltzmann equation.
        """
        
        ODEArray = None
        
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * self._epsilon(x, self._parametersDict) / LG_mPB_libs.constants.EPSILON_ZERO
                                         * (self._parametersDict["cSalt"]
                                            * numpy.exp(y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         + (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
            else:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 / (LG_mPB_libs.constants.EPSILON_ZERO * self._epsilon(x, self._parametersDict))
                                         * (self._parametersDict["cSalt"]
                                            * numpy.exp(y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         - (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
        
        elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * self._epsilon(x, self._parametersDict) / LG_mPB_libs.constants.EPSILON_ZERO
                                         * (self._parametersDict["cSalt"]
                                            * (- numpy.exp(- y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat)))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         + (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
            else:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 / (LG_mPB_libs.constants.EPSILON_ZERO * self._epsilon(x, self._parametersDict))
                                         * (self._parametersDict["cSalt"]
                                            * (- numpy.exp(- y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat)))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         - (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
        
        else:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * self._epsilon(x, self._parametersDict) / LG_mPB_libs.constants.EPSILON_ZERO
                                         * (self._parametersDict["cSalt"]
                                            * (numpy.exp(y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat))
                                               - numpy.exp(- y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat)))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         + (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
            else:
                ODEArray = numpy.vstack((y[1],
                                         LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 / (LG_mPB_libs.constants.EPSILON_ZERO * self._epsilon(x, self._parametersDict))
                                         * (self._parametersDict["cSalt"]
                                            * (numpy.exp(y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat))
                                               - numpy.exp(- y[0] - LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat)))
                                            - numpy.interp(x, self._x, self._rhoLipidsMD))
                                         - (1.0 / self._epsilon(x, self._parametersDict)) * self._epsilonDerivative(x, self._parametersDict) * y[1]))
        
        return ODEArray
    
    
    def boundaryConditions(self, ya: numpy.ndarray, yb: numpy.ndarray) -> numpy.ndarray:
        """
        Defines the boundary conditions to solve the modified Poisson-Boltzmann equation under the form of residues. Here, the first derivatives of the unknown are equal to zero at both borders.
        
        Parameters
        ----------
        ya : numpy.ndarray
            The evaluation of the unknown function (and its derivatives) at the left boundary.
        yb : numpy.ndarray
            The evaluation of the unknown function (and its derivatives) at the right boundary.
        
        Returns
        -------
        numpy.ndarray
            The boundary conditions used to solve the modified Poisson-Boltzmann equation.
        """
        
        return numpy.array([ya[1], yb[1]])
    
    
    def solve_mPB(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Solves the Ordinary Differential Equations system that represents the modified Poisson-Boltzmann equation.
        
        Parameters
        ----------
        x : numpy.ndarray
            The array of abscissa at which the modified Poisson-Boltzmann equation is solved.
        
        Returns
        -------
        numpy.ndarray
            The solution to the modified Poisson-Boltzmann equation (ie. the electric potential).
        """
        
        solution = scipy.integrate.solve_bvp(self.getODEsystem, self.boundaryConditions, x, self._phi, tol = 1.0E-6, max_nodes = 5000000)
        
        if solution.status != 0:
            print(solution.message)
        
        self._phi = solution.sol(self._x)
        
        return self._phi
    
    
    def getModelIonicChargeDensity(self) -> numpy.ndarray:
        """
        Returns the model for the ionic charge density, with respect to the ions species present in the problem.
        
        Returns
        -------
        numpy.ndarray
            The array containing the ionic charge density in the system.
        """
        
        modelChargeArray = None
        
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
            modelChargeArray = LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
        elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
            modelChargeArray = LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
        else:
            if self._saltFittedData == LG_mPB_libs.mpb.mpb_types.SaltFittedData.CATIONS_AND_ANIONS:
                modelChargeArray = numpy.concatenate(( LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat),
                                                       LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat) ))
            elif self._saltFittedData == LG_mPB_libs.mpb.mpb_types.SaltFittedData.NORMALIZED_CATIONS_AND_ANIONS:
                rhoCationsModel = LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
                rhoAnionsModel = LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
                modelChargeArray = numpy.concatenate(( rhoCationsModel / self._rhoCationsMD[-1], rhoAnionsModel / numpy.abs(self._rhoAnionsMD[-1]) ))
            else:
                modelChargeArray = numpy.concatenate(( LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat),
                                                       numpy.sqrt(numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
                                                                            * LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))) ))
        
        return modelChargeArray
    
    
    def solveOptimized(self) -> tuple[lmfit.model.ModelResult, pandas.DataFrame]:
        """
        Solves the modified Poisson-Boltzmann equation and the optimal values of the given parameters so that the solution allows to reproduce at best the ionic charge densities model.
        
        Returns
        -------
        tuple[lmfit.model.ModelResult, pandas.DataFrame]
            A tuple that contains the global report of the optimization and the dataframe containing the parameters and their optimized value.
        """
        
        self._results = self._model.fit(self._fittedData, self._parameters, x = self._x, method = "least_squares")
        
        lst = []
        for p in self._results.params:
            lst.append( { "value": self._results.params[p].value, "uncertainty": self._results.params[p].stderr } )
        
        dataFrame = pandas.DataFrame(lst, index = (p for p in self._results.params))
        
        return self._results, dataFrame
    
    
    def plotElectricPotential(self, saveURL: str = "", showInitialGuess: bool = True, showDerivative: bool = False) -> None:
        """
        Plots the electric potential, both initial guess and the numerical solution to the modified Poisson-Boltzmann equation.
        
        Parameters
        ----------
        saveURL : str
            The absolute path for saving the graph.
        showInitialGuess: bool
            Indicates we
        showDerivative : bool
            Indicates whether the numerical derivative of the electric potential should be displayed or not.
        """
        
        subFiguresNbr = 2 if showDerivative else 1
        
        figure, subFigures = matplotlib.pyplot.subplots(subFiguresNbr, figsize = FIGSIZE, sharex = True)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Electric potential", fontsize = SUPTITLE_FONTSIZE)
        
        subFigure_0_phi = subFigures[0] if showDerivative else subFigures
        subFigure_0_phi.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure_0_phi.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure_0_phi.set_ylabel(r"$\phi$", fontsize = AXES_FONTSIZE)
        subFigure_0_phi.tick_params(labelsize = TICKS_FONTSIZE)
        
        if showInitialGuess:
            subFigure_0_phi.plot(self._x, self._phiInitGuess[0], label = "guess", color = "dimgrey", linestyle = ":", linewidth = 1)
        
        subFigure_0_phi.plot(self._x, self._phi[0], label = "numerical", color = "darkorange")
        
        subFigure_0_phi.set_ylim(top = 0.0)
        
        subFigure_0_V = subFigure_0_phi.twinx()
        subFigure_0_V.set_ylabel(r"$V$ (mV)", fontsize = AXES_FONTSIZE)
        subFigure_0_V.tick_params(labelsize = TICKS_FONTSIZE)
        lim1, lim2 = subFigure_0_phi.get_ylim()
        subFigure_0_V.set_ylim(lim1 / (1e15 * LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E), lim2 / (1e15 * LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E))
        
        if showDerivative:
            subFigures[1].set_ylabel(r"$\frac{\mathrm{d}\phi}{\mathrm{d}z}$ (V.nm$^{-1}$)", fontsize = AXES_FONTSIZE)
            subFigures[1].tick_params(labelsize = TICKS_FONTSIZE)
            subFigures[1].plot(self._x, self._phiInitGuess[1], label = "guess derivative", color = "silver")
            subFigures[1].plot(self._x, self._phi[1], label = "numerical derivative", color = "sandybrown")
            
            lines0, labels0 = subFigures[0].get_legend_handles_labels()
            lines1, labels1 = subFigures[1].get_legend_handles_labels()
            
            matplotlib.pyplot.legend(lines0 + lines1, labels0 + labels1, prop = {"size" : LEGEND_FONTSIZE})
        elif showInitialGuess:
            subFigure_0_phi.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotCharges(self) -> None:
        """
        Plots all the charge densities of the system, both from the model data and the prediction comming from the solution of the modified Poisson-Boltzmann equation.
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Charge densities", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"$\rho$ ($e$.nm$^{-3}$)", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)
        
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
            subFigure.plot(self._x, self._rhoAnionsMD, color = "springgreen", linestyle = (0, (5, 5)), linewidth = 3)
            subFigure.plot(self._x, LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{-}$", color = "green")
        
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
            subFigure.plot(self._x, self._rhoCationsMD, color = "salmon", linestyle = (0, (5, 5)), linewidth = 3)
            subFigure.plot(self._x, LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{+}$", color = "red")
        
        subFigure.plot(self._x, self._rhoLipidsMD, label = r"$\rho_{lipids}^{MD}$", color = "black")
        
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
            subFigure.plot(self._x, self._rhoLipidsMD + self._rhoAnionsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3)
            subFigure.plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "purple")
        elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
            subFigure.plot(self._x, self._rhoLipidsMD + self._rhoCationsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3)
            subFigure.plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "purple")
        else:
            subFigure.plot(self._x, self._rhoLipidsMD + self._rhoAnionsMD + self._rhoCationsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3)
            subFigure.plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat) + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "purple")
        
        subFigure.legend(prop = {"size": LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotEpsilon(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None, showEpsilonBorn: bool = False) -> None:
        """
        Plots the dielectric permittivity profile used to solve the modified Poisson-Boltzmann equation.
        
        Parameters
        ----------
        otherSolvers : list[MPBSolver]
            Another solvers to compare all epsilon curves.
        labels : list[str]
            List of the labels for the curves. Length must be len(otherSolvers) + 1. First index is for the current solver, other indexes follow otherSolvers order.
        showEpsilonBorn : bool
            Whether or not to show the Born permittivity (permittivity without the electrostatic interaction)
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Dielectric permittivity", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"$\varepsilon_{\perp}$", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)
        
        if otherSolvers != None:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR:
                subFigure.plot(self._x, self._epsilon(self._x, self._parametersDict), label = labels[0], color ="black", linewidth = 2)
            else:
                subFigure.plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), label = labels[0], color ="black", linewidth = 2)
            
            index = 1
            for solver in otherSolvers:
                if solver._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR:
                    subFigure.plot(solver._x, solver._epsilon(solver._x, solver._parametersDict), label = labels[index], color ="black", linewidth = 2)
                else:
                    subFigure.plot(solver._x, 1.0 / solver._epsilon(solver._x, solver._parametersDict), label = labels[index], color ="black", linewidth = 2)
                
                index += 1
        
        else:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR:
                subFigure.plot(self._x, self._epsilon(self._x, self._parametersDict), label = "mPB", color ="black", linewidth = 2)
            else:
                subFigure.plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), label = "mPB", color ="black", linewidth = 2)
        
        if showEpsilonBorn:
            epsilonInvBorn = (16 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO
                                            / (LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * (1.0 / self._parametersDict["rCation"] + 1.0 / self._parametersDict["rAnion"]))
                                            * (numpy.log(self._parametersDict["cSalt"])
                                               - numpy.log(numpy.sqrt(numpy.abs(self._rhoCationsMD) * numpy.abs(self._rhoAnionsMD))))
                              + 1.0 / self._parametersDict["epsilonR_bulkWater"])

            for i in range(0, len(epsilonInvBorn)):
                if epsilonInvBorn[i] == numpy.inf: epsilonInvBorn[i] = numpy.NaN # Do not plot values where the log is not defined
            
            subFigure.plot(self._x, 1.0 / epsilonInvBorn, label = "Born", linewidth = 2, linestyle = "dashed", color = "red")
        
        if otherSolvers != None or showEpsilonBorn:
            subFigure.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    def plotEpsilonIntegrated(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None, showEpsilonBorn: bool = False) -> None:
        """
        Plots the profile of the inverse of the dielectric permittivity used to solve the modified Poisson-Boltzmann equation, integrated over the axis from 0 to the current value, divided by this distance.
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Inverse of dielectric permittivity", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"$\frac{l}{\int_0^l~\frac{\mathrm{d}z}{\varepsilon_{\perp}^{\mathrm{mPB}}}}$", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)
        
        if otherSolvers != None:
            subFigure.plot(self._x, self.getEpsilon_integrated(), label = labels[0])
            
            index = 1
            for solver in otherSolvers:
                subFigure.plot(solver._x, solver.getEpsilon_integrated(), label = labels[index])
        
        else:
            subFigure.plot(self._x, self.getEpsilon_integrated())
            
        if otherSolvers != None:
            subFigure.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotEpsilonInv(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None, showEpsilonBorn: bool = False) -> None:
        """
        Plots the profile of the inverse of the dielectric permittivity used to solve the modified Poisson-Boltzmann equation.
        
        Parameters
        ----------
        otherSolvers : list[MPBSolver]
            Another solvers to compare all epsilon curves.
        labels : list[str]
            List of the labels for the curves. Length must be len(otherSolvers) + 1. First index is for the current solver, other indexes follow otherSolvers order.
        showEpsilonBorn : bool
            Whether or not to show the Born permittivity (permittivity without the electrostatic interaction)
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Inverse of dielectric permittivity", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"$\varepsilon_{\perp}^{-1}$", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)
        
        if otherSolvers != None:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                subFigure.plot(self._x, self._epsilon(self._x, self._parametersDict), label = labels[0])
            else:
                subFigure.plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), label = labels[0])
            
            index = 1
            for solver in otherSolvers:
                if solver._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                    subFigure.plot(solver._x, solver._epsilon(solver._x, solver._parametersDict), label = labels[index])
                else:
                    subFigure.plot(solver._x, 1.0 / solver._epsilon(solver._x, solver._parametersDict), label = labels[index])
                
                index += 1
        
        else:
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                subFigure.plot(self._x, self._epsilon(self._x, self._parametersDict), label = "mPB")
            else:
                subFigure.plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), label = "mPB")

        if showEpsilonBorn:
            epsilonInvBorn = (16 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO
                                            / (LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * (1.0 / self._parametersDict["rCation"] + 1.0 / self._parametersDict["rAnion"]))
                                            * (numpy.log(self._parametersDict["cSalt"])
                                               - numpy.log(numpy.sqrt(numpy.abs(self._rhoCationsMD) * numpy.abs(self._rhoAnionsMD))))
                              + 1.0 / self._parametersDict["epsilonR_bulkWater"])

            for i in range(0, len(epsilonInvBorn)):
                if epsilonInvBorn[i] == numpy.inf: epsilonInvBorn[i] = numpy.NaN # Do not plot values where the log is not defined
            
            subFigure.plot(self._x, epsilonInvBorn, label = "Born", linewidth = 0.8, linestyle = "dashed", color = "black")
        
        if otherSolvers != None:
            subFigure.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    def plotEpsilonInvIntegrated(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None) -> None:
        """
        
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Inverse of dielectric permittivity", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"$\frac{1}{l} \int_0^l~\frac{\mathrm{d}z}{\varepsilon_{\perp}^{\mathrm{mPB}}}$", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)
        
        if otherSolvers != None:
            subFigure.plot(self._x, self.getEpsilonInv_integrated(), label = labels[0])
            
            index = 1
            for solver in otherSolvers:
                subFigure.plot(solver._x, solver.getEpsilonInv_integrated(), label = labels[index])
        
        else:
            subFigure.plot(self._x, self.getEpsilonInv_integrated())
            
        if otherSolvers != None:
            subFigure.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotW(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None) -> None:
        """
        Plots the Born energy used to solve the modified Poisson-Boltzmann equation.
        
        Parameters
        ----------
        otherSolvers : list[MPBSolver]
            Another solvers to compare all epsilon curves.
        labels : list[str]
            List of the labels for the curves. Length must be len(otherSolvers) + 1. First index is for the current solver, other indexes follow otherSolvers order.
        """
        
        figure, subFigure_betaW = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Born energy", fontsize = SUPTITLE_FONTSIZE)
        subFigure_betaW.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure_betaW.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure_betaW.set_ylabel(r"$\beta W$", fontsize = AXES_FONTSIZE)
        subFigure_betaW.tick_params(labelsize = TICKS_FONTSIZE)
        
        subfigure_W = subFigure_betaW.twinx()
        subfigure_W.set_ylabel(r"$W$ (aJ $\approx$ 6.24 eV)", fontsize = AXES_FONTSIZE)
        subfigure_W.tick_params(labelsize = TICKS_FONTSIZE)
        lim1, lim2 = subFigure_betaW.get_ylim()
        subfigure_W.set_ylim(lim1 / LG_mPB_libs.mpb.physics_properties.beta(self._T), lim2 / LG_mPB_libs.mpb.physics_properties.beta(self._T))
        
        if otherSolvers != None:
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat), label = r"$W_{-}$ - " + labels[0])
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat), label = r"$W_{+}$ - " + labels[0])
            
            index = 1
            for solver in otherSolvers:
                if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                    subFigure_betaW.plot(solver._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(solver.x, solver._epsilon, solver._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, solver._epsilonFormat), label = r"$W_{-}$ - " + labels[index])
                if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                    subFigure_betaW.plot(solver._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(solver.x, solver._epsilon, solver._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, solver._epsilonFormat), label = r"$W_{+}$ - " + labels[index])
                
                index += 1
        
        else:
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat), label = r"$W_{-a}$")
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat), label = r"$W_{+}$")
        
        subFigure_betaW.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotTotalChargeIntegral(self) -> None:
        """
        Plots the cumulative integral of the total charge density in the system, along the z axis.
        """
        
        figure, subFigure = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Total charge integral", fontsize = SUPTITLE_FONTSIZE)
        subFigure.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure.set_ylabel(r"Total charge integral ($e$.nm$^{-2}$)", fontsize = AXES_FONTSIZE)
        subFigure.tick_params(labelsize = TICKS_FONTSIZE)

        integral = scipy.integrate.cumulative_trapezoid(self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat) + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), self._x, initial = 0)
        
        subFigure.plot(self._x, integral)
        subFigure.axhline(0 , color = "black", zorder = 0)
        
        subFigure.annotate(f"{integral[-1]:.5f}", (self._x[-1], integral[-1]), xytext=(self._x[-1], 0.02), bbox = dict(boxstyle = "square, pad = 0.3", fc = "white", lw = 1), fontsize = LEGEND_FONTSIZE)
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotPlasmaParameter(self, otherSolvers: list["MPBSolver"] = None, labels: list[str] = None) -> None:
        """
        Plots the plasma parameter value along the z axis.
        
        Parameters
        ----------
        otherSolvers : list[MPBSolver]
            Another solvers to compare all epsilon curves.
        labels : list[str]
            List of the labels for the curves. Length must be len(otherSolvers) + 1. First index is for the current solver, other indexes follow otherSolvers order.
        """
        
        figure, subFigure_gamma = matplotlib.pyplot.subplots(figsize = FIGSIZE)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Plasma parameter", fontsize = SUPTITLE_FONTSIZE)
        # subFigure_gamma.set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        
        subFigure_gamma.set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        subFigure_gamma.set_ylabel(r"$\Gamma$", fontsize = AXES_FONTSIZE)
        subFigure_gamma.tick_params(labelsize = TICKS_FONTSIZE)
        
        # Compute the average distance between the ions from the concentration
        if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
            d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
        elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
            d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
        else:
            d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
                          + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
        
        if otherSolvers != None:
            subFigure_gamma.plot(self._x, LG_mPB_libs.mpb.characteristic_lengths.bjerrumLength(self._x, self._T, self._epsilon, self._parametersDict, self._epsilonFormat) * d, linewidth = 2, label = labels[0])
            
            index = 1
            styles = ("solid", "dashdot", "dashed", "dotted", (0, (3, 5, 1, 5)))
            for solver in otherSolvers:
                if solver._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
                    d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(solver._x, solver._T, solver._epsilon, solver._parametersDict, solver._phi[0], solver._epsilonFormat))**(1.0 / 3.0)
                elif solver._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
                    d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoCations(solver._x, solver._T, solver._epsilon, solver._parametersDict, solver._phi[0], solver._epsilonFormat))**(1.0 / 3.0)
                else:
                    d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(solver._x, solver._T, solver._epsilon, solver._parametersDict, solver._phi[0], solver._epsilonFormat)
                                  + LG_mPB_libs.mpb.physics_properties.rhoCations(solver._x, solver._T, solver._epsilon, solver._parametersDict, solver._phi[0], solver._epsilonFormat))**(1.0 / 3.0)
                
                subFigure_gamma.plot(solver._x, LG_mPB_libs.mpb.characteristic_lengths.bjerrumLength(solver._x, solver._T, solver._epsilon, solver._parametersDict, solver._epsilonFormat) * d, linewidth = 2, linestyle = styles[index % len(styles)], label = labels[index])
                
                index += 1
        
        else:
            subFigure_gamma.plot(self._x, LG_mPB_libs.mpb.characteristic_lengths.bjerrumLength(self._x, self._T, self._epsilon, self._parametersDict, self._epsilonFormat) * d, linewidth = 2)
        
        subFigure_gamma.axhline(1, color = "black", linewidth = 1.2, linestyle = (0, (5, 10)))
        # subFigure_gamma.axhline(numpy.sqrt(6 / numpy.pi), color = "black", linewidth = 0.8, linestyle = (0, (5, 10)))
        subFigure_gamma.set_ylim(bottom = 0.0)
        
        # subFigure_xi = subFigure_gamma.secondary_yaxis("right", functions = (lambda gamma : 2 * numpy.pi * gamma**2, lambda xi : numpy.sqrt(xi / (2 * numpy.pi))))
        # subFigure_xi.set_ylabel(r"$\Xi$", fontsize = AXES_FONTSIZE, math_fontfamily = "cm")
        # subFigure_xi.tick_params(labelsize = TICKS_FONTSIZE)
        # subFigure_xi.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 4))
        
        if otherSolvers != None:
            subFigure_gamma.legend(prop = {"size" : LEGEND_FONTSIZE})
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def plotMultiple(self, electricPotential: bool = True, showInitialGuess: bool = False, epsilon: bool = True, epsilonInv: bool = False, showEpsilonBorn: bool = True, chargeDensities: bool = True, bornEnergy: bool = False, totalChargeIntegral: bool = False, plasmaParameter: bool = False) -> None:
        """
        Allows to have multiple plots from the other plot functions in subplots of the same figure.
        """
        
        nbSubplots = 0
        
        if electricPotential: nbSubplots += 1
        if epsilon: nbSubplots += 1
        if epsilonInv: nbSubplots += 1
        if chargeDensities: nbSubplots += 1
        if bornEnergy: nbSubplots += 1
        if totalChargeIntegral: nbSubplots += 1
        if plasmaParameter: nbSubplots += 1

        figure, subFigures = matplotlib.pyplot.subplots(nrows = nbSubplots, ncols = 1, figsize = FIGSIZE, sharex = True)
        # figure.suptitle(f"{self._plotsMainTitleRoot}: Plasma parameter", fontsize = SUPTITLE_FONTSIZE)
        subFigures[0].set_title(f"{self._plotsSubtitle}", fontsize = TITLE_FONTSIZE)
        subFigures[-1].set_xlabel(r"$z$ (nm)", fontsize = AXES_FONTSIZE)
        
        currentPlot = 0
        
        
        ######################
        # INVERSE OF EPSILON #
        ######################
        
        if epsilonInv:
            subFigures[currentPlot].set_ylabel(r"$\varepsilon_{\perp}^{-1}$", fontsize = AXES_FONTSIZE)
            subFigures[currentPlot].tick_params(labelsize = TICKS_FONTSIZE)
            subFigures[currentPlot].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 0.1))
        
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
                subFigures[currentPlot].plot(self._x, self._epsilon(self._x, self._parametersDict), color = "black", label = "mPB")
            else:
                subFigures[currentPlot].plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), color = "black", label = "mPB")

            if showEpsilonBorn:
                epsilonInvBorn = (16 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO
                                                / (LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * (1.0 / self._parametersDict["rCation"] + 1.0 / self._parametersDict["rAnion"]))
                                                * (numpy.log(self._parametersDict["cSalt"])
                                                   - numpy.log(numpy.sqrt(numpy.abs(self._rhoCationsMD) * numpy.abs(self._rhoAnionsMD))))
                                  + 1.0 / self._parametersDict["epsilonR_bulkWater"])
    
                for i in range(0, len(epsilonInvBorn)):
                    if epsilonInvBorn[i] == numpy.inf: epsilonInvBorn[i] = numpy.NaN # Do not plot values where the log is not defined
                
                subFigures[currentPlot].plot(self._x, epsilonInvBorn, label = "Born", linestyle = "dashed", color = "red")

                if electricPotential:
                        subFigures[currentPlot].legend(prop = {"size" : LEGEND_FONTSIZE - 6}, bbox_to_anchor = (1, 1))
                else:
                    subFigures[currentPlot].legend(prop = {"size" : LEGEND_FONTSIZE - 6})
            
            currentPlot += 1
        
        
        
        ###########
        # EPSILON #
        ###########
        
        if epsilon:
            if showEpsilonBorn:
                subFigures[currentPlot].set_ylabel(r"$\varepsilon_{\perp}$", fontsize = AXES_FONTSIZE)
            else:
                subFigures[currentPlot].set_ylabel(r"$\varepsilon_{\perp}^{\mathrm{mPB}}$", fontsize = AXES_FONTSIZE)
            
            subFigures[currentPlot].tick_params(labelsize = TICKS_FONTSIZE)
            subFigures[currentPlot].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 20))
        
            if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR:
                subFigures[currentPlot].plot(self._x, self._epsilon(self._x, self._parametersDict), color = "black", label = "mPB")
            else:
                subFigures[currentPlot].plot(self._x, 1.0 / self._epsilon(self._x, self._parametersDict), color = "black", label = "mPB")
        
            if showEpsilonBorn:
                epsilonInvBorn = (16 * numpy.pi * LG_mPB_libs.constants.EPSILON_ZERO
                                                / (LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E**2 * (1.0 / self._parametersDict["rCation"] + 1.0 / self._parametersDict["rAnion"]))
                                                * (numpy.log(self._parametersDict["cSalt"])
                                                   - numpy.log(numpy.sqrt(numpy.abs(self._rhoCationsMD) * numpy.abs(self._rhoAnionsMD))))
                                  + 1.0 / self._parametersDict["epsilonR_bulkWater"])
    
                for i in range(0, len(epsilonInvBorn)):
                    if epsilonInvBorn[i] == numpy.inf: epsilonInvBorn[i] = numpy.NaN # Do not plot values where the log is not defined
                
                subFigures[currentPlot].plot(self._x, 1.0 / epsilonInvBorn, label = "Born", linestyle = "dashed", color = "red")
            
                if electricPotential:
                    subFigures[currentPlot].legend(prop = {"size" : LEGEND_FONTSIZE - 6}, bbox_to_anchor = (1, 1))
                else:
                    subFigures[currentPlot].legend(prop = {"size" : LEGEND_FONTSIZE - 6})
            
            currentPlot += 1
        

        
        ######################
        # ELECTRIC POTENTIAL #
        ######################
        
        if electricPotential:
            subFigure_0_phi = subFigures[currentPlot]
            subFigure_0_phi.set_ylabel(r"$\phi$", fontsize = AXES_FONTSIZE)
            subFigure_0_phi.tick_params(labelsize = TICKS_FONTSIZE)

            if showInitialGuess:
                subFigure_0_phi.plot(self._x, self._phiInitGuess[0], label = "guess", color = "dimgrey", linestyle = ":", linewidth = 1)
            
            subFigure_0_phi.plot(self._x, self._phi[0], label = "numerical", color = "black")
            
            subFigure_0_phi.set_ylim(top = 0, bottom = -4)
            # subFigure_0_phi.set_ylim(top = -2.25, bottom = -3.3)
            subFigure_0_phi.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 1))
            # subFigure_0_phi.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 0.5))
            
            subFigure_0_V = subFigure_0_phi.twinx()
            subFigure_0_V.set_ylabel(r"$V$ (mV)", fontsize = AXES_FONTSIZE)
            subFigure_0_V.tick_params(labelsize = TICKS_FONTSIZE)
            lim1, lim2 = subFigure_0_phi.get_ylim()
            subFigure_0_V.set_ylim(lim1 / (1e15 * LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E), lim2 / (1e15 * LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.constants.E))
            subFigure_0_V.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 25))
            # subFigure_0_V.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 10))
            
            if showInitialGuess:
                subFigure_0_phi.legend(prop = {"size" : LEGEND_FONTSIZE})

            currentPlot += 1
        
        
        
        ####################
        # CHARGE DENSITIES #
        ####################
        
        if chargeDensities:
            subFigures[currentPlot].set_ylabel(r"$\rho$ ($e$.nm$^{-3}$)", fontsize = AXES_FONTSIZE)
            subFigures[currentPlot].tick_params(labelsize = TICKS_FONTSIZE)
            
            subFigures[currentPlot].axhline(0, color = "black", linewidth = 0.8)

            mdDotsFrequency = 7
            mdMarkerSize = 5
            
            
            #--------#
            # Anions #
            #--------#
            
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                # subFigures[currentPlot].plot(self._x, self._rhoAnionsMD, color = "springgreen", linestyle = (0, (5, 5)), linewidth = 3, zorder = 20)
                subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoAnionsMD[::mdDotsFrequency], color = "darkgreen", marker = "v", linewidth = 0, fillstyle = "none", markersize = mdMarkerSize, zorder = 25)
                subFigures[currentPlot].plot(self._x, LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{-}$", color = "limegreen", zorder = 20)
            
            
            #---------#
            # Cations #
            #---------#
            
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                # subFigures[currentPlot].plot(self._x, self._rhoCationsMD, color = "salmon", linestyle = (0, (5, 5)), linewidth = 3, zorder = 10)
                subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoCationsMD[::mdDotsFrequency], color = "darkred", marker = "^", linewidth = 0, fillstyle = "none", markersize = mdMarkerSize, zorder = 15)
                subFigures[currentPlot].plot(self._x, LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{+}$", color = "red", zorder = 10)

            
            #--------#
            # Lipids #
            #--------#
            
            # subFigures[currentPlot].plot(self._x, self._rhoLipidsMD, label = r"$\rho_{lipids}^{MD}$", color = "black", zorder = 0)
            subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoLipidsMD[::mdDotsFrequency], label = r"$\rho_{\mathrm{lipids}}$", color = "black", marker = "o", linewidth = 0, fillstyle = "none", markersize = mdMarkerSize, zorder = 0)

            
            #--------------#
            # Total charge #
            #--------------#
            
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
                # subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + self._rhoAnionsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3, zorder = 30)
                subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoLipidsMD[::mdDotsFrequency] + self._rhoAnionsMD[::mdDotsFrequency], color = "darkblue", marker = "s", linewidth = 0, fillstyle = "none", markersize = mdMarkerSize, zorder = 35)
                subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "blue", zorder = 30)
            elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
                # subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + self._rhoCationsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3, zorder = 30)
                subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoLipidsMD[::mdDotsFrequency] + self._rhoCationsMD[::mdDotsFrequency], color = "darkblue", marker = "s", linewidth = 0, markersize = mdMarkerSize, zorder = 35)
                subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "blue", zorder = 30)
            else:
                # subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + self._rhoAnionsMD + self._rhoCationsMD, color = "violet", linestyle = (0, (5, 5)), linewidth = 3, zorder = 30)
                subFigures[currentPlot].plot(self._x[::mdDotsFrequency], self._rhoLipidsMD[::mdDotsFrequency] + self._rhoAnionsMD[::mdDotsFrequency] + self._rhoCationsMD[::mdDotsFrequency], color = "darkblue", marker = "s", linewidth = 0, fillstyle = "none", markersize = mdMarkerSize, zorder = 35)
                subFigures[currentPlot].plot(self._x, self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat) + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), label = r"$\rho_{\mathrm{free}}$", color = "blue", zorder = 30)

            
            #---------------#
            # Manual legend #
            #---------------#

            legendLines = []
            legendLabels = []
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                legendLines.append(Line2D([0], [0], color="red", marker = "^", markersize = mdMarkerSize, markeredgecolor = "darkred", fillstyle = "none"))
                legendLabels.append(r"$\rho_{+}$")
            
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                legendLines.append(Line2D([0], [0], color="limegreen", marker = "v", markersize = mdMarkerSize, markeredgecolor = "darkgreen", fillstyle = "none"))
                legendLabels.append(r"$\rho_{-}$")
            
            legendLines.append(Line2D([0], [0], color="black", linewidth = 0, marker = "o", markersize = mdMarkerSize, fillstyle = "none"))
            legendLabels.append(r"$\rho_{\mathrm{lipids}}$")
            legendLines.append(Line2D([0], [0], color="blue", marker = "s", markersize = mdMarkerSize, markeredgecolor = "darkblue", fillstyle = "none"))
            legendLabels.append(r"$\rho_{\mathrm{free}}$")
            
            if electricPotential:
                subFigures[currentPlot].legend(legendLines, legendLabels, prop = {"size" : LEGEND_FONTSIZE}, bbox_to_anchor = (1, 0.5), loc = "center left")
            else:
                subFigures[currentPlot].legend(legendLines, legendLabels, prop = {"size" : LEGEND_FONTSIZE}, loc = "lower left") 
            
            currentPlot += 1



        ###############
        # BORN ENERGY #
        ###############
        
        if bornEnergy:
            subFigure_betaW = subFigures[currentPlot]
            subFigure_betaW.set_ylabel(r"$\beta W$", fontsize = AXES_FONTSIZE)
            subFigure_betaW.tick_params(labelsize = TICKS_FONTSIZE)
            
            subFigure_W = subFigure_betaW.twinx()
            subFigure_W.set_ylabel(r"$W$ (aJ $\approx$ 6.24 eV)", fontsize = AXES_FONTSIZE)
            subFigure_W.tick_params(labelsize = TICKS_FONTSIZE)
            lim1, lim2 = subFigure_betaW.get_ylim()
            subFigure_W.set_ylim(lim1 / LG_mPB_libs.mpb.physics_properties.beta(self._T), lim2 / LG_mPB_libs.mpb.physics_properties.beta(self._T))
            
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.ANION, self._epsilonFormat), label = r"$W_{-a}$")
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION or self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.SALT:
                subFigure_betaW.plot(self._x, LG_mPB_libs.mpb.physics_properties.beta(self._T) * LG_mPB_libs.mpb.physics_properties.W(self._x, self._epsilon, self._parametersDict, LG_mPB_libs.mpb.mpb_types.IonType.CATION, self._epsilonFormat), label = r"$W_{+}$")

            subFigure_betaW.legend(prop = {"size" : LEGEND_FONTSIZE})

            currentPlot += 1
        


        #########################
        # TOTAL CHARGE INTEGRAL #
        #########################
        
        if totalChargeIntegral:
            subFigures[currentPlot].set_ylabel(r"Total charge integral ($e$.nm$^{-2}$)", fontsize = AXES_FONTSIZE)
            subFigures[currentPlot].tick_params(labelsize = TICKS_FONTSIZE)

            integral = scipy.integrate.cumulative_trapezoid(self._rhoLipidsMD + LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat) + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat), self._x, initial = 0)
            
            subFigures[currentPlot].plot(self._x, integral)
            subFigures[currentPlot].axhline(0 , color = "black", zorder = 0)
            
            subFigures[currentPlot].annotate(f"{integral[-1]:.5f}", (self._x[-1], integral[-1]), xytext=(self._x[-1], 0.02), bbox = dict(boxstyle = "square, pad = 0.3", fc = "white", lw = 1), fontsize = LEGEND_FONTSIZE)

            subFigures[currentPlot].legend(prop = {"size" : LEGEND_FONTSIZE})
            
            currentPlot += 1
        
        
        
        ####################
        # PLASMA PARAMETER #
        ####################
        
        if plasmaParameter:
            subFigure_gamma = subFigures[currentPlot]
            subFigure_gamma.set_ylabel(r"$\Gamma$", fontsize = AXES_FONTSIZE)
            subFigure_gamma.tick_params(labelsize = TICKS_FONTSIZE)
            
            # Compute the average distance between the ions from the concentration
            if self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.ANION:
                d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
            elif self._systemIons == LG_mPB_libs.mpb.mpb_types.IonType.CATION:
                d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
            else:
                d = numpy.abs(LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
                              + LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat))**(1.0 / 3.0)
        
            subFigure_gamma.plot(self._x, LG_mPB_libs.mpb.characteristic_lengths.bjerrumLength(self._x, self._T, self._epsilon, self._parametersDict, self._epsilonFormat) * d)
            
            subFigure_xi = subFigure_gamma.twinx()
            subFigure_xi.set_ylabel(r"$\Xi$", fontsize = AXES_FONTSIZE, math_fontfamily = "cm")
            subFigure_xi.tick_params(labelsize = TICKS_FONTSIZE)
            lim1, lim2 = subFigure_gamma.get_ylim()
            subFigure_xi.set_ylim(2 * numpy.pi * lim1**2, 2 * numpy.pi * lim2**2)
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()
        
        return
    
    
    def getRhoAnions(self) -> numpy.ndarray:
        """
        Returns the array containing the charge density prediction for the anions.
        
        Returns
        -------
        numpy.ndarray
            The model prediction for the anionic charge density.
        """
        
        return LG_mPB_libs.mpb.physics_properties.rhoAnions(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
    
    
    def getRhoCations(self) -> numpy.ndarray:
        """
        Returns the array containing the charge density prediction for the cations.
        
        Returns
        -------
        numpy.ndarray
            The model prediction for the cationic charge density.
        """
        
        return LG_mPB_libs.mpb.physics_properties.rhoCations(self._x, self._T, self._epsilon, self._parametersDict, self._phi[0], self._epsilonFormat)
    
    
    def getEpsilonR_water(self) -> float:
        """
        Returns the value of the dielectric permittivity in the middle of the water channel (ie. the last value of the array).
        
        Returns
        -------
        float
            The value of the dielectric permittivity in the middle of the water channel.
        """
        
        return self._epsilon(self._x, self._parametersDict)[-1] if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.REGULAR else 1.0 / self._epsilon(self._x, self._parametersDict)[-1]
    
    
    def getEpsilon_integrated(self) -> numpy.ndarray:
        """
        
        """
        
        return 1.0 / self.getEpsilonInv_integrated()
    
    def getEpsilonInv_integrated(self) -> numpy.ndarray:
        """
        
        """
        
        if self._epsilonFormat == LG_mPB_libs.mpb.mpb_types.EpsilonFormat.INVERTED:
            epsilonInv = self._epsilon(self._x, self._parametersDict)
        else:
            epsilonInv = 1.0 / self._epsilon(self._x, self._parametersDict)

        epsilonInv_reverse = epsilonInv[::-1]
        x_reverse = self._x[::-1]
        
        integral = scipy.integrate.cumulative_trapezoid(epsilonInv_reverse, self._x, initial = 0)
        
        integral[1:] = integral[1:] / (self._x - self._x[0])[1:]
        integral = integral[::-1]
        
        return integral
