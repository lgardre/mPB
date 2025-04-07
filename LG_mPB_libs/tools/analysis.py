"""
This module contains tools to analyse MD simulations.
"""

import MDAnalysis
import numpy
import os
import subprocess
import sys
import types

from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import LG_mPB_libs.constants
import LG_mPB_libs.lipid



#############
# ENUM TYPE #
#############

class AnalysisType(Enum):
    """
    This Enum represents any analysis that can be computed by Gromacs gmx tool. For more details, refer to Gromacs documentation.
    """
    
    GMX_DENSITY_MASS = "mass"
    GMX_DENSITY_NUMBER = "number"
    GMX_DENSITY_CHARGE = "charge"
    GMX_DENSITY_ELECTRON = "electron"
    GMX_DENSITY_NEUTRON_SLD = "neutronSLD"
    GMX_DENSITY_XRAY_SLD = "XraySLD"
    GMX_DIPOLES = "dipoles"
    GMX_MSD = "MSD"



#######################
# TRAJECTORY ANALYSIS #
#######################

def generateDensityProfile(trajectoryFilePath: str, topologyFilePath: str, outputFileRoot: str, analysisType: AnalysisType, densityResidue: str, symmetrize: bool, sliceNumber: int, center: bool, indexFilePath: str = None, centerResidue: str = None, firstFrame: int = 0, lastFrame: int = 0, electronsDatafilePath: str = None, neutronSldDatafilePath: str = None, xRaySldDatafilePath: str = None, itpDirectory: str = None) -> bool:
    """
    Generates the XVG file that contains the requested density profile (along the z axis) of a given trajectory, using Gromacs gmx density tool.
    For more details, refer to Gromacs documentation. 
    
    Parameters
    ----------
    trajectoryFilePath : str
        The location of the file that contains the MD trajectory (either XTC or TRR).
    topologyFilePath : str
        The location of the TPR file that contains the portable binary run.
    outputFileRoot : str
        The location of the output file (without file extension).
    analysisType : AnalysisType
        The type of density to compute (should be a GMX_DENSITY_[xxx] type otherwise nothing will be done in this function).
    densityResidue : str
        The residue to compute the density for.
    symmetrize : bool
        Whether or not to symmetrize the density profile.
    sliceNumber : int
        Number of slice in the direction of the density analysis (z axis).
    center : bool
        Whether or not to center the density around the center of mass of a residue.
    indexFilePath : str
        The location of the NDX file that contains the groups atom IDs.
    centerResidue : str
        The name of the residue to center the density around.
    firstFrame: int
        The time of the first frame for the profile calculation, in ps.
    lastFrame: int
        The time of the last frame for the profile calculation, in ps. Put 0 for last frame.
    electronsDatafilePath : str
        The location of the DAT file that contains the information about the number of electrons per atom type.
    neutronSldDatafilePath: str
        The location of the DAT file that contains the information about the neutron scattering length densities per atom type.
    xRaySldDatafilePath: str
        The location of the DAT file that contains the information about the X-rays scattering length densities per atom type.
    itpDirectory : str
        The location of the ITP topology files.
    
    Returns
    -------
    bool
        False if the analysisType is not a density. Otherwise, the generation process status: True for success or False for error.
    """
    
    returnValue = True
    
    if analysisType not in [AnalysisType.GMX_DENSITY_MASS, AnalysisType.GMX_DENSITY_NUMBER, AnalysisType.GMX_DENSITY_CHARGE,
                            AnalysisType.GMX_DENSITY_ELECTRON, AnalysisType.GMX_DENSITY_NEUTRON_SLD, AnalysisType.GMX_DENSITY_XRAY_SLD]:
        returnValue = False
        
    else:
        
        # Creation of electronsDatafile if needed
        if analysisType == AnalysisType.GMX_DENSITY_ELECTRON and not os.path.exists(electronsDatafilePath):
            dictAtomsElectrons = {}

            for itpFile in os.listdir(itpDirectory):
                universe = MDAnalysis.Universe(f"{itpDirectory}/{itpFile}")

                for atom in universe.atoms:
                    if not atom.name in dictAtomsElectrons:
                        question = f"Number of electrons for atom {atom.name}:"
                        electronNumber = input(question)

                        while not electronNumber.isnumeric():
                            electronNumber = input(question)

                        dictAtomsElectrons[atom.name] = electronNumber

            with open (electronsDatafilePath, 'w') as electronDataFile:
                electronDataFile.write(f"{str(len(dictAtomsElectrons))}\n")
                for atom in dictAtomsElectrons:
                    electronDataFile.write(f"{atom} = {dictAtomsElectrons[atom]}\n")

        # Creation of neutronSldDatafile if needed
        if analysisType == AnalysisType.GMX_DENSITY_NEUTRON_SLD and not os.path.exists(neutronSldDatafilePath):
            dictAtomsNeutronSLD = {}

            for itpFile in os.listdir(itpDirectory):
                universe = MDAnalysis.Universe(f"{itpDirectory}/{itpFile}")

                for atom in universe.atoms:
                    if not atom.name in dictAtomsNeutronSLD:
                        question = f"Neutron Scattering Length Density for atom {atom.name}:"
                        neutronSLD = input(question)

                        while not neutronSLD.isnumeric():
                            neutronSLD = input(question)

                        dictAtomsNeutronSLD[atom.name] = neutronSLD

            with open (neutronSldDatafilePath, 'w') as neutronSldDatafile:
                neutronSldDatafile.write(f"{str(len(dictAtomsNeutronSLD))}\n")
                for atom in dictAtomsNeutronSLD:
                    neutronSldDatafile.write(f"{atom} = {dictAtomsNeutronSLD[atom]}\n")

        # Creation of xRaySldDatafile if needed
        if analysisType == AnalysisType.GMX_DENSITY_XRAY_SLD and not os.path.exists(xRaySldDatafilePath):
            dictAtomsXraySLD = {}

            for itpFile in os.listdir(itpDirectory):
                universe = MDAnalysis.Universe(f"{itpDirectory}/{itpFile}")

                for atom in universe.atoms:
                    if not atom.name in dictAtomsXraySLD:
                        question = f"Number of electrons for atom {atom.name}:"
                        xRaySLD = input(question)

                        while not xRaySLD.isnumeric():
                            xRaySLD = input(question)

                        dictAtomsXraySLD[atom.name] = int(xRaySLD) * 2.81

            with open (xRaySldDatafilePath, 'w') as xRaySldDatafile:
                xRaySldDatafile.write(f"{str(len(dictAtomsXraySLD))}\n")
                for atom in dictAtomsXraySLD:
                    xRaySldDatafile.write(f"{atom} = {dictAtomsXraySLD[atom]}\n")

        gmxInput = densityResidue if center == False else f"{centerResidue} {densityResidue}"
        params = f"-f {trajectoryFilePath} -s {topologyFilePath} -o {outputFileRoot}.xvg -xvg none -sl {sliceNumber}"


        if analysisType == AnalysisType.GMX_DENSITY_NEUTRON_SLD:
            params = f"{params} -dens {AnalysisType.GMX_DENSITY_ELECTRON.value} -ei {neutronSldDatafilePath}"
        elif analysisType == AnalysisType.GMX_DENSITY_XRAY_SLD:
            params = f"{params} -dens {AnalysisType.GMX_DENSITY_ELECTRON.value} -ei {xRaySldDatafilePath}"
        elif analysisType == AnalysisType.GMX_DENSITY_ELECTRON:
            params = f"{params} -dens {analysisType.value} -ei {electronsDatafilePath}"
        else:
            params = f"{params} -dens {analysisType.value}"

        if center:
            params = f"{params} -center"

        if symmetrize:
            params = f"{params} -symm"

        if indexFilePath is not None:
            params = f"{params} -n {indexFilePath}"

        if firstFrame != 0:
            params = f"{params} -b {firstFrame}"

        if lastFrame != 0:
            params = f"{params} -e {lastFrame}"
        
        returnValue = subprocess.run(f"echo \"{gmxInput}\" | gmx density {params}", shell = True) == 0
    
    return returnValue


def generateDipolesAnalysis(trajectoryFilePath: str, topologyFilePath: str, outputFileRoot: str, residue: str, temperature: float, sliceNumber: int, indexFilePath: str = None, firstFrame: int = 0, lastFrame: int = 0) -> bool:
    """
    Generates the XVG files that contain the dipoles analysis of a given trajectory, using Gromacs gmx dipoles tool.
    For more details, refer to Gromacs documentation. 
    
    Parameters
    ----------
    trajectoryFilePath : str
        The location of the file that contains the MD trajectory (either XTC or TRR).
    topologyFilePath : str
        The location of the TPR file that contains the portable binary run.
    outputFileRoot : str
        The location of the output files (without file extension).
    residue : str
        The residue to compute the dipoles for.
    temperature : float
        The system temperature (usefull to get the dielectric constant value).
    sliceNumber : int
        Number of slice in the direction of the density analysis (z axis).
    indexFilePath : str
        The location of the NDX file that contains the groups id.
    firstFrame: int
        The time of the first frame for the profile calculation, in ps.
    lastFrame: int
        The time of the last frame for the profile calculation, in ps. Put 0 for last frame.
    
    Returns
    -------
    bool
        The generation process status: True for success or False for error.
    """
    
    params = f"-f {trajectoryFilePath} -s {topologyFilePath} -temp {temperature} -sl {sliceNumber} -o {outputFileRoot}_Mtot.xvg -eps {outputFileRoot}_epsilon.xvg -a {outputFileRoot}_aver.xvg -d {outputFileRoot}_dipdist.xvg -xvg none"

    if indexFilePath is not None:
        params = f"{params} -n {indexFilePath}"

    if firstFrame != 0:
        params = f"{params} -b {firstFrame}"

    if lastFrame != 0:
        params = f"{params} -e {lastFrame}"
    
    return subprocess.run(f"echo \"{residue}\" | gmx dipoles {params} | grep \"Epsilon =\" | awk '{{printf(\"%f\", $3)}}' > {outputFileRoot}_epsilonR.txt", shell = True) == 0


def generateMSDAnalysis(trajectoryFilePath: str, topologyFilePath: str, outputFileRoot: str, residue: str, dt: int = 0, indexFilePath: str = None, firstFrame: int = 0, lastFrame: int = 0) -> bool:
    """
    Generates the XVG files that contain the dipoles analysis of a given trajectory, using Gromacs gmx dipoles tool.
    For more details, refer to Gromacs documentation. 
    
    Parameters
    ----------
    trajectoryFilePath : str
        The location of the file that contains the MD trajectory (either XTC or TRR).
    topologyFilePath : str
        The location of the TPR file that contains the portable binary run.
    outputFileRoot : str
        The location of the output files (without file extension).
    residue : str
        The residue to compute the dipoles for.
    dt : int
        The time between each frame to use (if 0: use all frames).
    sliceNumber : int
        Number of slice in the direction of the density analysis (z axis).
    indexFilePath : str
        The location of the NDX file that contains the groups id.
    firstFrame: int
        The time of the first frame for the profile calculation, in ps.
    lastFrame: int
        The time of the last frame for the profile calculation, in ps. Put 0 for last frame.
    
    Returns
    -------
    bool
        The generation process status: True for success or False for error.
    """
    
    params = f"-f {trajectoryFilePath} -s {topologyFilePath} -o {outputFileRoot}.xvg -lateral z -xvg none"

    if indexFilePath is not None:
        params = f"{params} -n {indexFilePath}"
    
    if dt != 0:
        params = f"{params} -dt {dt}"
    
    if firstFrame != 0:
        params = f"{params} -b {firstFrame}"

    if lastFrame != 0:
        params = f"{params} -e {lastFrame}"
    
    return subprocess.run(f"echo \"{residue}\" | gmx msd {params}", shell = True) == 0


def generateOrderParameter(trajectoryFilePath: str, topologyFilePath: str, outputDirectoryPath: str, orderParameterAtomPairsDatafilePath: str = None, headAtomsDatafilePath: str = None, tailsAtomsDatafilePath: str = None, lipidItpFilePath: str = None, step: int = 1) -> bool:
    """
    Generates the XVG file(s) that contain(s) the requested order parameter(s).
    
    Parameters
    ----------
    trajectoryFilePath : str
        The location of the file that contains the MD trajectory (either XTC or TRR).
    topologyFilePath : str
        The location of the TPR file that contains the portable binary run.
    outputDirectoryPath : str
        The location of the directory where the output XVG file(s) will be writen.
    orderParameterAtomPairsDatafilePath : str
        The location of the file that defines which order parameters will be computed.
    headAtomsDatafilePath : str
        The location of the file that defines the lipid heads atoms.
    tailsAtomsDatafilePath : str
        The location of the file that defines the lipid tails atoms.
    lipidItpFilePath : str
        The location of the ITP file that contains the system topology.
    step : int
        The step for the calculation of thje order parameter(s).
    
    Returns
    -------
    bool
        The generation process status: True for success or False for error.
    """
    
    # Datafile path creation if not given in parameters
    if orderParameterAtomPairsDatafilePath is None:
        orderParameterAtomPairsDatafilePath = f"{outputDirectoryPath}/orderParameters.dat"
    
    if headAtomsDatafilePath is None:
        headAtomsDatafilePath = f"{outputDirectoryPath}/headAtoms.dat"
    
    if tailsAtomsDatafilePath is None:
        tailsAtomsDatafilePath = f"{outputDirectoryPath}/tailsAtoms.dat"
    
    success = True
    pairs = []
    headAtoms = []
    tailsAtoms = []
    usefullAtoms = []
    
    if not os.path.exists(trajectoryFilePath):
        success = False
        print(f"Error: {trajectoryFilePath} not found. This file is mandatory, the program will quit.")
    
    elif not os.path.exists(topologyFilePath):
        success = False
        print(f"Error: {topologyFilePath} not found. This file is mandatory, the program will quit.")
    
    else:
        # Creation of orderParameterAtomPairsDatafile if needed
        if not os.path.exists(orderParameterAtomPairsDatafilePath):
            if os.path.exists(lipidItpFilePath):
                print(f"{orderParameterAtomPairsDatafilePath} not found. Let's create it using the lipid ITP file.")
                
                atomNames = []
                universe = MDAnalysis.Universe(lipidItpFilePath)
                
                atomListStr = f"\nAtom listed in {lipidItpFilePath}:\n"
                for i in range(1, len(universe.atoms) + 1):
                    atomNames.append(universe.atoms[i - 1].name)
                    atomListStr = f"{atomListStr}{i}. {universe.atoms[i - 1].name}\n"
                atomListStr = f"{atomListStr}\n"
                atomListStr = f"{atomListStr}{len(universe.atoms) + 1}. Head geometrical center (defined from {headAtomsDatafilePath}) file)\n"
                atomListStr = f"{atomListStr}{len(universe.atoms) + 2}. Tails geometrical center (defined from {tailsAtomsDatafilePath}) file)\n"
                
                helpStr = "Enter ""quit"" to end the process and save the atom pairs. Enter ""list"" to print the list of atoms. Enter ""help"" to print this text again.\n"
                
                print(atomListStr)
                print(helpStr)
                
                pairNumber = 1
                while (True):
                    atom1 = ""
                    atom2 = ""
                    
                    while not ((atom1.isnumeric() and int(atom1) >= 1 and int(atom1) <= len(universe.atoms)) or atom1 in atomNames or atom1 == "quit"):
                        atom1 = input(f"Select the 1st atom of the pair #{pairNumber} from the atom list (1-{len(universe.atoms) + 2} or name):")
                        
                        if atom1.isnumeric() and (int(atom1) < 1 or int(atom1) > len(universe.atoms) + 2):
                            print(f"Error: index {atom1} does not exist in the atom list.")
                        elif atom1 == "quit":
                            break
                        elif atom1 == "list":
                            print(atomListStr)
                        elif atom1 == "help":
                            print(helpStr)
                        elif not atom1.isnumeric() and atom1 not in atomNames:
                            print(f"Error: there is no atom named {atom1} in the atom list.")
                    
                    if atom1 == "quit":
                        break
                    
                    if atom1.isnumeric():
                        if int(atom1) == len(universe.atoms) + 1:
                            atom1 = "$$HEAD$$"
                        elif int(atom1) == len(universe.atoms) + 2:
                            atom1 = "$$TAILS$$"
                        else:
                            atom1 = universe.atoms[int(atom1) - 1].name
                    
                    while not (atom2 != atom1 and (atom2.isnumeric() and int(atom2) >= 1 and int(atom2) <= len(universe.atoms)) or atom2 in atomNames or atom2 == "quit"):
                        atom2 = input(f"Select the 2nd atom of the pair #{pairNumber} from the atom list (1-{len(universe.atoms) + 2} or name):")
                        
                        if atom2.isnumeric() and (int(atom2) < 1 or int(atom2) > len(universe.atoms) + 2):
                            print(f"Error: index {atom2} does not exist in the atom list.")
                        elif atom2 == "quit":
                            break
                        elif atom2 == "list":
                            print(atomListStr)
                        elif atom2 == "help":
                            print(helpStr)
                        elif not atom2.isnumeric() and atom2 not in atomNames:
                            print(f"Error: there is no atom named {atom2} in the atom list.")
                        
                        if atom2.isnumeric():
                            if int(atom2) == len(universe.atoms) + 1:
                                atom2 = "$$HEAD$$"
                            elif int(atom2) == len(universe.atoms) + 2:
                                atom2 = "$$TAILS$$"
                            else:
                                atom2 = universe.atoms[int(atom2) - 1].name
                            
                        if atom2 == atom1:
                            print("Error: atom #2 cannot be the same as atom #1.")
                    
                    if atom2 == "quit":
                        break
                    
                    pairs.append([atom1, atom2])
                    pairNumber += 1
                    
                    if atom1 not in usefullAtoms:
                        usefullAtoms.append(atom1)
                    
                    if atom2 not in usefullAtoms:
                        usefullAtoms.append(atom2)
                
                with open(orderParameterAtomPairsDatafilePath, 'w') as orderParameterAtomPairsDatafile:
                    for pair in pairs:
                        orderParameterAtomPairsDatafile.write(f"{pair[0]} {pair[1]}\n")
                    
                    print(f"File {orderParameterAtomPairsDatafilePath} written.")
            
            else:
                print(f"Error: {orderParameterAtomPairsDatafilePath} and {lipidItpFilePath} not found. This program will quit.")
                success = False
        
        # Creation of headAtomsDatafile if needed
        if not os.path.exists(headAtomsDatafilePath):
            if os.path.exists(lipidItpFilePath):
                print(f"{headAtomsDatafilePath} not found. Let's create it using the lipid ITP file.")
                
                atomNames = []
                universe = MDAnalysis.Universe(lipidItpFilePath)
                
                atomListStr = f"\nAtom listed in {lipidItpFilePath}:\n"
                for i in range(1, len(universe.atoms) + 1):
                    atomNames.append(universe.atoms[i - 1].name)
                    atomListStr = f"{atomListStr}{i}. {universe.atoms[i - 1].name}\n"
                atomListStr = f"{atomListStr}\n"
                
                helpStr = "Enter ""quit"" to end the process and save the lipid head atoms. Enter ""list"" to print the list of atoms. Enter ""help"" to print this text again.\n"
                
                print(atomListStr)
                print(helpStr)
                
                number = 1
                while (True):
                    atom = ""
                    
                    while not ((atom.isnumeric() and int(atom) >= 1 and int(atom) <= len(universe.atoms)) or atom in atomNames or atom == "quit"):
                        atom = input(f"Select the atom #{number} from the atom list (1-{len(universe.atoms)} or name):")
                        
                        if atom.isnumeric() and (int(atom) < 1 or int(atom) > len(universe.atoms)):
                            print(f"Error: index {atom} does not exist in the atom list.")
                        elif atom == "quit":
                            break
                        elif atom == "list":
                            print(atomListStr)
                        elif atom == "help":
                            print(helpStr)
                        elif not atom.isnumeric() and atom not in atomNames:
                            print(f"Error: there is no atom named {atom} in the atom list.")
                    
                    if atom == "quit":
                        break
                    
                    if atom.isnumeric():
                        atom = universe.atoms[int(atom) - 1].name
                    
                    if atom not in headAtoms:
                        headAtoms.append(atom)
                        number += 1
                    else:
                        print(f"Atom {atom} is already present in the lipid head atom list, it will be ignored.")
                
                with open(headAtomsDatafilePath, 'w') as headAtomsDatafile:
                    for atom in headAtoms:
                        headAtomsDatafile.write(f"{atom}\n")
                    
                    print(f"File {headAtomsDatafilePath} written.")
            
            else:
                print(f"Error: {headAtomsDatafilePath} and {lipidItpFilePath} not found. This program will quit.")
                success = False
        
        # Creation of tailsAtomsDatafile if needed
        if not os.path.exists(tailsAtomsDatafilePath):
            if os.path.exists(lipidItpFilePath):
                print(f"{tailsAtomsDatafilePath} not found. Let's create it using the lipid ITP file.")
                
                atomNames = []
                universe = MDAnalysis.Universe(lipidItpFilePath)
                
                atomListStr = f"\nAtom listed in {lipidItpFilePath}:\n"
                for i in range(1, len(universe.atoms) + 1):
                    atomNames.append(universe.atoms[i - 1].name)
                    atomListStr = f"{atomListStr}{i}. {universe.atoms[i - 1].name}\n"
                atomListStr = f"{atomListStr}\n"
                
                helpStr = "Enter ""quit"" to end the process and save the lipid tails atoms. Enter ""list"" to print the list of atoms. Enter ""help"" to print this text again.\n"
                
                print(atomListStr)
                print(helpStr)
                
                number = 1
                while (True):
                    atom = ""
                    
                    while not ((atom.isnumeric() and int(atom) >= 1 and int(atom) <= len(universe.atoms)) or atom in atomNames or atom == "quit"):
                        atom = input(f"Select the atom #{number} from the atom list (1-{len(universe.atoms)} or name):")
                        
                        if atom.isnumeric() and (int(atom) < 1 or int(atom) > len(universe.atoms)):
                            print(f"Error: index {atom} does not exist in the atom list.")
                        elif atom == "quit":
                            break
                        elif atom == "list":
                            print(atomListStr)
                        elif atom == "help":
                            print(helpStr)
                        elif not atom.isnumeric() and atom not in atomNames:
                            print(f"Error: there is no atom named {atom} in the atom list.")
                    
                    if atom == "quit":
                        break
                    
                    if atom.isnumeric():
                        atom = universe.atoms[int(atom) - 1].name
                    
                    if atom not in tailsAtoms:
                        tailsAtoms.append(atom)
                        number += 1
                    else:
                        print(f"Atom {atom} is already present in the lipid tails atom list, it will be ignored.")
                
                with open(tailsAtomsDatafilePath, 'w') as tailsAtomsDatafile:
                    for atom in headAtoms:
                        tailsAtomsDatafile.write(f"{atom}\n")
                    
                    print(f"File {tailsAtomsDatafilePath} written.")
            
            else:
                print(f"Error: {tailsAtomsDatafilePath} and {lipidItpFilePath} not found. This program will quit.")
                success = False
        
        if success:
            if len(pairs) == 0:
                with open(orderParameterAtomPairsDatafilePath, 'r') as orderParameterAtomPairsDatafile:
                    for line in orderParameterAtomPairsDatafile.readlines():
                        pair = line.split()
                        atom1 = pair[0]
                        atom2 = pair[1]
                        
                        pairs.append([atom1, atom2])
                        
                        if atom1 not in usefullAtoms:
                            usefullAtoms.append(atom1)
                        if atom2 not in usefullAtoms:
                            usefullAtoms.append(atom2)
            
            if len(headAtoms) == 0:
                with open(headAtomsDatafilePath, 'r') as headAtomsDatafile:
                    for line in headAtomsDatafile.readlines():
                        headAtoms.append(line.strip())
            
            if len(tailsAtoms) == 0:
                with open(tailsAtomsDatafilePath, 'r') as tailsAtomDatafile:
                    for line in tailsAtomDatafile.readlines():
                        tailsAtoms.append(line.strip())
            
            lipids = []
            
            universe = MDAnalysis.Universe(topologyFilePath)
            for atom in universe.atoms:
                if atom.name in usefullAtoms or atom.name in headAtoms or atom.name in tailsAtoms:
                    searchResult = list(lipid for lipid in lipids if lipid._resid == atom.resid)
                    if len(searchResult) == 0:
                        lipid = Lipid(atom.resid)
                        lipids.append(lipid)
                        searchResult.append(lipid)
                    
                    searchResult[0]._atoms[atom.name] = atom.id
            
            
            trajectory = MDAnalysis.coordinates.XTC.XTCReader(trajectoryFilePath)
            
            flipflopCount = 0
            flipflops = {}
            searchForFlipflops = True
            
            for pair in pairs:
                pair0 = pair[0]
                if pair[0] == "$$HEAD$$":
                    pair0 = "HEAD"
                elif pair[0] == "$$TAILS$$":
                    pair0 = "TAILS"
                
                pair1 = pair[1]
                if pair[1] == "$$HEAD$$":
                    pair1 = "HEAD"
                elif pair[1] == "$$TAILS$$":
                    pair1 = "TAILS"
                
                pairStr = f"{pair0}-{pair1}"
                
                with open(f"{outputDirectoryPath}/orderParameter_{pairStr}.xvg", 'w') as orderParameterOutputFile:
                    orderParameterOutputFile.write(f"#{'{: >14}'.format('Time')}   LAYER    {'{: >9}'.format('OrderNrml')}    {'{: >9}'.format('MeanVectX')}    {'{: >9}'.format('MeanVectY')}    {'{: >9}'.format('MeanVectZ')}    {'{: >9}'.format('cosAngle')}    {'{: >9}'.format('OrderMean')}\n")
                    
                    for frame in trajectory[::step]:
                        headAtoms_meanZ = 0.0

                        tailsToHead_meanVector_unit = { LayerLocation.BOTTOM: numpy.asarray([0.0, 0.0, 0.0]),
                                                        LayerLocation.TOP: numpy.asarray([0.0, 0.0, 0.0]),
                                                        LayerLocation.MONOLAYER: numpy.asarray([0.0, 0.0, 0.0]) }

                        meanOrderParameter_normalVector = { LayerLocation.BOTTOM: 0.0,
                                                            LayerLocation.TOP: 0.0,
                                                            LayerLocation.MONOLAYER: 0.0 }

                        meanOrderParameter_LipidLayerMeanVector = { LayerLocation.BOTTOM: 0.0,
                                                                    LayerLocation.TOP: 0.0,
                                                                    LayerLocation.MONOLAYER: 0.0 }

                        meanCosAngle_LipidLayerMeanVector_LayerNormalVector = { LayerLocation.BOTTOM: 0.0,
                                                                                LayerLocation.TOP: 0.0,
                                                                                LayerLocation.MONOLAYER: 0.0 }

                        layerCount = { LayerLocation.BOTTOM: 0,
                                       LayerLocation.TOP: 0,
                                       LayerLocation.MONOLAYER: 0 }

                        # First loop to compute all lipid heads mean z position
                        for lipid in lipids:
                            lipid._headPosition = numpy.asarray([0.0, 0.0, 0.0])

                            for headAtom in headAtoms:
                                lipid._headPosition += numpy.asarray(frame.positions[ lipid._atoms[headAtom] ])

                            lipid._headPosition /= len(headAtoms)
                            headAtoms_meanZ += lipid._headPosition[2]

                            lipid._tailsPosition = numpy.asarray([0.0, 0.0, 0.0])
                            for tailAtom in tailsAtoms:
                                lipid._tailsPosition += numpy.asarray(frame.positions[ lipid._atoms[tailAtom] ])

                            lipid._tailsPosition /= len(tailsAtoms)

                            lipid._tailsToHeadVector = lipid._headPosition - lipid._tailsPosition

                        headAtoms_meanZ /= len(lipids)

                        # Second loop to:
                        #    - assign to each lipid its lipid layer
                        #    - compute the mean "tail to head vector" for each lipid layer
                        for lipid in lipids:
                            flipflopDetected = False

                            if numpy.dot(lipid._tailsToHeadVector, numpy.asarray([0.0, 0.0, 1.0])) <= 0:
                                if lipid._headPosition[2] <= headAtoms_meanZ:
                                    # Flip-flop detection
                                    if searchForFlipflops and lipid._location is not None and lipid._location != LayerLocation.BOTTOM:
                                        flipflopDetected = True

                                    lipid._location = LayerLocation.BOTTOM
                                    layerCount[LayerLocation.BOTTOM] += 1
                                    tailsToHead_meanVector_unit[LayerLocation.BOTTOM] += lipid._tailsToHeadVector
                                else:
                                    # Flip-flop detection
                                    if searchForFlipflops and lipid._location is not None and lipid._location != LayerLocation.MONOLAYER:
                                        flipflopDetected = True

                                    lipid._location = LayerLocation.MONOLAYER
                                    layerCount[LayerLocation.MONOLAYER] += 1
                                    tailsToHead_meanVector_unit[LayerLocation.MONOLAYER] += lipid._tailsToHeadVector
                            else:
                                # Flip-flop detection
                                if searchForFlipflops and lipid._location is not None and lipid._location != LayerLocation.TOP:
                                    flipflopDetected = True

                                lipid._location = LayerLocation.TOP
                                layerCount[LayerLocation.TOP] += 1
                                tailsToHead_meanVector_unit[LayerLocation.TOP] += lipid._tailsToHeadVector

                            if flipflopDetected:
                                if not frame.time in flipflops:
                                    flipflops[frame.time] = []
                                
                                flipflops[frame.time].append(lipid._resid)
                                flipflopCount += 1

                        tailsToHead_meanVector_unit[LayerLocation.BOTTOM] /= layerCount[LayerLocation.BOTTOM]
                        norme = numpy.linalg.norm(tailsToHead_meanVector_unit[LayerLocation.BOTTOM])
                        if norme != 0: tailsToHead_meanVector_unit[LayerLocation.BOTTOM] /= norme

                        tailsToHead_meanVector_unit[LayerLocation.TOP] /= layerCount[LayerLocation.TOP]
                        norme = numpy.linalg.norm(tailsToHead_meanVector_unit[LayerLocation.TOP])
                        if norme != 0: tailsToHead_meanVector_unit[LayerLocation.TOP] /= norme

                        tailsToHead_meanVector_unit[LayerLocation.MONOLAYER] /= 1
                        norme = numpy.linalg.norm(tailsToHead_meanVector_unit[LayerLocation.MONOLAYER])
                        if norme != 0: tailsToHead_meanVector_unit[LayerLocation.MONOLAYER] /= norme
                        
                        #Third loop to compute all chosen order parameters, with respect to the normal to the layer and the mean "tail to head vector" in the layer
                        for lipid in lipids:
                            position1 = lipid._headPosition if pair[0] == "$$HEAD$$" else lipid._tailsPosition if pair[0] == "$$TAILS$$" else numpy.asarray(frame.positions[ lipid._atoms[pair[0]] ])
                            position2 = lipid._headPosition if pair[1] == "$$HEAD$$" else lipid._tailsPosition if pair[1] == "$$TAILS$$" else numpy.asarray(frame.positions[ lipid._atoms[pair[1]] ])

                            vector_unit = (position2 - position1) / numpy.linalg.norm(position2 - position1)
                            normalVector_unit = numpy.asarray([0.0, 0.0, -1.0]) if lipid._location == LayerLocation.BOTTOM or lipid._location == LayerLocation.MONOLAYER else numpy.asarray([0.0, 0.0, 1.0])

                            cos_angleNormalVector = numpy.dot(vector_unit, normalVector_unit)
                            meanOrderParameter_normalVector[lipid._location] += (3.0 * cos_angleNormalVector**2 - 1)

                            cos_angleMeanLipidAngleLayer = numpy.dot(vector_unit, tailsToHead_meanVector_unit[lipid._location])
                            meanOrderParameter_LipidLayerMeanVector[lipid._location] += (3.0 * cos_angleMeanLipidAngleLayer**2 - 1)

                        if layerCount[LayerLocation.BOTTOM] != 0: meanOrderParameter_normalVector[LayerLocation.BOTTOM] /= (2 * layerCount[LayerLocation.BOTTOM])
                        if layerCount[LayerLocation.TOP] != 0: meanOrderParameter_normalVector[LayerLocation.TOP] /= (2 * layerCount[LayerLocation.TOP])
                        if layerCount[LayerLocation.MONOLAYER] != 0: meanOrderParameter_normalVector[LayerLocation.MONOLAYER] /= (2 * layerCount[LayerLocation.MONOLAYER])
                        
                        if layerCount[LayerLocation.BOTTOM] != 0: meanOrderParameter_LipidLayerMeanVector[LayerLocation.BOTTOM] /= (2 * layerCount[LayerLocation.BOTTOM])
                        if layerCount[LayerLocation.TOP] != 0: meanOrderParameter_LipidLayerMeanVector[LayerLocation.TOP] /= (2 * layerCount[LayerLocation.TOP])
                        if layerCount[LayerLocation.MONOLAYER] != 0: meanOrderParameter_LipidLayerMeanVector[LayerLocation.MONOLAYER] /= (2 * layerCount[LayerLocation.MONOLAYER])

                        orderParameterOutputFile.write(f"{'{: >15.6f}'.format(frame.time)}    MONO    {'{: >9.6f}'.format(meanOrderParameter_normalVector[LayerLocation.MONOLAYER])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.MONOLAYER][0])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.MONOLAYER][1])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.MONOLAYER][2])}    {'{: >9.6f}'.format(numpy.dot(numpy.asarray([0.0, 0.0, -1.0]), tailsToHead_meanVector_unit[LayerLocation.MONOLAYER]))}    {'{: >9.6f}'.format(meanOrderParameter_LipidLayerMeanVector[LayerLocation.MONOLAYER])}\n")
                        
                        orderParameterOutputFile.write(f"{'{: >15.6f}'.format(frame.time)}     TOP    {'{: >9.6f}'.format(meanOrderParameter_normalVector[LayerLocation.TOP])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.TOP][0])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.TOP][1])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.TOP][2])}    {'{: >9.6f}'.format(numpy.dot(numpy.asarray([0.0, 0.0, 1.0]), tailsToHead_meanVector_unit[LayerLocation.TOP]))}    {'{: >9.6f}'.format(meanOrderParameter_LipidLayerMeanVector[LayerLocation.TOP])}\n")
                        
                        orderParameterOutputFile.write(f"{'{: >15.6f}'.format(frame.time)}     BOT    {'{: >9.6f}'.format(meanOrderParameter_normalVector[LayerLocation.BOTTOM])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.BOTTOM][0])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.BOTTOM][1])}    {'{: >9.6f}'.format(tailsToHead_meanVector_unit[LayerLocation.BOTTOM][2])}    {'{: >9.6f}'.format(numpy.dot(numpy.asarray([0.0, 0.0, -1.0]), tailsToHead_meanVector_unit[LayerLocation.BOTTOM]))}    {'{: >9.6f}'.format(meanOrderParameter_LipidLayerMeanVector[LayerLocation.BOTTOM])}\n")
            
            searchForFlipflops = False
                
            if flipflopCount != 0:
                with open(f"{outputDirectoryPath}/flipflops.xvg", 'w') as flipflopsOutputFile:
                    for frame in flipflops:
                        for resid in flipflops[frame]:
                            flipflopsOutputFile.write(f"{'{: >15.6f}'.format(frame)}    {'{: >5}'.format(resid)}\n")
                    
                    print(f"{flipflopCount} flip-flop(s) detected. File flipflops.xvg written.")
    
    return success


def generateIndex_byAtomName(topologyFilePath: str):
    univers = mda.Universe(topologyFilePath)
    lstNames = univers.select_atoms("all").names

    with mda.selections.gromacs.SelectionWriter("atomsByName.ndx", mode = 'w') as ndx:
        for atomName in lstNamesUnique:
            atoms = univers.select_atoms(f"name {atomName}")
            ndx.write(atoms, name = atomName)

    return
