import os
import sys
import glob
import numpy as np
import time
import json
import SlaterDensity as sd
import CoordFileHelper as cfh
quadBatch = 1000000
ANGS_TO_AU = 1.8897259885789

## ----------------------------- Main function runs from here --------------------------------------
inpFileName = str(sys.argv[1])
inp_dict = {}
with open(inpFileName,'r') as f:
    inpDict = json.load(f)

initDir = os.getcwd()
sysdObjir = inpDict['sys_dir']
os.chdir(sysdObjir)
DMFiles = inpDict['DM_files']
coordFile = inpDict['coords_file']
atomicCoordUnit = inpDict['coords_unit']
quadFile = inpDict['quad_file']
quadUnit = inpDict['quad_unit']
outFilenamePrefix = inpDict['out_file']

# read quadrature data
quad = np.loadtxt(quadFile, dtype = np.float64, usecols=(0,1,2,3))
quadConversionFactor = cfh.getFactor(inUnit = quadUnit, outUnit = 'bohr')
quad[:,0:4] = quadConversionFactor*quad[:,0:4]

SMatrixOutFname = "SMatrix_evaluated"
SMatrixOutF = open(SMatrixOutFname, 'w')
nSpin = len(DMFiles)

atoms = cfh.readCoordFile(coordFile, inUnit = atomicCoordUnit, outUnit = 'bohr')
sdObj = sd.SlaterDensity(atoms, DMFiles)
nQuadAll = quad.shape[0]
ne = [0.0]*nSpin

for i in range(0,nQuadAll,quadBatch):
    minId = i
    maxId = min(minId+quadBatch, nQuadAll)
    x = quad[minId:maxId, 0:3]
    w = quad[minId:maxId, 3]
    for iSpin in range(nSpin):
        rho = sdObj.getDensity(x, iSpin)
        ne[iSpin] += np.dot(rho,w)

for iSpin in range(nSpin):
    print ("Num. of spin", iSpin, "electrons:", ne[iSpin])


S = sdObj.getSMatrix(quad)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        print(S[i,j], file = SMatrixOutF, end =" ")

    print(file=SMatrixOutF)

SMatrixOutF.close()


outFs = []
for iSpin in range(nSpin):
    outF = open(outFilenamePrefix+str(iSpin),"w")
    outFs.append(outF)

start_time = time.time()

print("Entering derivative loop :")
for i in range(0,nQuadAll,quadBatch):
    print(i)
    minId = i
    maxId = min(minId+quadBatch, nQuadAll)
    x = quad[minId:maxId, 0:3]
    nQuad = x.shape[0]

    for iSpin in range(nSpin):
        rho = sdObj.getDensity(x,iSpin)
        drho = sdObj.getDensityGradient(x, iSpin)
        ddrho = sdObj.getDensityHessian(x,iSpin)
        outF = outFs[iSpin]
        for iQuad in range(nQuad):
            print(''.join([str(y) + " " for y in x[iQuad]]), file=outF, end=' ')
            print(rho[iQuad], file=outF, end=' ')
            print(''.join([str(y) + " " for y in drho[iQuad]]), file=outF, end=' ')
            for iComp in range(3):
                print(''.join([str(y) + " " for y in ddrho[iQuad, iComp]]), file=outF, end=' ')

            print(file=outF)

for iSpin in range(nSpin):
    outFs[iSpin].close()

print("--- %s seconds ---" % (time.time() - start_time))
os.chdir(initDir)
