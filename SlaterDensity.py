import time
import numpy as np
import SlaterBasis as SB

#
# DMFiles contains a list of filenames that contain the density matrix (DM) in the atomic orbital (AO) basis
# Density matrix (DM) in atomic orbital basis
# Usual convention is to have the file named as DM for a spin-unpolarized system.
# For a spin-polarized, the convention is to name them as DM0 and DM1 for spin-up and spin-down, respectively.
# For any other convention, directly provide the DM filenames as a list, e.g.
# DMFiles = ['FileName'] for spin-unpolarized system
# DMFiles = ['FileNameSpinUp', 'FileNameSpinDown'] for spin-polarized system
# Whether a system is spin-unpolarized or spin-polarized is determined by the number of DM files.
# That is one DM file means spin-unpolarized and two DM files means spin-polarized

# atoms is a list of dictionary of the following form:
# atoms = [ {'name': 'atom_name_1', 'coord' = [x_1, y_1, z_1], 'basisfile' = 'name of atomic basis file for the atom_1'},
#           {'name': 'atom_name_2', 'coord' = [x_2, y_2, z_2], 'basisfile' = 'name of atomic basis file for the atom_2'},
#           {'name': 'atom_name_3', 'coord' = [x_3, y_3, z_3], 'basisfile' = 'name of atomic basis file for the atom_2'},
#            ....
#          ]
# The unit for x,y,z of each atom is assumed to be in bohr (atomic units)
class SlaterDensity():
    def __init__(self, atoms, DMFiles):
        atomBasisFileSet = set()
        for atom in atoms:
            atomBasisFileSet.add(atom['basisfile'])

        atomBasisMap = {}
        for basisFile in atomBasisFileSet:
            atomBasisMap[basisFile] = SB.getAtomSlaterBasis(basisFile)

        self.basisList = []

        for atom in atoms:
            basisFile= atom['basisfile']
            basisList = atomBasisMap[basisFile]
            for basis in basisList:
                b = {}
                b['atom'] = atom['name']
                b['center'] = atom['coord']
                b['primitive'] = basis
                self.basisList.append(b)

        self.nbasis = len(self.basisList)
        numSpinComponents = len(DMFiles)
        if numSpinComponents > 2:
            raise Exception("More than two density matrices provided. The number of density matrices can be either 1 (for a spin unpolarized system) or 2 (for a spin polarized system)")
        self.DM = []
        for i in range(numSpinComponents):
            if self.nbasis== 1:
                DM = np.array([[np.loadtxt(DMFiles[i], dtype = np.float64)]])
            else:
                DM = np.loadtxt(DMFiles[i],  dtype = np.float64, usecols=range(0,self.nbasis))
            self.DM.append(DM)


    def getSMatrix(self, quad):
        npoints = quad.shape[0]
        x = quad[:,0:3]
        w = quad[:,3]
        basisVals = np.zeros((npoints, self.nbasis))
        for i in range(self.nbasis):
            center = self.basisList[i]['center']
            primitive = self.basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            xShifted = x-center
            slaterFunctionVals_dict = SB.getSlaterFunctionVals(n , l, m, a, xShifted, max_deriv=0)
            basisVals[:,i] = nrm*slaterFunctionVals_dict[0]

        S = np.zeros((self.nbasis, self.nbasis))
        for i in range(self.nbasis):
            for j in range(i,self.nbasis):
                NiNj = basisVals[:,i]*basisVals[:,j]
                S[i,j] = np.dot(NiNj,w)
                S[j,i] = S[i,j]

        return S


    def getDensity(self,x, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        npoints = x.shape[0]
        basisVals = np.zeros((npoints, self.nbasis))
        t1 = time.time()
        for i in range(self.nbasis):
            center = basisList[i]['center']
            primitive = basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            xShifted = x-center
            slaterFunctionVals_dict = SB.getSlaterFunctionVals(n, l, m, a, xShifted, max_deriv=0)
            basisVals[:,i] = nrm*slaterFunctionVals_dict[0]

        t2 = time.time()
        # U_qi = \sum_j basisVals_qj D_ji, where q is the point index and i,j are the basis indices
        # rho_q = \sum_j U_qj basisVals_qj
        U = np.einsum('qj,ji->qi', basisVals, DM)
        rho = np.einsum('qj,qj->q', basisVals, U)

        t3 = time.time()
        print("Density eval for", npoints, "BasisVals time:", t2-t1, "secs",", Products:", t3-t2, "secs")

        return rho


    def getDensityGradient(self, x, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        npoints = x.shape[0]
        basisVals = np.zeros((npoints, nbasis))
        basisGrad = np.zeros((npoints, 3, nbasis))
        t1 = time.time()
        for i in range(nbasis):
            center = basisList[i]['center']
            primitive = basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            xShifted = x - center
            slaterFunctionVals_dict = SB.getSlaterFunctionVals(n, l, m, a, xShifted, max_deriv=1)
            basisVals[:,i] = nrm*slaterFunctionVals_dict[0]
            basisGrad[:,:,i] = nrm*slaterFunctionVals_dict[1]

        t2 = time.time()
        # U_qi = \sum_j basisVals_qj D_ji, where q is the point index and i,j are the basis indices
        # B_qa = sum_j U_qj basisGrad_qaj where q is the point index; j is the basis indices; and a is the dimension index (i.e., x, y or z)
        # gradRho_qa = 2 * B_qa
        U = np.einsum('qj,ji->qi', basisVals, DM)
        gradRho = 2.0*np.einsum('qaj,qj->qa', basisGrad, U)

        t3 = time.time()

        print("Density derivative eval for", npoints, "BasisVals and derivative time:", t2-t1, "secs",", Products:", t3-t2, "secs")
        return gradRho


    def getDensityHessian(self, x, spinComponent):
        basisList = self.basisList
        nbasis = self.nbasis
        DM = self.DM[spinComponent]
        npoints = x.shape[0]
        basisVals = np.zeros((npoints, nbasis))
        basisGrad = np.zeros((npoints, 3, nbasis))
        basisHessian = np.zeros((npoints, 3, 3, nbasis))
        t1 = time.time()
        for i in range(nbasis):
            center = basisList[i]['center']
            primitive = basisList[i]['primitive']
            n, l, m = primitive.nlm()
            a = primitive.alpha()
            nrm = primitive.normConst()
            xShifted = x - center
            slaterFunctionVals_dict = SB.getSlaterFunctionVals(n, l, m, a, xShifted, max_deriv=2)
            basisVals[:,i] = nrm*slaterFunctionVals_dict[0]
            basisGrad[:,:,i] = nrm*slaterFunctionVals_dict[1]
            basisHessian[:,:,:,i] = nrm*slaterFunctionVals_dict[2]

        t2 = time.time()
        # U_qi = \sum_j basisVals_qj D_ji, where q is the point index and i,j are the basis indices
        # V_qai = \sum_j basisGrad_qaj D_ji,  where q is the point index; i,j are the basis indices; and a is the dimension index (i.e., x, y or z)
        # B_qab = \sum_j basisHessian_qabj U_qj + \sum_j V_qaj basisGrad_qbj where q is the point index; j is the basis indices; and a,b are the dimension indices (i.e., x, y or z)
        # gradRho_qa = 2 * B_qab
        U = np.einsum('qj,ji->qi', basisVals, DM)
        V = np.einsum('qaj,ji->qai', basisGrad, DM)
        B = np.einsum('qabj,qj->qab', basisHessian, U) + np.einsum('qaj,qbj->qab', V, basisGrad)
        hessianRho = 2.0*B

        t3 = time.time()

        print("Density derivative eval for", npoints, "BasisVals and derivative time:", t2-t1, "secs",", Products:", t3-t2, "secs")
        return hessianRho
