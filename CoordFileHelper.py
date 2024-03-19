def getUnitChar(unit):
    if unit.lower()[:4] == 'angs' or unit.lower()[0] == 'a':
        return 'a'

    elif unit.lower()[:4] == 'bohr' or unit.lower()[:,2] == 'au' or unit.lower()[0] == 'b':
        return 'b'

    else:
        raise Exception('''Invalid input unit " + unit + " encountered. Valid units are as follows.'''\
                        '''For angstrom: 'angs', 'angstrom', or 'a'. For bohr: 'bohr', 'b', or ''au'.''')

def getFactor(inUnit, outUnit):
    __ANGS_TO_BOHR__ = 1.8897259885789
    inUnitChar = getUnitChar(inUnit)
    outUnitChar = getUnitChar(outUnit)
    factor = 1.0
    if inUnitChar == 'a' and outUnitChar == 'b':
        factor = __ANGS_TO_BOHR__

    elif inUnitChar == 'b' and outUnitChar == 'a':
        factor = 1.0/__ANGS_TO_BOHR__

    else:
        factor = 1.0

    return factor


def readCoordFile(coordFile, inUnit, outUnit):
    f = open(coordFile, 'r')
    lines = f.readlines()
    atoms = []
    factor = getFactor(inUnit, outUnit)
    for line in lines:
        words = line.split()
        if len(words) != 5:
            raise Exception("Expects only 5 values in coord file " + coordFile + "Line read: " + line +". Num words: " + str(len(words)))

        atom = {}
        atom['name'] = words[0]
        atom['coord'] = [factor*float(w) for w in words[1:4]]
        atom['basisfile'] = words[4]
        atoms.append(atom)

    f.close()
    return atoms
