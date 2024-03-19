## This repository contains a bunch of python scripts to work with electron density represented in an atomic orbital basis, such as Slater-type orbitals and Gaussian-type orbitals. 

## Prerequisites
+ ```numpy```
+ ```JAX``` 

## Coordinate file format
The coordinate file should have the following format:
```
<Atom1-Symbol> <x-coord> <y-coord> <z-coord> <basis file name>
<Atom2-Symbol> <x-coord> <y-coord> <z-coord> <basis file name>
<Atom3-Symbol> <x-coord> <y-coord> <z-coord> <basis file name>
....
```
 
## Quadrature file format
The format of the file should be
```
<x_1> <y_1> <z_1> <w_1>
<x_2> <y_2> <z_2> <w_2>
<x_3> <y_3> <z_3> <w_3>
...
```
In the above, x_i, y_i, and z_i refer to the x,y,z coordinate of the i-th point. w_i denotes the weight associated with the i-th quadrature point. 

## Slater basis file format
The slater basis file for a given atom should be of the following format
```
<Atom-Symbol>
<n_1><l_character_1> <alpha_1>
<n_2><l_character_2> <alpha_2>
<n_3><l_character_3> <alpha_3>
...
```
In the above, n_i denotes the i-th *n* quantum number, l-character_i denotes the character of i-th *l* quantum number (e.g., s, p, d, f); and alpha_i denotes the exponent for the radial part of the slater function. Example file,

```
H
1S   3.300
1S   2.000
1S   1.400
1S   1.000
1S   0.710
2P   2.000
2P   1.000
3D   2.500
3D   1.500
```
