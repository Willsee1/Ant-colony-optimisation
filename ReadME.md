#  Title
Ant colony optimisation for Travelling salesman problem

##  Author 
William See

##  Description
The algorithm is written in python and performs the ACO algorithm to solve the travelling salesman problem. It forms a distance matrix out of an xml file and sends iterations of ants to 'travel' across the citys in hopes of finding the shortest paths. The code returns values for the shortest, longest and the average path length as well as providing the paths taken for the shortest and longest paths.

##  Installations
Imports
 xml.etree.ElementTree 
 numpy 

## Usage guide
The code requires an xml file in the style of 'Burma14' or 'Brazil58' to be stored in the same folder as the the python code. The XML file the code runs can be changed on line 4.

The code can be run by pressing the 'F5' OR 'Crtl + F5' and runtimes will vary depending on the distance matrix created or the heuristic and variables used.

Heuristics can be switched between '1/d' and 'Q/d' by changing the function called on line 179 in the 'ACO_algorithm' function.

Parameters found and the bottom of the code can be changed and will have an impact on results and runtime.

To change to an elitest approach can the 'False' to 'True' on line 234, found just above the print statements.
