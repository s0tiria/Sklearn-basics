
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################
###########SYNOPSIS############
###############################
#
# Auteur : Sotiria BAMPATZANI
#
# Date : 18/02/18
#
# But : création d'un classifieur pour décider si l'IMC d'une une personne est bonne
#
# Usage : python3 decisionTreeBasic.py
###############################

from sklearn import tree

# input : taille, poids
features = [
    [160, 70],  
    [190, 80],
    [170, 60], 
    [180, 90],
    [160, 55]
]

# output attendu
# 1 corpulence normale, 0 surpoids
labels = [0, 1, 1, 0, 1]

# type of the classifier : decision tree
clf = tree.DecisionTreeClassifier()

# in sklearn, the training algorithm is included in the classifier object
clf = clf.fit(features, labels)

print(clf.predict([[165, 80]]))