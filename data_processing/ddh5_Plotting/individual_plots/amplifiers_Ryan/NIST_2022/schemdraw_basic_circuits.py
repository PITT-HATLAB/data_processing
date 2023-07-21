# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:01:02 2022

@author: Ryan
"""

class test(): 
    def __init__(self, val):
        self.val = val
    def __add__(self, other): 
        return other.val+self.val
    def __or__(self, other): 
        if other.val == self.val: 
            return True
        else: 
            return False
#draw the simple circuit with schemdraw, label optional
import schemdraw
import schemdraw.elements as elm
from schemdraw.flow import flow

#%% generator and load
with schemdraw.Drawing() as d: 
    d+=elm.Ground()
    d+= elm.SourceSin(length = 1).up()
    d+=elm.Line().right()
    d.push()
    # d+= elm.Line().down(length = 1)
    d+=elm.ResistorIEC(label = '$Y_{gen}$', font = 'Arial').down()
    # d+= elm.Line().down(length = 1)
    d+=elm.Ground()
    d.pop()
    d+=elm.Line().right()
    d+=elm.ResistorIEC(label = '$Y_{load}$').down()
    d+=elm.Ground()
#%%degenerate paramp
with schemdraw.Drawing() as d: 
    d+=elm.Ground()
    d+= elm.SourceSin().up()
    d+=elm.Line().right()
    d.push()
    # d+= elm.Line().down(length = 1)
    d+=elm.ResistorIEC(label = '$50\Omega$').down()
    # d+= elm.Line().down(length = 1)
    d+=elm.Ground()
    d.pop()
    d+=elm.Line().right()
    d.push()
    d+=elm.Capacitor().down()
    d+=elm.Ground()
    d.pop()
    d+=elm.Line().right()
    d.push()
    d+=elm.Inductor2().down()
    d+=elm.Ground()
    d.pop()
    d+= elm.Line().right()
    d+= elm.Resistor(label = '-R').down()
    d+= elm.Ground()
#%%inverter
with schemdraw.Drawing() as d:
    d.config(fontsize=12)
    IC555def = elm.Ic(pins=[elm.IcPin(name = 'a', side='left'),
                            elm.IcPin(name = 'b', side='left'),
                            elm.IcPin(name = 'c', side='right'),
                            elm.IcPin(name = 'd',side='right'),
                            ],
                       edgepadW=.5,
                       edgepadH=0,
                       pinspacing=1.5,
                       leadlen=1,
                       label='$J$', 
                       fontsize = 30)
    d += (T := IC555def)
    # d += (LEFT := elm.Ground().at(T.a))
    # d += (RIGHT := elm.Ground().at(T.c))
    d += elm.ResistorIEC().endpoints(T.d, T.c).label('$Y_{load}$', loc = 'right', 
                       fontsize = 15)
    d+= elm.lines.ZLabel(ofst = 0, hofst = 2, length = 1.5).at(IC555def).label(r"$Y_{in} = \frac{J^2}{Y_L}$", 
                       fontsize = 15)
