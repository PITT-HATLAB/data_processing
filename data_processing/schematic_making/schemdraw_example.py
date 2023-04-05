# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:24:25 2022

@author: Hatlab-RRK
"""

import schemdraw
import schemdraw.elements as elm
l = 1
with schemdraw.Drawing(show = False) as d: 

    d+= elm.Capacitor(label = 'Test').right().length(2*l)
    d+=elm.Inductor2().down()
    d+=elm.Line().length(l).right()
    d+=elm.Ground()
    d+=elm.Line().length(l).right()
    d+=elm.Capacitor().up()
    d.push()
    d+=elm.Line().left().length(2*l)
    d.pop()
    # d+=elm.Capacitor().right()

with schemdraw.Drawing() as d1: 
    for i in range(3): 
        d1+=elm.ElementDrawing(d)

# with schemdraw.Drawing() as d: 
#     d+= elm.Capacitor().right()
#     d+=elm.Inductor2().down()
#     d+=elm.Line().length(1.5).right()
#     d+=elm.Ground()
#     d+=elm.Line().length(1.5).right()
#     d+=elm.Capacitor().up()
#     d.push()
#     d+=elm.Line().left()
#     d.pop()
#     d+=elm.Capacitor().right()
#     d+=elm.Josephson().down() 
#     d.push()
#     d+=elm.Capacitor().left()
#     d.pop()
#     d+=elm.Line().length(1.5).right()
#     d+=elm.Capacitor().up()
#     d+=elm.Line().left().length(1.5)
    
# data.sel({'frequency': 4.5, 'power': 0}, method = 'nearest')

# slice_val = np.argmin(np.abs(np.unique(frequency)-f_val))