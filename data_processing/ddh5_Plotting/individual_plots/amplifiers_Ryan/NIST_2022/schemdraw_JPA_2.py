# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:01:02 2022

@author: Ryan
"""


#draw the simple circuit with schemdraw, label optional
import schemdraw
import schemdraw.elements as elm


l2 = 1.5
l1 = l2/2
ld = 2
color = 'black'
with schemdraw.Drawing(transparent = True) as d: 
    # d+=elm.Coax(leadlen = 0, length = 1).right().reverse()
    d+=elm.Capacitor(color = color).right().length(l2)
    d.push()
    d+= elm.Capacitor(color = color).down().length(ld)
    d+= elm.Line().right().length(l1)
    d+=elm.Ground()
    d+=elm.Line().right().length(l1)
    color = 'black'
    d+=elm.Inductor2(color = color, loops = 4).up().length(ld)
    d.pop()
    d+=elm.Line().right().length(l2)
    # d.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAMP_schematic.svg')
#%%Broadened version
l2 = 1.5
l1 = l2/2
ld = 2
color = 'black'
with schemdraw.Drawing(transparent = True) as d: 
    d+=elm.Line(arrow = '>', arrowwidth = 0.75, arrowlength = 0.75).right().length(0.00000001)
    d+=elm.Line().right().length(1)
    d.push()
    d+=elm.Coax().down().reverse().label(r'$\frac{\lambda}{2}$', fontsize = 20).color('green')
    d+=elm.Ground().color('green')
    d.pop()
    d+=elm.Coax(length = 1.5+0.6).right().reverse().label(r'$\frac{\lambda}{4}$', fontsize = 20).color('darkorange')
    d.push()
    d+=elm.Coax().down().reverse().label(r'$\frac{\lambda}{2}$', fontsize = 20).color('green')
    d+=elm.Ground().color('green')
    d.pop()
    d+=elm.Coax(length = 1.5+0.6).right().reverse().label(r'$\frac{\lambda}{4}$', fontsize = 20).color('darkorange')
    d.push()
    d+= elm.Capacitor(color = color).down().length(ld)
    d+= elm.Line().right().length(l1)
    d+=elm.Ground()
    d+=elm.Line().right().length(l1)
    color = 'black'
    d+=elm.Inductor2(color = color, loops = 4).up().length(ld)
    d.pop()
    d+=elm.Line().right().length(l2)
    # d.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAMP_schematic.svg')

#%%
l2 = 1.5
l1 = l2
ld = 2
with schemdraw.Drawing(show=False) as d1:
    # d1 += elm.Line().down().length(l2)
    d1 += elm.Inductor2(loops = 3, leadlength = 10).length(l1)
    d1.push()
    d1 += elm.Line().down().length(l2)
    d1 += elm.Josephson().left().length(l1)
    d1 += elm.Line().up().length(l2)
    d1.pop()

with schemdraw.Drawing() as d2:
    for i in range(3):
        d2 += elm.ElementDrawing(d1)

    d2.push()
    d2 += elm.Line().length(d2.unit/6)
    d2 += elm.DotDotDot()
    d2 += elm.ElementDrawing(d1)
    d2.pop()
    d2.here = (d2.here[0], d2.here[1]-l2)
    d2 += elm.Line().right().length(d2.unit/6)
    d2 += elm.DotDotDot()
    d2.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAMP_subschematic.svg')
#%%
from schemdraw import dsp
l2 = 1.5
ll = 1
lh = 0.5
with schemdraw.Drawing() as d2: 
    d2.push()
    # d2+=elm.Coax(length = 2, leadlen = 0.1).right().reverse()
    d2+=elm.Capacitor().right().length(l2)
    d2.push()
    d2+= elm.Capacitor().down().length(ld)
    d2+= elm.Line().right().length(l1)
    d2+=elm.Ground()
    d2+=elm.Line().right().length(l1)
    d2+=elm.Inductor2().up().length(ld)
    d2.pop()
    d2+=elm.Line().right().length(l2)
    d2.pop()
    d2.push()
    d2+=elm.Line().up().length(ll)
    d2+=elm.Line().left().length(ll)
    d2+= dsp.dsp.Filter2(response = 'hp').left()
    d2+= elm.Line().left().length(0.75)
    d2+= elm.Coax().left()
    d2.pop()
    d2+=elm.Line().down().length(ll)
    d2+=elm.Line().left().length(ll)
    d2+= dsp.dsp.Filter2(response = 'lp').left()
    d2+= elm.Line(alpha = 0).left().length(ll)
    d2.push()
    d2+= elm.Line().left().length(lh)
    d2.push()
    d2+= elm.Line().down().length(lh+ll)
    d2.pop()
    d2+= elm.Line().left().length(ll)
    d2.push()
    d2+= elm.Line().down().length(lh+ll)
    d2.pop()
    d2+= elm.Line().left().length(ll+lh)
    
    d2.pop()
    # d2+= dsp.dsp.Circulator().left().fill('white')
    # d2+= dsp.dsp.Circulator().left()
    d2.push()
    d2.save('JPA_plus_diplexer.svg', transparent = True)
with schemdraw.Drawing() as d3: 
    d3+= dsp.dsp.Circulator().left().fill('white')
    d3+= dsp.dsp.Circulator().left().fill('white')
    d3.save('doubleCirc.svg', transparent = True)
#%% generator and load