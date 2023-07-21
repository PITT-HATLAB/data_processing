# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:01:02 2022

@author: Ryan
"""


#draw the simple circuit with schemdraw, label optional
import schemdraw
import schemdraw.elements as elm
from schemdraw import flow
#%%
l2 = 1.5
l1 = l2/2
ld = 2

with schemdraw.Drawing(transparent = True) as d: 
    # d+=elm.Coax(leadlen = 0).right().reverse()
    d+=elm.Capacitor().right().length(l2)
    d.push()
    d+= elm.Capacitor().down().length(ld)
    d+= elm.Line().right().length(l1)
    d+=elm.Ground()
    d+=elm.Line().right().length(l1)
    d+=elm.Line().right().length(0.5).dot(open = True)
    d+=elm.Line().left().length(0.5)
    d+=elm.Inductor2().up().length(ld).color('darkblue')
    d+=elm.Line().right().length(0.5).dot(open = True)
    d+=elm.Line().left().length(0.5)
    d.pop()
    d+=elm.Line().right().length(l2)
    d.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAMP_schematic.svg')
#%%
l2 = 1.5
l1 = l2
ld = 2
focuscolor = 'darkblue'
focuscolor1 = 'darkblue'
with schemdraw.Drawing(show=False) as d1:
    # d1 += elm.Line().down().length(l2)
    d1 += elm.Inductor2(loops = 3, leadlength = 10).length(l1).color(focuscolor)
    d1.push()
    d1 += elm.Line().down().length(l2)
    d1 += elm.Josephson().left().length(l1).color(focuscolor1)
    d1 += elm.Line().up().length(l2)
    d1.pop()

with schemdraw.Drawing() as d2:
    d2.push()
    d2+=elm.Line().left().dot(open = True).length(0.5)
    d2.pop()
    for i in range(3):
        d2 += elm.ElementDrawing(d1)

    d2.push()
    d2 += elm.Line().length(d2.unit/6).color(focuscolor)
    d2 += elm.DotDotDot()
    d2 += elm.ElementDrawing(d1)
    d2+=elm.Line().right().dot(open = True).length(0.5)
    d2.pop()
    d2.here = (d2.here[0], d2.here[1]-l2)
    d2 += elm.Line().right().length(d2.unit/6).color(focuscolor)
    d2 += elm.DotDotDot()
    d2.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\NISTAMP_subschematic.svg')
#%%
from schemdraw import dsp
l2 = 1.5
ll = 1
lh = 0.5
ld = 2
bgcolor = 'black'
focuscolor = 'darkblue'
with schemdraw.Drawing() as d2: 
    d2.push()
    # d2+=elm.Coax(length = 2, leadlen = 0.1).right().reverse()
    d2+=elm.Capacitor().right().length(l2).color(bgcolor)
    d2.push()
    d2+= elm.Capacitor().down().length(ld).color(bgcolor)
    d2+= elm.Line().right().length(0.75).color(bgcolor)
    d2+=elm.Ground()
    d2+=elm.Line().right().length(0.75).color(bgcolor)
    
    d2+=elm.Inductor2().up().length(ld).color(focuscolor)
    
    d2.pop()
    d2+=elm.Line().right().length(1.5).color(bgcolor)
    d2.pop()
    d2.push()
    d2+=elm.Line().up().length(ll).color(bgcolor)
    d2+=elm.Line().left().length(ll).color(bgcolor)
    d2+= dsp.dsp.Filter2(response = 'hp').left().color(bgcolor)
    d2+= elm.Line().left().length(0.75).color(bgcolor)
    d2+= elm.Coax().left().color(bgcolor)
    d2.pop()
    d2+=elm.Line().down().length(ll).color(bgcolor)
    d2+=elm.Line().left().length(ll).color(bgcolor)
    d2+= dsp.dsp.Filter2(response = 'lp').left().color(bgcolor)
    d2+= elm.Line(alpha = 0).left().length(ll).color(bgcolor)
    d2.push()
    d2+= elm.Line().left().length(lh).color(bgcolor)
    d2.push()
    d2+= elm.Line().down().length(lh+ll).color(bgcolor)
    d2.pop()
    d2+= elm.Line().left().length(ll).color(bgcolor)
    d2.push()
    d2+= elm.Line().down().length(lh+ll).color(bgcolor)
    d2.pop()
    d2+= elm.Line().left().length(ll+lh).color(bgcolor)
    d2+= flow.Box(anchor = 'W', w = 1.5, h = 1.5).left().color(bgcolor)
    
    d2.pop()
    # d2+= dsp.dsp.Circulator().left().fill('white')
    # d2+= dsp.dsp.Circulator().left()
    d2.push()
    d2.save(r'C:\Users\Ryan\OneDrive - University of Pittsburgh\slides_figures\JPA_plus_diplexer.svg', transparent = True)
with schemdraw.Drawing() as d3: 
    d3+= dsp.dsp.Circulator().left().fill('white')
    d3+=elm.Line().left().length(0.25)
    d3+= dsp.dsp.Circulator().left().fill('white')
    d3.save('doubleCirc.svg', transparent = True)
#%%
