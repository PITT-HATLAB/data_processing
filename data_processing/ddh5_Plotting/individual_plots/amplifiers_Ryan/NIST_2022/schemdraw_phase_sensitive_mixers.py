# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:01:02 2022

@author: Ryan
"""

#draw the simple circuit with schemdraw, label optional
import schemdraw
import schemdraw.elements as elm
from schemdraw.flow import flow
from schemdraw import dsp

with schemdraw.Drawing() as d:
    d.config(fontsize=12)
    d += dsp.Antenna().up()
    d += dsp.Line().right(d.unit/4)
    # d += dsp.Filter(response='bp').fill('thistle').anchor('W').label('RF filter\n#1', 'bottom', ofst=.2)
    d += dsp.Line().length(d.unit/4)
    d += dsp.Amp().fill('lightblue').label('LNA')
    d += dsp.Line().length(d.unit/4)
    d += dsp.Filter(response='bp').anchor('W').fill('thistle').label('RF filter\n#2', 'bottom', ofst=.2)
    d += dsp.Line().length(d.unit/3)
    mix = dsp.Mixer().fill('navajowhite').label('Mixer')
    d += mix
    d += dsp.Line().at(mix.S).down(d.unit/3)
    d += dsp.Oscillator().right().anchor('N').fill('navajowhite').label('Local\nOscillator', 'right', ofst=.2)
    d += dsp.Line().at(mix.E).right(d.unit/3)
    d += dsp.Filter(response='bp').anchor('W').fill('thistle').label('IF filter', 'bottom', ofst=.2)
    d += dsp.Line().right(d.unit/4)
    d += dsp.Amp().fill('lightblue').label('IF\namplifier')
    d += dsp.Line().length(d.unit/4)
    d += dsp.Demod().anchor('W').fill('navajowhite').label('Demodulator', 'bottom', ofst=.2)
    d += dsp.Arrow().right(d.unit/3)

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
