"""
2 réseaux pareils (voir pertinence plus tard)
relaxer des contraintes => différence du bruit
P_edA dans les deux synapses creation

=> Réseaux connectés aléatoirement: 
-1P_ed ou 2P_ed(sym ou asym) => 2pop inh st connectées (avec les exc locales)++++ (6 simulations)
=> conserver ext drive fixe (avec asym faire une moyenne : 0.8 en base => asym un à 6 et un à 1 => mean 8.)
(fluct plus grande ou val moyenne ou spectre de fourier à regarder)
- si meme bruit dans les deux réseaux  => besoin deux pop poisson ?
- interaction avec taille des réseaux, (2 P_ed meme réseau /2 et 1 P_ed taille *2)
-variabilité du P_ed dependant du tps (meme FR => suit une sinuisoide)
-input fixe asymétrique (différence de frq.) => dynamique ++
-conncetion des pops entres elles 


"""


import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import scipy.fftpack

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)


start_scope()
DT=0.1 #fenetre de temps à modifier 
defaultclock.dt = DT*ms
N1 = 2000#2000
N2 = 8000#8000

TotTime=2000
duration = TotTime*ms



eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens
dGsynE/dt = -GsynE/Tsyn : siemens
Is:ampere
Cm:farad
gl:siemens
El:volt
a:siemens
tau_w:second
Dt:volt
Vt:volt
Ee:volt
Ei:volt
Tsyn:second
'''

# Populations----------------------------------------------------------------------------------

############################ First Adex FS-RS population 1-2 ####################################
# Population 1 - FS
b1 = 0.0*pA # no adaptation for FS 
G1 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
#
#init:
G1.v = -65*mV
G1.w = 0.0*pA
G1.GsynI=0.0*nS
G1.GsynE=0.0*nS
#parameters
G1.Cm = 200.*pF
G1.gl = 10.*nS
G1.El = -65.*mV
G1.Vt = -50.*mV
G1.Dt = 0.5*mV
G1.tau_w = 1.0*ms
G1.a = 0.0*nS
G1.Is = 0.0 #[0.0 for i in range(N1)]*nA

G1.Ee=0.*mV
G1.Ei=-80.*mV
G1.Tsyn=5.*ms

# Population 2 - RS
b2 = 1.*pA #adaptation for RS (slow adaptation or spike frequency adapatation)
G2 = NeuronGroup(N2, eqs, threshold='v > -40.0*mV', reset='v = -65*mV; w += b2', refractory='5*ms',  method='heun')
G2.v = -65.*mV
G2.w = 10.0*pA
G2.GsynI=0.0*nS
G2.GsynE=0.0*nS
G2.Cm = 200.*pF
G2.gl = 10.*nS
G2.El = -70.*mV
G2.Vt = -50.*mV
G2.Dt = 2.*mV
G2.tau_w = 1000.*ms
G2.a = 0.*nS
G2.Is = 0.0*nA #2.50*nA #[0.0 for i in range(N2)]*nA
#G2.Is[0]=0.*nA

G2.Ee=0.*mV
G2.Ei=-80.*mV
G2.Tsyn=5.*ms


# external drive-------------------------------------------------------------------------- Poisson distribution because of complex pattern of firing rate population spikes (fundamental hypothesis of the AdEx and because of chaos theory).

P_edA=PoissonGroup(8000, .8*Hz) #see what happens with external input on A network, compare witgh same input on B at the same time 

# connections-----------------------------------------------------------------------------

Qi=5.0*nS
Qe=2.0*nS

prbC= 0.05 #0.05 => 5% of connectivity probability to explain (admit now, in research), "threshold to observe realisftic neural dynamics"
prbC2=0.05
S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV'
S_12.connect('i!=j', p=prbC2)

S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')
S_11.connect('i!=j',p=prbC2)

S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
S_21.connect('i!=j',p=prbC)

S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
S_22.connect('i!=j', p=prbC)

#Synaptic Connections with external input and the inhibitors/excitators cells in B network 
#Si_inex2=[a for a in S_ed_inB.i] # inibitors-external input from neuron i
#Sj_inex2=[a for a in S_ed_inB.j] # inibitors-external input from neuron j
#Si_exex2=[a for a in S_ed_exB.i] # excitators-external input from neuron i
#Sj_exex2=[a for a in S_ed_exB.j] # excitators-external input from neuron j


############## #synapses from external drive to both populations of A network:
S_ed_inA = Synapses(P_edA, G1, on_pre='GsynE_post+=Qe') # Synapses receives the extrernal input in the A network, incremental quantal Qe because the external input is excitator 
#S_ed_inA.connect(i=Si_inex2, j=Sj_inex2) # synaptic connections of inibitors-external input of inibitors cells synapses connections
S_ed_inA.connect(p=prbC)

S_ed_exA = Synapses(P_edA, G2, on_pre='GsynE_post+=Qe')
S_ed_exA.connect(p=prbC)
#S_ed_exA.connect(i=Si_exex2, j=Sj_exex2) # synaptic connections of excitators-external input of inibitors cells synapses connections



Vtt=1 #Vtt => mean threshold of the membran potential of each adex

M1G1 = SpikeMonitor(G1)
M2G1 = StateMonitor(G1, 'v', record=range(Vtt))
M3G1 = StateMonitor(G1, 'w', record=range(Vtt))
M4G1 = StateMonitor(G1, 'GsynE', record=range(Vtt)) #monitoring of the synchronous conductance of the FS in G1
M5G1 = StateMonitor(G1, 'GsynI', record=range(Vtt)) #monitoring of the synchronous conductance of the RS in G1
FRG1 = PopulationRateMonitor(G1)

M1G2 = SpikeMonitor(G2)
M2G2 = StateMonitor(G2, 'v', record=range(Vtt))
M3G2 = StateMonitor(G2, 'w', record=range(Vtt))
M4G2 = StateMonitor(G2, 'GsynE', record=range(Vtt))
M5G2 = StateMonitor(G2, 'GsynI', record=range(Vtt))
FRG2 = PopulationRateMonitor(G2)

######################################################## Second ADEX FS-RS ##########################################################
##################################################################################################################

# Population 3 - FS
b3 = 0.0*pA
G3 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
#init:
G3.v = -65*mV
G3.w = 0.0*pA
G3.GsynI=0.0*nS
G3.GsynE=0.0*nS
#parameters
G3.Cm = 200.*pF
G3.gl = 10.*nS
G3.El = -65.*mV
G3.Vt = -50.*mV
G3.Dt = 0.5*mV
G3.tau_w = 1.0*ms
G3.a = 0.0*nS
G3.Is = 0.0 #[0.0 for i in range(N1)]*nA

G3.Ee=0.*mV
G3.Ei=-80.*mV
G3.Tsyn=5.*ms

# Population 4 - RS
b4 = 1.*pA
G4 = NeuronGroup(N2, eqs, threshold='v > -40.0*mV', reset='v = -65*mV; w += b4', refractory='5*ms',  method='heun')
G4.v = -65.*mV
G4.w = 10.0*pA
G4.GsynI=0.0*nS
G4.GsynE=0.0*nS
G4.Cm = 200.*pF
G4.gl = 10.*nS
G4.El = -70.*mV
G4.Vt = -50.*mV
G4.Dt = 2.*mV
G4.tau_w = 1000.*ms
G4.a = 0.*nS
G4.Is = 0.0*nA #2.50*nA #[0.0 for i in range(N2)]*nA
#G2.Is[0]=0.*nA

G4.Ee=0.*mV
G4.Ei=-80.*mV
G4.Tsyn=5.*ms

# external drive--------------------------------------------------------------------------

P_edB=PoissonGroup(8000, .8*Hz) #here we have only the external input in B network 

# connections-----------------------------------------------------------------------------

Qi=5.0*nS
Qe=2.0*nS

"""
#a mettre en commentaire pour test (structure pareille)
Si34=[a for a in S_12.i]
Sj34=[a for a in S_12.j]
Si33=[a for a in S_11.i]
Sj33=[a for a in S_11.j]
Si43=[a for a in S_21.i]
Sj43=[a for a in S_21.j]
Si44=[a for a in S_22.i]
Sj44=[a for a in S_22.j]
"""

prbC= 0.05 #0.05
prbC2=0.05
S_34 = Synapses(G3, G4, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV')
S_34.connect('i!=j', p=prbC)

S_33 = Synapses(G3, G3, on_pre='GsynI_post+=Qi')
S_33.connect('i!=j', p=prbC)

S_43 = Synapses(G4, G3, on_pre='GsynE_post+=Qe')
S_43.connect('i!=j', p=prbC)

S_44 = Synapses(G4, G4, on_pre='GsynE_post+=Qe')
S_44.connect('i!=j', p=prbC)

"""
#Synaptic Connections with external input and the inhibitors/excitators cells in A network 
Si_inex=[a for a in S_ed_inA.i] # inibitors-external input from neuron i
Sj_inex=[a for a in S_ed_inA.j] # inibitors-external input from neuron j
Si_exex=[a for a in S_ed_exA.i] # excitators-external input from neuron i
Sj_exex=[a for a in S_ed_exA.j] # excitators-external input from neuron j
"""


####### display Synaptic Connections with external input and the inhibitors/excitators cells in B network 
S_ed_inB = Synapses(P_edB, G3, on_pre='GsynE_post+=Qe')
#S_ed_inB.connect(i=Si_inex, j=Sj_inex) # synaptic connections of inibitors-external input of inibitors cells synapses connections in B
S_ed_inB.connect(p=prbC)


S_ed_exB = Synapses(P_edB, G4, on_pre='GsynE_post+=Qe')
#S_ed_exB.connect(i=Si_exex, j=Sj_exex) # synaptic connections of excitators-external input of excitatory cells synapses connections in B
S_ed_exB.connect(p=prbC)


Vtt=1

M1G3 = SpikeMonitor(G3)
M2G3 = StateMonitor(G3, 'v', record=range(Vtt))
M3G3 = StateMonitor(G3, 'w', record=range(Vtt))
M4G3 = StateMonitor(G3, 'GsynE', record=range(Vtt))
M5G3 = StateMonitor(G3, 'GsynI', record=range(Vtt))
FRG3 = PopulationRateMonitor(G3)

M1G4 = SpikeMonitor(G4)
M2G4 = StateMonitor(G4, 'v', record=range(Vtt))
M3G4 = StateMonitor(G4, 'w', record=range(Vtt))
M4G4 = StateMonitor(G4, 'GsynE', record=range(Vtt))
M5G4 = StateMonitor(G4, 'GsynI', record=range(Vtt))
FRG4 = PopulationRateMonitor(G4)


##################################################################################################################
##################################################################################################################


prbCBA=0.05

#Synaptic Connection from B (G3-G4) A (G1-G2) to  synapses 
#S_ABii = Synapses(G3, G1, on_pre='GsynI_post+=Qi')
#S_ABii.connect('i!=j',p=prbCBA)

#S_ABie = Synapses(G3, G2, on_pre='GsynI_post+=Qi')
#S_ABie.connect('i!=j',p=prbCBA)



S_ABei = Synapses(G4, G1, on_pre='GsynE_post+=Qe',delay=5*ms)
S_ABei.connect('i!=j',p=prbCBA)

S_ABee = Synapses(G4, G2, on_pre='GsynE_post+=Qe',delay=5*ms)
S_ABee.connect('i!=j',p=prbCBA)



##################################################################################################################
##################################################################################################################


#Synaptic Connection from A (G1-G2) to  B (G3-G4) synapses 
prbCAB=0.05
#
#S_BAii = Synapses(G1, G3, on_pre='GsynI_post+=Qi')
#S_BAii.connect('i!=j',p=prbCAB)
#
#S_BAie = Synapses(G1, G4, on_pre='GsynI_post+=Qi')
#S_BAie.connect('i!=j',p=prbCAB)
#
#
#
S_BAei = Synapses(G2, G3, on_pre='GsynE_post+=Qe',delay=5*ms)
S_BAei.connect('i!=j',p=prbCAB)
#
S_BAee = Synapses(G2, G4, on_pre='GsynE_post+=Qe',delay=5*ms)
S_BAee.connect('i!=j',p=prbCAB)



##################################################################################################################
##################################################################################################################

print('--##Start simulation##--')
run(duration)
print('--##End simulation##--')


# Plots -------------------------------------------------------------------------------


RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]])
RasG2 = np.array([M1G2.t/ms, M1G2.i])


LVG1=[]
LwG1=[]
LVG2=[]
LwG2=[]

LgseG1=[]
LgsiG1=[]
LgseG2=[]
LgsiG2=[]

for a in range(Vtt):
    LVG1.append(array(M2G1[a].v/mV))
    LwG1.append(array(M3G1[a].w/mamp))
    LVG2.append(array(M2G2[a].v/mV))
    LwG2.append(array(M3G2[a].w/mamp))
    LgseG1.append(array(M4G1[a].GsynE/nS))
    LgsiG1.append(array(M5G1[a].GsynI/nS))
    LgseG2.append(array(M4G2[a].GsynE/nS))
    LgsiG2.append(array(M5G2[a].GsynI/nS))


BIN=5
time_array = np.arange(int(TotTime/DT))*DT



LfrG2=np.array(FRG2.rate/Hz)
TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)

LfrG1=np.array(FRG1.rate/Hz)
TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)



Lt1G1=array(M2G1.t/ms)
Lt2G1=array(M3G1.t/ms)
Lt1G2=array(M2G2.t/ms)
Lt2G2=array(M3G2.t/ms)



fig=plt.figure(figsize=(12,4))
fig.suptitle('A network')
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)

for a in range(len(LVG1)):
    ax1.plot(Lt1G1, LVG1[a],'r')
    ax2.plot(Lt2G1, LwG1[a],'r')
    ax1.plot(Lt1G2, LVG2[a],'g')
    ax2.plot(Lt2G2, LwG2[a],'g')


ax1.set_ylim([-100, 0])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('$V_m$ (mV)') # v : membrane voltage 

ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('$w$ (pA)')# synapses weights 

ax3=fig.add_subplot(223)
ax3.plot(RasG1[0], RasG1[1], ',r')
ax3.plot(RasG2[0], RasG2[1], ',g')

ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Neuron index')
ax4=fig.add_subplot(224)
ax4.plot(TimBinned,popRateG1, 'r')
ax4.plot(TimBinned,popRateG2, 'g')

ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Firing Rate (Hz)')




#######################################################
RasG3 = np.array([M1G3.t/ms, [i+N2 for i in M1G3.i]])
RasG4 = np.array([M1G4.t/ms, M1G4.i])


LVG3=[]
LwG3=[]
LVG4=[]
LwG4=[]

LgseG3=[]
LgsiG3=[]
LgseG4=[]
LgsiG4=[]

for a in range(Vtt):
    LVG3.append(array(M2G3[a].v/mV))
    LwG3.append(array(M3G3[a].w/mamp))
    LVG4.append(array(M2G4[a].v/mV))
    LwG4.append(array(M3G4[a].w/mamp))
    LgseG3.append(array(M4G3[a].GsynE/nS))
    LgsiG3.append(array(M5G3[a].GsynI/nS))
    LgseG4.append(array(M4G4[a].GsynE/nS))
    LgsiG4.append(array(M5G4[a].GsynI/nS))


BIN=5
time_array = np.arange(int(TotTime/DT))*DT



LfrG4=np.array(FRG4.rate/Hz)
TimBinned,popRateG4=bin_array(time_array, BIN, time_array),bin_array(LfrG4, BIN, time_array)

LfrG3=np.array(FRG3.rate/Hz)
TimBinned,popRateG3=bin_array(time_array, BIN, time_array),bin_array(LfrG3, BIN, time_array)



Lt1G3=array(M2G3.t/ms)
Lt2G3=array(M3G3.t/ms)
Lt1G4=array(M2G4.t/ms)
Lt2G4=array(M3G4.t/ms)



figB=plt.figure(figsize=(12,4))
figB.suptitle('B network')
ax1B=figB.add_subplot(221)
ax2B=figB.add_subplot(222)

for a in range(len(LVG3)):
    ax1B.plot(Lt1G3, LVG3[a],'r')
    ax2B.plot(Lt2G3, LwG3[a],'r')
    ax1B.plot(Lt1G4, LVG4[a],'g')
    ax2B.plot(Lt2G4, LwG4[a],'g')


ax1B.set_ylim([-100, 0])
ax1B.set_xlabel('Time (ms)')
ax1B.set_ylabel('$V_m$ (mV)') # v : membrane voltage 

ax2B.set_xlabel('Time (ms)')
ax2B.set_ylabel('$w$ (pA)')# synapses weights 

ax3B=figB.add_subplot(223)
ax3B.plot(RasG3[0], RasG3[1], ',r')
ax3B.plot(RasG4[0], RasG4[1], ',g')

ax3B.set_xlabel('Time (ms)')
ax3B.set_ylabel('Neuron index')
ax4B=figB.add_subplot(224)
ax4B.plot(TimBinned,popRateG3, 'r')
ax4B.plot(TimBinned,popRateG4, 'g')

ax4B.set_xlabel('Time (ms)')
ax4B.set_ylabel('Firing Rate (Hz)')

print(np.mean(popRateG2))
plt.show()
