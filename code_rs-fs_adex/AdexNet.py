
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *


#########################################################
#Define conditions for simulation
#start Brian scope:
start_scope()
#set dt value for integration (ms):
DT=0.1 # time steps
seed(1)
defaultclock.dt = DT*ms #time conversion in ms

#total duration of the simulation (ms):
TotTime=8000
duration = TotTime*ms


#######################################################
#set the number of neuron of each population:
#inhibitory Fast Spiking (FS, population 1):
N1 = 2000
#Excitatory Regular Spiking (RS, population 2):
N2 = 8000

#More excitators than inhibitors (four times more)

########################################################
# define equations of the model 
# define units of parameter
#Gsyn : synaptic neurons group
#siemens is the unit of the variavble GsynI, : permits to characterize the unit)
# a : adaptation couppling parameter
eqs='''
dv/dt = (-GsynE*(v-Ee)-GsynI*(v-Ei)-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-w + Is)/Cm : volt (unless refractory)
dw/dt = (a*(v-El)-w)/tau_w:ampere
dGsynI/dt = -GsynI/Tsyn : siemens 
dGsynE/dt = -GsynE/Tsyn : siemens
Pvar:1
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
#Noise can be added by Gaussian random  distribution 'Xi'
########################################################
#Create populations:

	# Population 1 - FS

b1 = 0.0*pA #no adaptation for FS
#generate the population
#First Group generated by FS cells (inihibitors), v=voltage
G1 = NeuronGroup(N1, eqs, threshold='v > -47.5*mV', reset='v = -65*mV', refractory='5*ms', method='heun')
#set values:
# initial values of variables:
G1.v = -65 *mV
G1.w = 0.0 *pA #le poids des FS cells
G1.GsynI =0.0 *nS
G1.GsynE =0.0 *nS
# parameters values:
#soma:
G1.Cm = 200.*pF
G1.gl = 10.*nS
G1.El = -65.*mV
G1.Vt = -50.*mV
G1.Dt = 0.5*mV
G1.tau_w = 1.0 *ms #(no adapation, just to do not have error due to zero division)
G1.a = 0.0 *nS
G1.Is = 0.*pA  
#synapses:
G1.Ee =0.*mV
G1.Ei =-80.*mV
G1.Tsyn =5.*ms


# Population 2 - RS
b2 = 40.*pA
#generate the population
G2 = NeuronGroup(N2, eqs, threshold='v > -40*mV', reset='v = -55*mV; w += b2', refractory='5*ms',  method='heun')
#set values:
# initial values of variables:
G2.v = -65.*mV #v assumed as a post synaptic variable
G2.w = 0.0 *pA 
G2.GsynI =0.0 *nS
G2.GsynE =0.0 *nS
# parameters values:
#soma:
G2.Cm = 200.*pF
G2.gl = 10.*nS
G2.El = -65.*mV
G2.Vt = -50.*mV
G2.Dt = 2.*mV
G2.tau_w = 1500.*ms
G2.a = 2.*nS
G2.Is = 0.*nA  
#synpases:
G2.Ee =0.*mV
G2.Ei =-80.*mV
G2.Tsyn =5.*ms




#######################################################
# external drive--------------------------------------- external input => generating spikes methods

P_ed=PoissonGroup(8000, rates= 2.5*Hz) 


#######################################################
# connections-------------------------------------------
#quantal increment when spike:
Qi=5.*nS #inhibitor 
Qe=1.5*nS #excittor

#probability of connection
prbC= 0.05 

#synapses from FS to RS:
S_12 = Synapses(G1, G2, on_pre='GsynI_post+=Qi') #'v_post -= 1.*mV'=> each synapse receiving a presynaptic spike (on_pre), v_post/GsynI_post/GsynE_post is a pre-synaptic variable 
# on_post => postsynaptic neuron has fired a spike.
S_12.connect('i!=j', p=prbC)#synaptic connections of neurons from G1 to G2 when neurons index are different
#synapses from FS to FS:
S_11 = Synapses(G1, G1, on_pre='GsynI_post+=Qi')#Qi and Qe are synapses-specific weights (increment)
S_11.connect('i!=j',p=prbC)#synaptic connections of neurons from G1 to G1
#synapses from RS to FS:
S_21 = Synapses(G2, G1, on_pre='GsynE_post+=Qe')
S_21.connect('i!=j',p=prbC) #synaptic connections of neurons from G2 to G1 excluding autapses connections
#synapses from RS to RS:
S_22 = Synapses(G2, G2, on_pre='GsynE_post+=Qe')
S_22.connect('i!=j', p=prbC)#synaptic connections of neurons from G2 to G2 


"""
S.connect()
S.connect(i=[1, 2], j=[3, 4])
S.connect(i=numpy.arange(10), j=1)


The first statement connects all neuron pairs. The second statement creates synapses between neurons 1 and 3, and between neurons 2 and 4. The third statement creates synapses between the first ten neurons in the source group and neuron 1 in the target group.

syn.w = '1.0/N_incoming' => normalization of synaptic weigths
"""



#synapses from external drive to both populations:
S_ed_in = Synapses(P_ed, G1, on_pre='GsynE_post+=Qe') 
S_ed_in.connect(p=prbC)

S_ed_ex = Synapses(P_ed, G2, on_pre='GsynE_post+=Qe')
S_ed_ex.connect(p=prbC)


######################################################
#set recording during simulation
#number of neuron record of each population:
Nrecord=1

M1G1 = SpikeMonitor(G1)
"""
only records the time of the spike and the index of the neuron that spiked. Sometimes it can be useful to addtionaly record other variables, e.g. the membrane potential for models where the threshold is not at a fixed value 'v' or 'w'
"""
M2G1 = StateMonitor(G1, 'v', record=range(Nrecord)) #monitoring of v from the synapse 1 in G1
M3G1 = StateMonitor(G1, 'w', record=range(Nrecord)) #monitoring of w from the synapse 1 in G1
FRG1 = PopulationRateMonitor(G1)


M1G2 = SpikeMonitor(G2)
M2G2 = StateMonitor(G2, 'v', record=range(Nrecord))#monitoring of v from the synapse 1 in G2
M3G2 = StateMonitor(G2, 'w', record=range(Nrecord))#monitoring of v from the synapse 1 in G2
FRG2 = PopulationRateMonitor(G2)#moyenne : calcul de pas de temps (


#######################################################

#Run the simulation

print('--##Start simulation##--')
run(duration)
print('--##End simulation##--')



#######################################################
#Prepare recorded data

#organize arrays for raster plots:
RasG1 = np.array([M1G1.t/ms, [i+N2 for i in M1G1.i]]) ## all spikes in G1 originating from neuron i increment with RS population (N2) (i index of a neuron responsible for a spike from G1)
RasG2 = np.array([M1G2.t/ms, M1G2.i]) #all spikes of G2 (i index of a neuron responsible for a spike from G2) 

#Note that for plotting all recorded values at once, you have to transpose the variable values


#organize time series of single neuron variables
LVG1=[]
LwG1=[]
LVG2=[]
LwG2=[]
for a in range(Nrecord):
    LVG1.append(array(M2G1[a].v/mV)) #monitor of each spike pair accumulation of each G2 spike, one neuron
    LwG1.append(array(M3G1[a].w/mamp)) 
    LVG2.append(array(M2G2[a].v/mV))
    LwG2.append(array(M3G2[a].w/mamp))

Ltime=array(M2G1.t/ms)

#Calculate population firing rate :

#function for binning:
def bin_array(array, BIN, time_array): # smoooth 
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)


BIN=5
time_array = np.arange(int(TotTime/DT))*DT

LfrG2=np.array(FRG2.rate/Hz) #FRG2 : population rate monoitor
TimBinned,popRateG2=bin_array(time_array, BIN, time_array),bin_array(LfrG2, BIN, time_array)

LfrG1=np.array(FRG1.rate/Hz)
TimBinned,popRateG1=bin_array(time_array, BIN, time_array),bin_array(LfrG1, BIN, time_array)

arr=[popRateG1,popRateG2,LwG2,LVG1,LVG2,LfrG1,LfrG2,TimBinned,time_array]
np.save('2pop_data.npy', arr)
print("Your array has been saved to 2pop_data.npy")

##############################################################################
# prepare figure and plot

fig=plt.figure(figsize=(12,5))
fig.suptitle('AdEx Network', fontsize=12)
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)

ax1.set_title('Network activity')
ax1.plot(RasG1[0], RasG1[1], 'g',label='FS cells') #spikes plot
ax1.plot(RasG2[0], RasG2[1], 'r',label='FS cells')

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Neuron index')

ax3.plot(TimBinned,popRateG1, 'g',label='RS cells')
ax3.plot(TimBinned,popRateG2, 'r',label='FS cells')

ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('population Firing Rate')

for a in range(Nrecord):
    ax2.plot(Ltime, LVG1[a],'g',label='RS cells') 
    ax4.plot(Ltime, LwG1[a],'g',label='RS cells')
    ax2.plot(Ltime, LVG2[a],'r',label='FS cells')
    ax4.plot(Ltime, LwG2[a],'r',label='FS cells')


ax2.set_title('Single neurons variables')
ax2.set_ylabel('Membrane potential $V_m$ in mV') # v : membrane voltage 

ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Adaptation $w$ in pA')# synapses weights 

plt.tight_layout()
plt.show()
