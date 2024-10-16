from numpy import *
import re
import os
import matplotlib.pyplot as plt
import linecache
from scipy.ndimage import gaussian_filter1d
import glob
print()
print()
print ("------------------------ WELCOME in VE-RE Framework ----------------------------")
print ("    VERE-pyfw: A framework for seismic-induced landslide hazard mapping\
            using physically informed machine learning supported by InSAR investigation")

# ---------------------------------- Library for Utility --------------------------------------

def first_neg(a):# returns the index of the first negative value in the lis
  i = 0
  while i < len(a):
    if a[i] < 0:
      return i
    else:
      i = i + 1

def rad(a):# returns rad from deg
  a=a*pi/180
  return a

def deg(a):# returns deg from rad
  a=a*180/pi
  return a

def ind_val(array, val):# returns the index of the element in the array closest to the given value
  ind=argmin((absolute(array-val)))
  return ind

def random_multi_index(n, interval_size):
    J = []
    for i in range(n):
        j = int(random.uniform(0, interval_size))
        J.append(j)
    return array(J,int)
    
def cum_probability(A):# computes the cumulative probability distribution and normalizes the array
  for i in range (1, len(A)):
    A[i]=A[i-1]+A[i]
  A=A/max(A)
  return A

def mov_avg(arr,step):# calculates the moving average of the array with the specified window size (step)
  i = step
  ext=zeros(i-1)
  arrr=concatenate((ext, arr))
  moving_averages = []
  j=0
  while i <= len(arrr):
    window_average = sum(arrr[j:i])
    moving_averages.append(window_average)
    i += 1
    j += 1
  return(moving_averages)


# ------------------------------ Library for Earthquake Analysis ------------------------------

def cut_TH(a, value):# cuts the accelerogram starting from the point where it reaches the specified fraction of the peak ground acceleration (PGA)
    a = asarray(a)
    target=max(a)*value
    index=min(enumerate(a), key=lambda x: abs(target - x[1]))
    return a[index[0]::]
  

def nmk(ak,at,dt):# computes the Newmark integral for velocity and displacement from acceleration input
  V=ak*0.
  D=ak*0.
  at[at<0]=0.
  delta_v=(at-ak)*dt
  v=0.
  s=0.
  for i in range (0, len(at)):
    vi=v+delta_v[i]
    if vi<0.:
      vi=0.
    else:
      pass
    v=vi
    V[i]=v
    si=s+(vi*dt)
    if si<0.:
      si=0.
    else:
      pass
    s=si
    D[i]=s
  return(V,D)
  
def t_star(ak_t,ah_t,t,dt):# computes cumulative time_star for values of ah_t greater than ak_t based on the difference between the two acceleration arrays/
                           # ak_t and ah_t: arrays of the same size; dt is the time step of the accelerogram
  A=ah_t-ak_t
  A[A<0]=0
  A[A>0]=1
  for i in range (0, len(t)):
    t[i]=sum(A[0:i])*dt
  return t
    
def ve_dist(Vo,a, z):# computes the Vs distribution in the underground based on a log-linear model
  vs=Vo+a*log(1+z)
  return vs

# ------------------------------ Library for Earthquake Analysis ------------------------------

def interpp(x,X,Ytarget):# performs interpolation for a given x (which can be a scalar or a 2D array) using X as the reference and Ytarget as the values to interpolate
  a=interp(x, X, Ytarget)
  return a

def drange(start, stop, step):# generates a range of values from start to stop with a specified step
  r = start
  while r < stop:
    yield r
    r += step
  
def func_quantil(X,Y,minn,maxx,n,q):# computes the quantiles of Y for segments defined by the ranges in X, with specified min, max, and number of segments
  A=arange(minn,maxx+n,n)
  QQ=[]
  XX=[]
  for j in range(0, len(A)-1):
    Q=[]
    for i in range (0, len(Y)):
      if  (A[j] <= X[i] < A[j+1]):
        Q.append(Y[i])
    if len(Q)>0:
     QQ.append(percentile(Q, q))
     XX.append(0.5*(A[j]+A[j+1]))
    else:
      pass

  return XX,QQ


###################################################################################################################################################################             
######################################################   MODULE n. 1 METADATA FOR ML REGRESSION   #################################################################
###################################################################################################################################################################

path1 = 'OUTPUT-VE'
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);    

path1 = 'OUTPUT-VE/Normal_Distribution'
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);    

path1 = 'OUTPUT-VE/Distribution for Training'
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

# *************************** CREATED VIRTUAL ENVIRONMENT **************************************
# ................. Geotechnical Parameters and Initial Conditions ...................

# Insert (Average, Standard Deviation)
alpha = (12, 4)  # (mean, standard deviation) for slope angle
sfs = (22, 4)    # (mean, standard deviation) for peack angle (sfs)
sre = (13, 3)    # (mean, standard deviation) for residual angle (sre)
ru = (0.15, 0.10) # (mean, standard deviation) for porewater pressure  (ru)
ir = (0, 0.8)    # (mean, standard deviation) for modified residual factor (ir)

# Population of Parameters' Normal Distribution
N_samp = 50  # number of samples for normal distribution

# Number of Combinations to Create Representative Virtual Scenario (Monte Carlo selection)
N_comb = 2000  # number of combinations for Monte Carlo simulations

# Percentil levels for trining (n. 3 levels) 
Perc_1 = 90  # high
Perc_2 = 50  # medium
Perc_3 = 20  # low

#*******************************************************************************************************
#............................ R U N N I N G ................................................
#...........................................................................................
Alpha=[]
Sfs=[]
Sre=[]
Ru=[]
Ir=[]

print("............ Parameters Distribution Generation ................")
for i in range(0, N_samp):
  alpha_i = abs(random.normal(alpha[0], alpha[1], 1))
  Alpha.append(rad(alpha_i))
  sfs_i = abs(random.normal(sfs[0], sfs[1], 1))
  Sfs.append(tan(rad(sfs_i)))
  sre_i = abs(random.normal(sre[0], sre[1], 1))
  if sre_i>=sfs_i:
    sre_i=sre_i-2.
  else:
    pass
  Sre.append(tan(rad(sre_i)))
  ru_i = abs(random.normal(ru[0], ru[1], 1))
  Ru.append(ru_i)
  ir_i = abs(random.normal(ir[0], ir[1], 1))
  Ir.append(ir_i)
Alpha=array(Alpha,float)
Sfs=array(Sfs,float)
Sre=array(Sre,float)
Ir=array(Ir,float)
Ir[Ir>1.]=0.
Ru=array(Ru,float)

#plot and save Alpha_Slope angle
print("Plot and save Alpha_Slope angle")
plt.hist(deg(arctan(Alpha)), bins=10,color='green', edgecolor='black')
plt.xlabel('Slope angle (°)')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-VE/Normal_Distribution/Slope_angle', dpi=600)
x=deg(arctan(histogram(Alpha, bins=10)[1]))
y=histogram(Alpha, bins=10)[0]
x=mov_avg(x,1)
x=x[1::]
y=y.flatten()
savetxt('OUTPUT-VE/Normal_Distribution/Slope_angle.csv', transpose((x, y)), delimiter=',', comments='x_data, y_frequency')
plt.show()

#plot and save Peak Angle
print("Plot and save Peak Angle")
plt.hist(deg(arctan(Sfs)), bins=10,color='red', edgecolor='black')
plt.xlabel('Peak angle (°)')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-VE/Normal_Distribution/Peak angle', dpi=600)
x=deg(arctan(histogram(Sfs, bins=10)[1]))
y=histogram(Sfs, bins=10)[0]
x=mov_avg(x,1)
x=x[1::]
y=y.flatten()
savetxt('OUTPUT-VE/Normal_Distribution/Peak_angle.csv', transpose((x, y)), delimiter=',', comments='x_data, y_frequency')
plt.show()

#plot and save Residual Angle
print("Plot and save Residual Angle")
plt.hist(deg(arctan(Sre)), bins=10,color='orange', edgecolor='black')
plt.xlabel('Residual angle (°)')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-VE/Normal_Distribution/Residual angle', dpi=600)
x=deg(arctan(histogram(Sre, bins=10)[1]))
y=histogram(Sre, bins=10)[0]
x=mov_avg(x,1)
x=x[1::]
y=y.flatten()
savetxt('OUTPUT-VE/Normal_Distribution/Residual_angle.csv', transpose((x, y)), delimiter=',', comments='x_data, y_frequency')
plt.show()


#plot and Save Ru
print("Plot and save Ru")
plt.hist(Ru, bins=10,color='blue', edgecolor='black')
plt.xlabel('Ru ratio ()')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-VE/Normal_Distribution/Ru ratio', dpi=600)
x=histogram(Ru, bins=10)[1]
y=histogram(Ru, bins=10)[0]
x=mov_avg(x,1)
x=x[1::]
y=y.flatten()
savetxt('OUTPUT-VE/Normal_Distribution/Ru_ratio.csv', transpose((x, y)), delimiter=',', comments='x_data, y_frequency')
plt.show()

#plot and save Residual factor Ir
print("Plot and save Ir")
plt.hist(Ir, bins=10,color='pink', edgecolor='black')
plt.xlabel('Residula Factor Ir ()')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-VE/Normal_Distribution/Residula Factor', dpi=600)
x=histogram(Ir, bins=10)[1]
y=histogram(Ir, bins=10)[0]
x=mov_avg(x,1)
x=x[1::]
y=y.flatten()
savetxt('OUTPUT-VE/Normal_Distribution/Residula_Factor Ir.csv', transpose((x, y)), delimiter=',', comments='x_data, y_frequency')
plt.show()

print("............ Parameters Distribution Generation DONE................")



# ......................... Earthquake Waveforms ...........................
#---------------------- Earthquakes folder -----------------------------------
print("............ Import WAVEFORMS................")
DB= glob.glob("INPUT-VE/Ground_motions/*.ASC")
ID=[]
Files=[]
for i,j in enumerate(DB):
    ID.append(i)
    Files.append(j)

D_all=[]
V_all=[]
AI_all=[]
PGA_all=[]
ak_all=[]
Ratio=[]
Collaps=[]

#.......................generate in VE by combining
Beta_VE_set=[]
Ru_VE_set=[]
Phy_VE_set=[]
Res_VE_set=[]
Ir_VE_set=[]


#.................. Generates Mata-solutions by combining and simulation................... 

print("............ Generates Meta-solutions by Combining and Simulation ................")

for i in range (0,int(N_comb)):
  j=random_multi_index(5, len(Alpha))
  #j=int(random.uniform(0,len(Alpha)))
  k=int(random.uniform(0,len(Files)))
  print("Comb_n.%d of %d" %((i+1),N_comb))
  #print("Combination n.%d ----> beta = %.2f; ru = %.2f; phi_deg = %.2f; res_deg = %.2f; Ir = %.2f"%(i+1,Alpha[j[0]], Ru[j[1]], Sfs[j[2]], Sre[j[3]], Ir[j[4]]))
  #print("                       Ground Motion file ID: %s"%(Files[k]))

  Beta_VE_set.append(Alpha[j[0]])
  Ru_VE_set.append(Ru[j[1]])
  Phy_VE_set.append(Sfs[j[2]])
  Res_VE_set.append(Sre[j[3]])
  Ir_VE_set.append(Ir[j[4]])
   
#---------------------- Select Earthquake  -----------------------------------
  a=linecache.getline("%s"%(Files[k]), 5)
  dt=re.findall(r'[\d\.\d]+',a)
  dt=array(dt,float)

  gm= genfromtxt("%s"%(Files[k]),skip_header=10,usecols=0)
  gm=cut_TH(gm,0.0001)#cut low values
  gm=-gm/9.81#in gravity acceleration ratio 
  t_all=arange(0,len(gm),1)*dt
  nn = len(gm)
   

#---------------------- Displacement  -----------------------------------

# Viscoplastic model parameters
  Smin = Sre[0] * 1.20  # The factor 1.20 can vary and defines the final residual value under dynamic conditions
  one = ones(len(gm))   # Creates an array of ones with the same length as gm
  beta = 3.6            # Hyperbolic degradation factor (ranges from 2 to 4)
  lamb = 0.30           # Viscous increment (non-Newtonian)
  r = 0.95              # Additional viscous increment (non-Newtonian)

#............Yeld acceleration................
  v=gm*0.#velocity before iteration
  d=gm*0.#displacement before iteration
  t=gm*0.#t* before iteration

  So=(Ir[j[4]]*Sre[j[3]])+((1.-Ir[j[4]])*Sfs[j[2]]) # Initial Shear Strength

  for ii in range (0,5):#iteration to converging the viscous behaviour depending by velocity
    S_vp= (beta*(So + (lamb*v**r))+(Smin*t**1.2))/(t**1.2+beta)# Grelle's viscoplastic model
    ak_o=(one+(0.66*gm))*(((1-Ru[j[1]])*S_vp)-Alpha[j[0]])/((Alpha[j[0]]*(1-Ru[j[1]])*S_vp)+one)#equivalent indefinite slope model

    t=t_star(ak_o,gm,t,dt)
    D_ak=gm-ak_o
    D_ak[D_ak<0]=0
    v=nmk(ak_o,gm,dt)[0]*9.81 #in "m/s"
  ak=ak_o*1.
  d=nmk(ak_o,gm,dt)[1]*9.81#in "m"
  D=d[-1]
  AI=sum((gm*9.81)**2)*dt*pi/(2*9.81)#in "m/s"
  #print("                      Max acceleration (m/s2) = %.4f; Arias Intensity = (m/s) = %.4f" % (amax(gm) * 9.81, AI))
  #print("                      Sliding Dispacement (m) = %.4f; Max sliding velocity (m/s) = %.4f; Time in coseismic sliding (s): = %.2f"%(D, amax(v), t[-1]))

#.................... Split solution into motion to displacement and motion to collapse ......................................
  if min(ak)>0.02 and (ak[0])>0:
    if AI==0:
      pass
    elif ak[0]>max(gm):
      pass
    else:
      D_all.append(D*AI**2)
      AI_all.append(AI)
      PGA_all.append(max(gm))
      ak_all.append(ak[0])
      Ratio.append(log10(max(gm)/ak[0]))
  else:
    pass

  if min(ak)<0.021 and (ak[0])>0.021:#.02g is the lower limit of critical acceleration reached for collapse
    Collaps.append(float((max(gm)/ak[0])*AI**2))
  else:
    pass

#...................... Save geomechanical setting of VE for radar diagram ........................................................
Beta_VE_set= array(Beta_VE_set)                                                                          
Ru_VE_set=array(Ru_VE_set)                                                                           
Phy_VE_set=array(Phy_VE_set)                                                                           
Res_VE_set=array(Res_VE_set)                                                                          
Ir_VE_set=array(Ir_VE_set)
  
header = "Slide angle ratio; Ru, Phy_peak_ratio, Residual_ratio,Residual factor)"
savetxt('OUTPUT-VE/Distribution for Training/VE-Geo-parameters_set.csc', transpose([Beta_VE_set.flatten(),\
                                                                              Ru_VE_set.flatten(),\
                                                                              Phy_VE_set.flatten(),\
                                                                              Res_VE_set.flatten(),\
                                                                              Ir_VE_set.flatten()]), delimiter=',', header=header, comments='')

#############################################################################################################################################################

D_allx=array(D_all)
D_all=D_allx.flatten()
D_all[D_all<0.0001]=0.0001
D_all=log10(D_all)#in meters
QQ90=func_quantil(Ratio,D_all,min(Ratio),max(Ratio),0.05, int(Perc_1))#0.05 assures monotonic increasing
QQ50=func_quantil(Ratio,D_all,min(Ratio),max(Ratio),0.05, int(Perc_2))
QQ20=func_quantil(Ratio,D_all,min(Ratio),max(Ratio),0.05, int(Perc_3))

Qx90=QQ90[0]
Qy90=QQ90[1]
for i in range(1,4):
  Qy90 = gaussian_filter1d(QQ90[1], i)

Qx50=QQ50[0]
Qy50=QQ50[1]
for i in range(1,4):
  Qy50 = gaussian_filter1d(QQ50[1], i)
Qx20=QQ20[0]
Qy20=QQ20[1]
for i in range(1,4):
  Qy20 = gaussian_filter1d(QQ20[1], i)


plt.plot(Ratio, D_all, '.')
plt.plot(Qx90,Qy90)
plt.plot(Qx50,Qy50)
plt.plot(Qx20,Qy20)
plt.xlabel('Log(Amax/Ak)')
plt.ylabel('Log(D/Ia2)')
plt.title('Training on metadata')
plt.savefig('OUTPUT-VE/Distribution for Training/Simuling_distribution.asc.png', dpi=600)
plt.show()

############################ Save simuling to displacement #######################################################################################################

header = "x_col=log10(Amax/Ak); y_col=log10(D/Ia2) with D in meter"
savetxt('OUTPUT-VE/Distribution for Training/Simuling_to_displacements.csc', transpose([Ratio, D_all]), delimiter=',', header=header, comments='')

percentil=90
header = "x_col=log10(Amax/Ak); y_col=log10(D/Ia2) with D in meter"
savetxt('OUTPUT-VE/Distribution for Training/Simuling_to_displ_confid_curve_%sth-perc.csc'%(percentil), transpose([Qx90,Qy90]), delimiter=',', header=header, comments='')

percentil=50
header = "x_col=log10(Amax/Ak); y_col=log10(D/Ia2) with D in meter"
savetxt('OUTPUT-VE/Distribution for Training/Simuling_to_displ_confid_curve_%sth-perc.csc'%(percentil), transpose([Qx50,Qy50]), delimiter=',', header=header, comments='')


percentil=20
header = "x_col=log10(Amax/Ak); y_col=log10(D*Ia2) with D in meter"
savetxt('OUTPUT-VE/Distribution for Training/Simuling_to_displ_confid_curve_%sth-perc.csc'%(percentil), transpose([Qx20,Qy20]), delimiter=',', header=header, comments='')


Col_distr=histogram(log10(Collaps), bins=100)
Col_freq=Col_distr[0]
Cum_Col_freq=cum_probability(Col_freq)
Col_range=Col_distr[1]


Y=Cum_Col_freq
X=Col_range[0:-1]

############################ Save simuling to collaps probability ######################################################################################################
header = "x_col=log10(Amax*Ia^2/Ac(t0)); y_col= Cumulate probability"
savetxt('OUTPUT-VE/Distribution for Training/Cumulate probability.csc', transpose([X, Y]), delimiter=',', header=header, comments='')     

plt.plot(X,Y)
plt.plot(X,gaussian_filter1d(Y, 0.5))
plt.title('Collaps probability')
plt.xlabel('Log(Amax*Ia^2/Ac(t0)')
plt.ylabel('Cumulative probability')
plt.savefig('OUTPUT-VE/Distribution for Training/Collaps_distribution.asc.png', dpi=600)
plt.show()        

print ("---------------------------------- END MODULE n.1 ---------------------------------------")
print (" Pass to Module n. 2 for Map-sets generation - or change Input parameters and re-running")


