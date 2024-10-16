from numpy import *
import re
import linecache
import os
import string
from scipy.ndimage import gaussian_filter
from scipy import ndimage, datasets
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterExponent # <-- one new import here
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import warnings


def rad(a):
  a=a*pi/180
  return a

def deg(a):
  a=a*180/pi
  return a

def one(a):
    a=a/a
    return a
def zero(a):
    a*0
    return a


def slope_gen(DEM, cellsize):#return slope from DTM
    px, py = gradient(DEM, cellsize)
    slope = sqrt(px ** 2 + py ** 2)
    # If needed in degrees, convert using
    slope_deg = degrees(arctan(slope))
    return slope_deg

def slope_pen(DEM, cellsize):#return slope from DTM
    px, py = gradient(DEM, cellsize)
    slope = sqrt(px ** 2 + py ** 2)
    # If needed in degrees, convert using
    #slope_deg = degrees(arctan(slope))
    return slope

def modal_split(SLOPE,mu):#SLOPE as tangent
    y=exp((SLOPE+mu)**10)
    a=SLOPE*y
    return a
  

#............................... Optimization on Mesri model...........................
class ContinueI(Exception):
    pass

continue_i = ContinueI()

def model(X,m_o,S_100):
  f= X*tan(rad(S_100))*(100/X)**(1-m_o)
  return f
@vectorize
def errfunc(m_o,S_100,t_coul):
  y = model(X,m_o,S_100)
  sse = sum((y - t_coul)**2)
  return sse

def interpp(x,X,Ytarget):
  a=interp(x, X, Ytarget)
  return a



###################################################################################################################################################################             
######################################################   MODULE n. 2 MAP-SETS GENERATION   #################################################################
###################################################################################################################################################################

print('****************** MODULE n. 2 MAP-SETS GENERATION ****************************')

###############    INSTRUCTIONS   #####################################
ru=0.27

perc=20

PR=475                   

################################## Depth parameter ######################################################
H=20# in (m)
#**************** natural and satured weight (in kN/m3 = kPa/m3)  ***********
gd=18
gs=21
################################## Strength parameters ##################################################
#************* Mean strength out-landlsides
phi=16.7
c=20.03

#************ Mean strength in-landslides
phi_fs=16.7
phi_re=9.98

#maximum and minimum values of PGA abserved in the PGA (m/s2) (475 years) points interpolate for create PGA map.
max_PGA_point=3.077
min_PGA_point=1.853
#maximum and minimum values of PGA abserved in the PGA (m/s2) point using for define local hazard: town, village, other....
PGA_target_475=2.615
PGA_target_975=3.365

########################################################################################################                     
path1 = "OUTPUT-RE"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);
  
path1 = "OUTPUT-RE\Ir_distribution"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

warnings.filterwarnings('ignore')
path1 = "OUTPUT-RE\PGA_distribution"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

path1 = "OUTPUT-RE\Critical_acceleration"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

path1 = "OUTPUT-RE\DISPLACEMENTS"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);


path1 = "OUTPUT-RE\DISPLACEMENTS\Means"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

path1 = "OUTPUT-RE\DISPLACEMENTS\Means\Statistic"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);

path1 = "OUTPUT-RE\COLLAPSES"
if os.path.exists(path1):
  pass
else:
  os.makedirs(path1, 755);
  
###############################################################################################################  

header = linecache.getline('INPUT-RE/DEM_20m.asc' ,1)
header +=linecache.getline('INPUT-RE/DEM_20m.asc', 2)
header +=linecache.getline('INPUT-RE/DEM_20m.asc', 3)
header +=linecache.getline('INPUT-RE/DEM_20m.asc', 4)
header +=linecache.getline('INPUT-RE/DEM_20m.asc', 5)
header +=linecache.getline('INPUT-RE/DEM_20m.asc', 6)

cellsize=re.sub("[^\d\.]", "", (linecache.getline('INPUT-RE/DEM_20m.asc' ,5)))
cellsize=array(cellsize,int)

DTM=loadtxt('INPUT-RE/DEM_20m.asc', skiprows=6)#DTM map
Land=loadtxt('INPUT-RE/Landslides_20m.asc',skiprows=6)#Landslides map
Int_rate=loadtxt('INPUT-RE/InSAR_distribution_20m.asc',skiprows=6)#PS InSar map distribution

Land=array(Land,float)
Int_rate=array(Int_rate,float)
Int_rate=abs(Int_rate)

n_row= len(DTM)
n_col=DTM.size/n_row

DTM[DTM<0]=0
slope = slope_gen(DTM,cellsize)
slopep=slope_pen(DTM,cellsize)
DTM= gaussian_filter(DTM, sigma=0.2)#delete outlies 

slope_p=tan(slope*3.14/180)
slopex=modal_split(slope_p,0.5)

savetxt('OUTPUT-RE\Slope_angle_deg.asc', deg(arctan(slopex)), \
  header=(header), comments='', fmt="%1.2f")

print('PLOT: Digital Elevation Model')
plt.figure(figsize=(7, 7))              
plt.imshow(DTM, interpolation='kaiser')
plt.colorbar()
plt.title("Digital Elevation Model (m)")
plt.show()

print('PLOT: Slope angle')
plt.figure(figsize=(7, 7))
plt.imshow(slope, interpolation='kaiser')
plt.colorbar()
plt.title("Slope angle (°)")
plt.show()

print('PLOT: Landslides activity Map')
plt.figure(figsize=(7, 7))
plt.imshow(Land, interpolation='kaiser')
plt.colorbar()
plt.title("Landslides activity")
plt.show()


#************ Residual Factor Ranking and Combination************************
print('................ Ranking for landslides............................')

Kl=(Land*1.)
Vg=Int_rate*0.1
Vg=around(Vg, 3)
Vg[Vg>1.]=1.
A=(1-Vg)
B=A**Vg
Ir=1-(B/(1+Kl))

print('PLOT: Residual Factor Map')
plt.figure(figsize=(7, 7))
extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
ax = plt.gca()
step=5
xticks = arange(extent[0], extent[1], step)
ax.set_xticks(xticks)

plt.imshow(Ir, cmap='Purples', interpolation='nearest',extent=extent, aspect='equal')
plt.xticks(fontsize=10)  
plt.yticks(fontsize=10)
cbar=plt.colorbar(orientation='vertical', label='Residual factor')
cbar.set_label('Residual factor', fontsize=11)
cbar.ax.tick_params(labelsize=10)
plt.title("Residual Factor")

plt.savefig('OUTPUT-RE\Ir_distribution\Distribution_Ir_%sm_res.asc.png'%(cellsize), dpi=600)
plt.show()
savetxt("OUTPUT-RE\Ir_distribution\Distribution_Ir_%sm_res.asc"%(cellsize), Ir, header=(header),fmt='%.2f',comments='', delimiter=' ')



#####################################   PGA _DISTRIBUTION_FACTOR ###################################################################
print('................ SSF Computation............................')
if PR==475:
  PGA_map=loadtxt('INPUT-RE/PGA_475.asc', skiprows=7)
  Dc=PGA_map/PGA_target_475 #  475y return period on target site 
else:
  PGA_map=loadtxt('INPUT-RE/PGA_975.asc', skiprows=7)
  Dc=PGA_map/PGA_target_975 #  975y return period on target site
PGA_map=array(PGA_map,float)


#.....................................................................................................................................
PGA_mean= (amax(PGA_map)+amin(PGA_map))/2
medd=(max_PGA_point+min_PGA_point)/2
F_max=(max_PGA_point-medd)/(amax(PGA_map)-PGA_mean)# coeff di normalizzazione per i valori maggiori della media; 3.077 è il valore massimo reale riscontrato nell'area 
F_min=(min_PGA_point-medd)/(amin(PGA_map)-PGA_mean)# coeff di normalizzazione per i valori minoridella media; 1.853 è il valore minimo reale riscontrato nell'area

PGA_map[PGA_map > PGA_mean] = PGA_mean + F_max * (PGA_map[PGA_map > PGA_mean] - PGA_mean)
PGA_map[PGA_map < PGA_mean] = PGA_mean + F_max* (PGA_map[PGA_map < PGA_mean] - PGA_mean)

print('PLOT: PGA Distribution Factor Map')
plt.figure(figsize=(7, 7))
extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
ax = plt.gca()
step=5
xticks = arange(extent[0], extent[1], step)
ax.set_xticks(xticks)
plt.imshow(Dc, cmap='terrain', interpolation='nearest',extent=extent, aspect='equal')
plt.xticks(fontsize=10) 
plt.yticks(fontsize=10)
cbar=plt.colorbar(orientation='vertical', label='PGA Distribution Factor')
cbar.set_label('PGA Distribution Factor', fontsize=11)
cbar.ax.tick_params(labelsize=10)


savetxt("OUTPUT-RE\PGA_distribution\PGA_map_%sm_%sm_res.asc"%(PR,cellsize), PGA_map, header=(header),fmt='%.2f',comments='', delimiter=' ')


savetxt("OUTPUT-RE\PGA_distribution\PGA_SF_map_%sm_%sm_res.asc"%(PR,cellsize), Dc, header=(header),fmt='%.2f',comments='', delimiter=' ')
plt.savefig('OUTPUT-RE\PGA_distribution\PGA_SF_map_%sm_%sm_res.png'%(PR,cellsize), dpi=600)
plt.show()



################################ Earthquake parameters ###########################
if PR==475:
  file = open('INPUT-RE/parameter_EQ_set/normativa_475_anni.txt', 'r')
else:
  file = open('INPUT-RE/parameter_EQ_set/normativa_975_anni.txt', 'r')

earthQ = csv.DictReader(file)

PGA = []
PGV=[]

AI=[]
Pp=[]
AvSa=[]

if PR==475:
  data = loadtxt('INPUT-RE/parameter_EQ_set/normativa_475_anni.txt',dtype=str,delimiter=';',skiprows=1)
else:
  data = loadtxt('INPUT-RE/parameter_EQ_set/normativa_975_anni.txt',dtype=str,delimiter=';',skiprows=1)
data=transpose((data))
PGA.append(data[1])

PGV.append(data[2])
AI.append(data[4])
Pp.append(data[5])
AvSa.append(data[6])

#.................ATTENTION!!!!!!!you do not change zero values after ...............
PGA=array(PGA[0],float)
PGV=array(PGV[0],float)
AI=array(AI[0],float)
Pp=array(Pp[0],float)
AvSa=array(AvSa[0],float)


#-----------------  Shear strength non linear optimization ----------------------------

sig_n=arange(30, 600, 10)#this is "X" values in the midel
t_coul=c+(sig_n*tan(rad(phi)))#this is the "Y target" values (known data)

m_o,S_100 = 0.5, 50 #initial coefficents


popt, pcov = curve_fit(model, sig_n, t_coul)
#print ('regression coefficient for phi=', popt)
errorx = mean((t_coul-model(sig_n, *popt))**2)


######################################  LOOP _Computation for Eartquakes parameters and resolution size ################
print('................ Hazard MAP-sets............................')   
D3x=[]
for j in range(0, len(PGA)):
  slopep=slope_pen(DTM,cellsize)
  d3xx=[]
  for i in range (0,1):
    cls_0=int(cellsize*(i+1))
    slopep= gaussian_filter(slopep, sigma=i)
    x=log10(cls_0**2/slopep**0.5)
    H=(9.07*x**3)-(49.44*x**2)+(88.94*x)-(50.62)
    H[H>120]=120
    xx=(slopep**2)/(cls_0**0.5)
    betap=0.9216 + (0.16*log(xx))
    betap[betap<0.01]=0.01
    
    #-----------------------------return secant phi value---------------------------------------
    s1_n=H*average((gd,gs))*(1-ru)#efficacy normal stress at base ot the block 1
    fisec=deg(arctan(model(s1_n, popt[0],popt[1])/s1_n))#secant frictional angle of the block n.1
    #----------------------------------------------------------------------------
    mu=tan(rad(fisec))
    mu_r=tan(rad(phi_fs))-(Ir*(tan(rad(phi_fs))-tan(rad(phi_re))))

    #************  Mask from phi with slope angle depending *******************
    if i==0:
      cut_r=mu_r/(slopep)
      cut_r[cut_r<0.99]=0
      cut_r[cut_r>0.]=1.

    cut_in=cut_r*Land

    #************ Compute DISPLACEMENTS *****************************
    D1=[]#from PGA
    D2=[]#from PGV
    D3=[]#from AI
    D4=[]#from Pp
    D5=[]#from AvSa
  
  #****************  compute ak  ****************
  
    ak=(1+(0.66*(Dc*PGA[0]/9.81)))*(((1-ru)*mu)-betap)/((betap*(1-ru)*mu)+one(betap))
    ak_r=(1+(0.66*(Dc*PGA[0]/9.81)))*(((1-ru)*mu_r)-betap)/((betap*(1-ru)*mu_r)+one(betap))
    ak_r=minimum(ak,ak_r)

    ############ Compute displacement on training curve  #############

    #***** import trained curve ********
    training_90 = loadtxt('OUTPUT-VE/Distribution for Training/Simuling_to_displ_confid_curve_%sth-perc.csc'%(perc),dtype=str,delimiter=',',skiprows=1)
    
    x_t=array(training_90[:,0], float)#....by curve
    y_t=array(training_90[:,1], float)#.....by  curve

    ak[ak<0.01]=0.01
    ak_r[ak_r<0.01]=0.01
    


#######save and plot yeald acceleration map ###################################
 
    if j==0:
      print('PLOT: Critical Acceleration Map')
      akk=ak_r*1.
      akk[akk==0]=nan
      ak_j=ak_r*cut_r#cut_in
      ak_j[ak_j>5]=5
      ak_j[ak_j==0]=nan
          
      plt.figure(figsize=(7, 7))
      extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
      ax = plt.gca()
      step=5
      xticks = arange(extent[0], extent[1], step)
      ax.set_xticks(xticks)

      norm = mcolors.Normalize(vmin=0, vmax=0.35)
      

      
      plt.imshow(ak_j, cmap='hot', interpolation='nearest',extent=extent, aspect='equal', norm=norm)
      plt.xticks(fontsize=10)  # Modifica la dimensione delle etichette sull'asse X
      plt.yticks(fontsize=10)
      cbar=plt.colorbar(orientation='vertical', label='Critical Acceleration (g)')
      cbar.set_label('Critical Acceleration (g)', fontsize=11)
      cbar.ax.tick_params(labelsize=10)
        
      savetxt("OUTPUT-RE\Critical_acceleration\Distribution_Yeld_Acc_Ru=%.2f_%sm_res.asc"%(ru,cellsize), ak_j, header=(header),fmt='%.2f',comments='', delimiter=' ')
      plt.savefig('OUTPUT-RE\Critical_acceleration\Distribution_Yeld_Acc_Ru=%.2f_%sm_res.png'%(ru,cellsize), dpi=600)

      plt.show()
    else:
      pass

####################################################################################

############# RESOLVE DISPLACEMENT MAP ################################################
    print('PLOT:Displacement_map n.%d'%(j+1))

    X_t=(Dc*PGA[j]/9.81)/ak_j
    X_t[X_t<0.0001]=0.0001
    X_t=log10(X_t)
    print (PGA[j],AI[j])
    

    Y_t=interpp(X_t,x_t,y_t)
    d3r=100*(10**Y_t)/(Dc*(AI[j])**2)
    d3r[d3r<1]=1
    
    d3f=d3r*cut_in
    d3xx.append(d3f)
  
    fold_dir='DISPLACEMENTS\Dist_%sth_%syears'%(perc,PR)
    file_path = "OUTPUT-RE/%s/Disp_%sth__Earthquake%sm-0%d_PGA=%.2f__Ru=%.2f_%s.txt" % (fold_dir,perc,PR,j+1, PGA[j], ru,cellsize)

   
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

  
    savetxt(file_path, d3f, header=header, fmt='%.2f', comments='', delimiter=' ')

    plt.figure(figsize=(7, 7))
    extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
    ax = plt.gca()
    step=5
    xticks = arange(extent[0], extent[1], step)
    ax.set_xticks(xticks)

    d3f = gaussian_filter(d3f, sigma=0.2, mode= "constant")
    
    cmap = plt.cm.jet
    cmap_array = cmap(arange(cmap.N))
    cmap_array[0] = [1, 1, 1, 1]  
    new_cmap = mcolors.ListedColormap(cmap_array)

    plt.imshow(d3f, cmap=new_cmap,norm=LogNorm(vmin=10**-1, vmax=10**2.5), interpolation='nearest',extent=extent, aspect='equal')

    plt.xticks(fontsize=10)  
    plt.yticks(fontsize=10)
    cbar=plt.colorbar(orientation='vertical', label='Displacement (cm)')
    cbar.set_label('Displacement (cm)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)   

    plt.savefig('OUTPUT-RE\%s\Disp_%sth__Earthquake%sm-0%d_PGA=%.2f__Ru=%.2f_%s.png'%(fold_dir,perc,PR,j+1, PGA[j], ru,cellsize), dpi=600)
    plt.show()
        
print('Mean Displacement')
    
plt.figure(figsize=(7, 7))
extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
ax = plt.gca()
step=5
xticks = arange(extent[0], extent[1], step)
ax.set_xticks(xticks)
Mean=mean(d3xx,axis=0)
Mean[Mean>10**3]=10**3
Mean = gaussian_filter(Mean, sigma=0.2, mode= "constant")

cmap = plt.cm.jet
cmap_array = cmap(arange(cmap.N))
cmap_array[0] = [1, 1, 1, 1] 
new_cmap = mcolors.ListedColormap(cmap_array)

plt.imshow(Mean, cmap=new_cmap,norm=LogNorm(vmin=10**-1, vmax=10**2.5), interpolation='nearest',extent=extent, aspect='equal')#, norm=norm)

plt.xticks(fontsize=10) 
plt.yticks(fontsize=10)
cbar=plt.colorbar(orientation='vertical', label='Displacement (cm)')
cbar.set_label('Displacement (cm)', fontsize=11)
cbar.ax.tick_params(labelsize=10)   
   
savetxt("OUTPUT-RE\DISPLACEMENTS\Means\Mean_%sth_Earthq_%sm_Ru=%.2f_%sm_res.asc"%(perc, PR,ru,cellsize), mean(d3xx,axis=0), header=(header),fmt='%.2f',comments='', delimiter=' ')
plt.savefig('OUTPUT-RE\DISPLACEMENTS\Means\Mean_%sth_Earthq_%sm_Ru=%.2f_%sm_res.png'%(perc, PR,ru,cellsize), dpi=600)
plt.show()

MEAN=log10(mean(d3xx,axis=0))
MEAN=MEAN.flatten()
MEAN= MEAN[~isnan(MEAN)]
MEAN= MEAN[~isinf(MEAN)]
MEAN= MEAN[MEAN>=-1.]
#plot and save hystogram on Mean
bins=[-1,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0]
bins_plus=[-1,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,10]
plt.hist(MEAN.flatten(),bins=bins ,color='red', edgecolor='black')
plt.xlabel('Displacement (cm)')
plt.ylabel('Frequency (n.)')
plt.savefig('OUTPUT-RE\DISPLACEMENTS\Means\Statistic\Mean_%sth_Earthq_%sm_Ru=%.2f.png'%(perc, PR,ru), dpi=600)
x=histogram(MEAN, bins=bins)[1]
y=histogram(MEAN, bins=bins_plus)[0]

savetxt('OUTPUT-RE\DISPLACEMENTS\Means\Statistic\Mean_%sth_Earthq_%sm_Ru=%.2f.csv'%(perc, PR,ru), transpose((x, y)), delimiter=',',fmt='%.2f', header='log10 of x_data, y_frequency')
plt.show()


################################ Collaps ########################################################################
print('PLOT: Probability Collaps MAP') 
x_clp = array(loadtxt('OUTPUT-VE/Distribution for Training/Cumulate probability.csc',dtype=str,delimiter=',',skiprows=1,usecols=0),float)
y_clp = array(loadtxt('OUTPUT-VE/Distribution for Training/Cumulate probability.csc',dtype=str,delimiter=',',skiprows=1,usecols=1),float)

y_clp_sm=convolve(y_clp, ones(5) / 5, mode='same')

y_clp_sm[-4::]=y_clp_sm[-5]#correzione anomalia di media mobile su ultimo valore

X_map=log10(Dc**3)+(log10(median(AI**2)*(median(PGA/9.81)/ak_j)))


Ycolp= interpp(X_map,x_clp,y_clp_sm)
Ycolp=Ycolp
Ycolp=Ycolp

plt.plot(x_clp,y_clp)
plt.plot(x_clp,y_clp_sm)
plt.savefig('OUTPUT-RE\\COLLAPSES\Probability collaps_training_%sm_res.asc.png'%(cellsize), dpi=600)

plt.show()
            
header = "x_col=Log10(Amax*Ia^2/Ak); y_col_0= cumulate probability, y_col_1= cumulate smoothed probability"
savetxt('OUTPUT-VE/Distribution for Training\Cumulate probability_smoothed_%sy_ru=%.2f.csc'%(PR,ru), transpose([x_clp, y_clp,y_clp_sm]), delimiter=',', header=header, comments=';',fmt='% 4f')            

plt.figure(figsize=(7, 7))
extent = [0, n_col/1000 *cellsize, 0, n_row/1000 * cellsize]
ax = plt.gca()
step=5
xticks = arange(extent[0], extent[1], step)
ax.set_xticks(xticks)

norm = mcolors.Normalize(vmin=0, vmax=1.)
plt.imshow(Ycolp, cmap='pink_r', interpolation='nearest',extent=extent, aspect='equal', norm=norm)

plt.xticks(fontsize=10)  
plt.yticks(fontsize=10)
cbar=plt.colorbar(orientation='vertical', label='Probability)')
cbar.set_label('Probability', fontsize=11)
cbar.ax.tick_params(labelsize=10)


plt.savefig('OUTPUT-RE\COLLAPSES\Probability collaps_%sy_ru=%.2f_%sm_res.png'%(PR,ru,cellsize), dpi=600)
plt.show()

savetxt("OUTPUT-RE\COLLAPSES\Probability collaps_%sy_ru=%.2f_%sm_res.asc"%(PR,ru,cellsize), Ycolp, header=(header),fmt='%.2f',comments='', delimiter=' ')

print ("------------------------------ END MODULE n.2 -------------------------------------")
print ("                  Bye or change Input parameters and re-running")


   
