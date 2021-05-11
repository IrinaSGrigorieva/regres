import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
from scipy.stats import f

def func (x,y,fi):
	Emax=1
	k=1
	return Emax*np.sin(2*k*x+2*np.pi*y)

def check (Model,Rensponse,dot,lvl):
	DA = statistics.mean(np.power((Model-Response),2))
	DB = noise*noise
	Fval= DA/DB
	fi_a = dot-lvl
	fi_b = dot-1
	qF = f.ppf(0.99, fi_a, fi_b)
	print ("Znachenie F-kriteria: "+ str(Fval))
	print ("Kvantil F-raspredelenia: " + str(qF))

def perv (PlanX_Vec, PlanY_Vec, Response):
	Plan = np.matrix([PlanX_Vec,PlanY_Vec, np.ones(len(PlanX_Vec))])
	Plan = Plan.transpose()
	koef = np.dot(np.dot(np.linalg.pinv(np.dot(Plan.transpose(),Plan)),Plan.transpose()),Response)
	koef = np.squeeze(np.asarray(koef))
	result = PlanX*koef[0]+PlanY*koef[1]+koef[2]
	check(result.ravel(),Response,len(PlanX_Vec),len(koef))
	return result

def vtor (PlanX_Vec, PlanY_Vec, Response):
	Plan2 = np.matrix([PlanX_Vec*PlanX_Vec,PlanY_Vec*PlanY_Vec,PlanX_Vec*PlanY_Vec,PlanX_Vec,PlanY_Vec, np.ones(len(PlanX_Vec))])
	Plan2 = Plan2.transpose()
	koef2 = np.dot(np.dot(np.linalg.pinv(np.dot(Plan2.transpose(),Plan2)),Plan2.transpose()),Response)
	koef2 = np.squeeze(np.asarray(koef2))
	result2 = PlanX*PlanX*koef2[0] + PlanY*PlanY*koef2[1] + PlanX*PlanY*koef2[2] + PlanX*koef2[3] + PlanY*koef2[4] + koef2[5]
	check(result2.ravel(),Response,len(PlanX_Vec),len(koef2))
	return result2


xs=np.linspace(-1,0,50)
ys=np.linspace(-0.9,-0.2,50)
X,Y = np.meshgrid(xs,ys)
fi = np.linspace(0,180./np.pi,50)
Z = func(X,Y,fi)
xp = np.linspace(-1,0,10)
yp = np.linspace(-0.9,-0.2,10)
PlanX, PlanY = np.meshgrid(xp,yp)
PlanZ = func(PlanX,PlanY,fi)
noise = 0.1
Response = func(PlanX,PlanY,fi)+noise*np.random.randn(len(xp),1)
Response = (Response.ravel()).transpose()
PlanX_Vec = PlanX.ravel()
PlanY_Vec = PlanY.ravel()
result=perv(PlanX_Vec, PlanY_Vec, Response)
result2=vtor(PlanX_Vec, PlanY_Vec, Response)
fig= plt.figure()
ax= fig.add_subplot(121, projection= '3d')
plt.title(u"Аппроксимация 1-го порядка")
ax.set_xlim([-1, 0])
ax.set_ylim([-0.9,-0.2])
ax.set_zlim([-1, 1])
ax.scatter(PlanX,PlanY,PlanZ, c="r")
surf1=ax.plot_surface(X,Y,Z,color='grey',linewidth=0,antialiased='True',rstride=3,cstride=3,alpha=0.9)
surf2 = ax.plot_surface(PlanX,PlanY,result,color='blue',linewidth=0,antialiased='True',rstride=3,cstride=3)
ax= fig.add_subplot(122, projection= '3d')
plt.title(u"Аппроксимация 2-го порядка")
ax.set_xlim([-1, 0])
ax.set_ylim([-0.9,-0.2])
ax.set_zlim([-1, 1])
ax.scatter(PlanX,PlanY,PlanZ, c="r")
surf1=ax.plot_surface(X,Y,Z,color='grey',linewidth=0,antialiased='True',rstride=3,cstride=3,alpha=0.5)
surf3 = ax.plot_surface(PlanX,PlanY,result2,color='green',linewidth=0,antialiased='True',rstride=3,cstride=3)
plt.show()
