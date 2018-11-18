import numpy as np
# from scipy.sparse import diags
# import scipy
# import scipy.linalg
# import scipy.sparse
# import scipy.sparse.linalg
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
# import time
# from astropy.units import dT

T=3*600 #cas v sekundach
R=0.1 #polomer v metrech
D=2*10**(-6) #tekute prostredi v m^2/s
# D=3*10**(-7) #pevne prostredi v m^2/s
c0=300 #pocatecni koncentrace v Bq/m^3

n, m=10**4, 90
tau, h=T/n, R/m

def test_stability():
    sigma=tau/h**2
    print("tau/h^2: "+str(sigma))
    print("prava strana: "+str(1/(2*D)))
    if sigma*D<1/2:
        print("Schema je stabilni")
        return True
    else:
        print("Schema NENI stabilni!!!!!")
        print("Zvetsi n (cas) nebo zmensi m (prostor).")
        print("Navyseni T muze tez zpusobit nestabilitu!")
        return False

def make_matrix_FTCS():
    #pp...pocatecni podminka
    #op...okrajova podminka
    sigma=tau/h**2
    A=np.zeros((m+1,m+1))
    A[0,0]=1-6*D*sigma
    A[0,1]=6*D*sigma
    for j in np.arange(1,m):
        A[j,j-1]=D*sigma*(1-1/j)
        A[j,j]=1-2*D*sigma
        A[j,j+1]=D*sigma*(1+1/j)
    A[m,m]=1
    return A


def vypocet_FTCS(c):
    #uloziste vysledku
    vysledek=np.zeros((n+1,m+1))
    vysledek[0]=c #ulozeni pocatecni podminky

    #matice soustavy
    A=make_matrix_FTCS()

    for t in np.arange(1,n+1):
        c=A.dot(c)
        vysledek[t]=c
    return vysledek


def vykresleni(vysledek):
    k, j=np.linspace(0,n,num=n+1,dtype=int), np.linspace(0,m,num=m+1,dtype=int)
    t, r=k*tau, j*h
    t, r=np.meshgrid(t, r)
    fig=plt.figure()
    ax=fig.gca(projection='3d')

    # surf=ax.plot_surface(t,r,np.transpose(vysledek), cmap=cm.coolwarm)
    surf=ax.plot_surface(t,r,np.transpose(vysledek))
    ax.set_xlabel('t [s]')
    ax.set_ylabel('r [m]')
    ax.set_zlabel('c [Bq/m^3]')

    # fig.colorbar(surf,shrink=0.5,aspect=5)

    plt.show()

def run(c):
    if test_stability():
        vysledek=vypocet_FTCS(c)
        vykresleni(vysledek)
        return vysledek
    else:
        return 0

def main():
    #Uloha a)
    #pp
    pp=np.zeros(m+1)
    #op
    pp[-1]=c0

    vysledek=run(pp)

    return 0

if __name__ == "__main__":
    main()
