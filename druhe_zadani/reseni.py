import numpy as np
import scipy
import scipy.sparse.linalg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from matplotlib import cm
import matplotlib.animation as animation
import time
import math
# from astropy.units import dT

# T=600*3 #cas v sekundach
T=10*600
#po cca 300 min je to v rovnovaznem stavu
R=0.1 #polomer v metrech
c0=300 #pocatecni koncentrace v Bq/m^3

# n=5*10**4
# m=200
# n=2000
# m=1000
m=1000
n=int(2/600*T*m)
tau=T/n
h=R/m
sigma=tau/h**2

prem_konst=math.log(2)/(3.8235*24*60*60)

def test_stability(D):
    sigma=tau/h**2
    print("-------------------------------------------")
    print("TEST STABILITY")
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

def make_matrix_FTCS(D):
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

def vypocet_FTCS(c,D):
    start=time.time()

    # uloziste vysledku
    vysledek=np.zeros((n+1,m))
    #DO VYSLEDKU SE OKR. PODM. NEUKLADA!!!

    vysledek[0]=c[0:-1] #ulozeni pocatecni podminky
    # ulozeni_obrazku(c,0)

    #matice soustavy
    A=make_matrix_FTCS(D)

    #vypocet
    for k in np.arange(1,n+1):
        c=A.dot(c)
        vysledek[k]=c[0:-1]
        # ulozeni_obrazku(c,k)
    stop=time.time()
    print("-------------------------------------------")
    print("Spotrebovany cas pro nalezeni numerickeho reseni: "+str(stop-start))
    return vysledek


def vypocet_CN(c,D,theta=1/2):
    start=time.time()
    # uloziste vysledku
    vysledek=np.zeros((n+1,m))
    #DO VYSLEDKU SE OKR. PODM. NEUKLADA!!!
    vysledek[0]=c[0:-1] #ulozeni pocatecni podminky

    j_array = np.linspace(0, m, m+1)
    pom1=D*sigma*theta
    pom2 = D*(1. - theta)

    def make_matrix():
        subDiag = np.zeros(m)
        superDiag = np.zeros(m)
        mainDiag = np.zeros(m+1)

        mainDiag[0] = 1 + 6*pom1
        # mainDiag[0] = 1 + 6*pom1 + tau*prem_konst*theta
        superDiag[0] = -6*pom1

        subDiag[:-1] = -pom1*(1 - 1/j_array[1:-1])
        mainDiag[1:-1] = np.ones(m-1)*(1 + 2*pom1)
        # mainDiag[1:-1] = np.ones(m-1)*(1 + 2*pom1 + tau*prem_konst*theta)
        superDiag[1:] = -pom1*(1 + 1/j_array[1:-1])

        mainDiag[-1]=1
        return scipy.sparse.diags([subDiag, mainDiag, superDiag], [-1, 0, 1], format='csc')

    A=make_matrix()

    def jeden_krok(cOld):
        """ Args:
                cOld(array): The entire solution vector for the previous timestep, n.

            Returns:
                cNew(array): solution at timestep n+1
        """
        PS = np.zeros(m+1)
        PS[0] = (1 - 6*pom2*sigma)*cOld[0] + 6*pom2*sigma*cOld[1]
        # PS[0] = (1 - 6*pom2*sigma - tau*prem_konst*(1-theta))*cOld[0] + 6*pom2*sigma*cOld[1]

        a = pom2*sigma*(1 - 1./(j_array[1:-1]))*cOld[:-2]
        b = (1 - 2*pom2*sigma)*cOld[1:-1]
        # b = (1 - 2*pom2*sigma - tau*prem_konst*(1-theta))*cOld[1:-1]
        c = pom2*sigma*(1 + 1/(j_array[1:-1]))*cOld[2:]
        PS[1:-1] = a + b + c

        PS[-1]=cOld[-1]

        cNew = scipy.sparse.linalg.spsolve(A, PS)
        return cNew

    for k in np.arange(1,n+1):
        # print(len(c))
        c=jeden_krok(c)
        # print(len(c))
        vysledek[k]=c[0:-1]
        # ulozeni_obrazku(c,k)

    stop=time.time()
    print("-------------------------------------------")
    print("Spotrebovany cas pro nalezeni numerickeho reseni: "+str(stop-start))
    print("-------------------------------------------")
    return vysledek

def animace(vysledek,op,interval=1,ulozit=False):
    j=np.linspace(0,m-1,num=m,dtype=int)
    # plt.cla()
    # plt.clf()
    # plt.close()
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(0, R+R/10), ylim=(-5,c0+50))
    ax.grid()
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [Bq/m$^3$]')
    # k=np.linspace(0,n,num=n+1,dtype=int)
    line, = ax.plot(j*h, vysledek[0],)
    ax.axvline(x=R, color='k')
    ax.axhline(y=op,xmin=0.1/0.11,linestyle=':',linewidth=2, color='r')
    text_min = ax.text(0.02,0.97, "", ha="left", va="center", transform=ax.transAxes)
    text_sek = ax.text(0.02,0.94, "", ha="left", va="center", transform=ax.transAxes)

    def prepocet(i):
        s=i*tau
        return math.floor(s/60),s

    def animate(i):
        line.set_ydata(vysledek[i])  # update the data
        minuty,sekundy=prepocet(i)
        text_min.set_text("$t=$ %.0f min" % minuty)
        text_sek.set_text("$t=$ %.0f s" % sekundy)
        return line, text_min,text_sek

    def init():
        line.set_ydata('')
        text_min.set_text('')
        text_sek.set_text('')
        return line, text_min, text_sek

    anim = animation.FuncAnimation(fig, animate, np.arange(0, n+1), init_func=init,
                                interval=interval, blit=True, repeat=False)
    if ulozit:
        anim.save('animace.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

def animace_obe_D(vysledek,op,interval=1,ulozit=False):
    # PRVNI MUSI BYT VZDY TEKUTE PROSTREDI (v promenne vysledek)
    #TO DO:
    #Proc se otevrou dve okna animaci???
    a=vysledek[0]
    b=vysledek[1]
    j=np.linspace(0,m-1,num=m,dtype=int)
    # plt.cla()
    # plt.clf()
    # plt.close()
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(0, R+R/10), ylim=(-5,c0+50))
    ax.grid()
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [Bq/m$^3$]')
    # k=np.linspace(0,n,num=n+1,dtype=int)
    ax.axvline(x=0.1, color='k')
    line1, = ax.plot(j*h, a[0])
    line2, = ax.plot(j*h, b[0])
    ax.axhline(y=op,xmin=0.1/0.11,linestyle=':',linewidth=2, color='r')
    text_min = ax.text(0.02,0.97, "", ha="left", va="center", transform=ax.transAxes)
    text_sek = ax.text(0.02,0.94, "", ha="left", va="center", transform=ax.transAxes)

    def prepocet(i):
        s=i*tau
        return math.floor(s/60),s

    def animate(i):
        line1.set_ydata(a[i])  # update the data
        line1.set_label('tekuté prostředí')
        line2.set_ydata(b[i])  # update the data
        line2.set_label('pevné prostředí')
        # legend.remove()
        legend = plt.legend(loc=9)
        minuty,sekundy=prepocet(i)
        text_min.set_text("$t=$ %.0f min" % minuty)
        text_sek.set_text("$t=$ %.0f s" % sekundy)
        return line1, line2, text_min, text_sek, legend

    def init():
        line1.set_ydata('')
        line2.set_ydata('')
        text_min.set_text('')
        text_sek.set_text('')
        return line1, line2, text_min, text_sek

    anim = animation.FuncAnimation(fig, animate, np.arange(0, n+1), init_func=init,
                                interval=interval, blit=True, repeat=False)
    if ulozit:
        anim.save('animace.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

def run_FTCS(pp,D,interval_animace=10**(-n),ulozit_animace=False):
    if test_stability(D):
        start=time.time()
        vysledek=vypocet_FTCS(pp,D)
        # animace(vysledek,pp[-1],interval=interval_animace,ulozit=ulozit_animace)
        # vykresleni(vysledek)
        stop=time.time()
        print("-------------------------------------------")
        print("Spotrebovany cas celkove: "+str(stop-start))
        print()
        return vysledek
    else:
        print('SCHEMA NENI STABILNI, VIZ VYPIS VYSE!!!')
        return 0

def main():
    print("SPUSTENI PROGRAMU")
    vysledek_celk=0 #bude matice, pokud probehne nejaky vypocet
    D_t=2*10**(-6) #tekute prostredi v m^2/s
    D_p=3*10**(-7) #pevne prostredi v m^2/s

    #Uloha a)
    #pp
    pp_a=np.zeros(m+1)
    #op
    pp_a[-1]=c0
    # vysledek_tekute_a=run_FTCS(pp_a,D_t)
    # vysledek_pevne_a=run_FTCS(pp_a,D_p)
    vysledek_tekute_a=vypocet_CN(pp_a,D_t)
    vysledek_pevne_a=vypocet_CN(pp_a,D_p)

    #Uloha b)
    #pp
    pp_b=np.zeros(m+1)+c0
    #op
    pp_b[-1]=0
    # vysledek_tekute_b=run(pp_b,D_t)
    # vysledek_pevne_b=run(pp_b,D_p)

    vysledek_celk=[vysledek_tekute_a, vysledek_pevne_a]
    animace_obe_D(vysledek_celk,pp_a[-1],interval=1)

    #CN METODA
    # vysledek_celk=vypocet_CN(pp_a,D_t)
    # animace(vysledek_celk, pp_a[-1])
    return vysledek_celk


if __name__ == "__main__":
    vysledek_celk=main()
