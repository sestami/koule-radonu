"""
SPOUSTENI PROGRAMU: python resesi.py d k_u k_v
d je tloustka steny v intervalu (0,R]
k_u je koeficient transportu z steny do koule uvnitr v m/s
k_v je koeficient transportu do steny z vnejsku v m/s
"""
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
import pdb
import sys
# from astropy.units import dT

# T=600*3 #cas v sekundach
T=1*600
#po cca 300 min je to v rovnovaznem stavu
R=0.1 #polomer v metrech
# R=10 #polomer v centimetrech
c0=3000 #pocatecni koncentrace v Bq/m^3

D_t=2*10**(-6) #tekute prostredi v m^2/s
D_p=3*10**(-7) #pevne prostredi v m^2/s
prem_konst=math.log(2)/(3.8235*24*60*60)

#relaxacni doby

if len(sys.argv)>3:
    k_u=float(sys.argv[2])
    k_v=float(sys.argv[3])
    if float(sys.argv[1])<=R and float(sys.argv[1])>0:
        d=float(sys.argv[1]) #d\in(0;0.1]
        print("tloustka steny="+str(d))
        print("koeficient transportu ze steny dovnitr="+str(k_u))
        print("koeficient transportu z vnejsku do steny="+str(k_v))
    else:
        print("Tloustka steny zadana mimo interval (0;R]!")
        print("Nyni nastavena na 0.01")
        print("R="+str(R))
        d=R/2
else:
    d=R/2
    k_u=1E-4
    k_v=1
    print("Nedostatecny pocet vstupnich argumentu!")

# d=R/2
# l=10**(-5)
print("Argumenty nastaveny na: d="+str(d)+"; k_u="+str(k_u)+"; k_v="+str(k_v))

#pristup na dalsich dvou radcich doladit
# m_pouzite=100 #pocet bodu pouzite prostorove site
# m=int(R/d*m_pouzite) #pocet bodu prostorove site

m=2000
# n=int(8/600*T*m*np.log10(R/d*100000)) #pocet bodu casove site
# n=int(8/600*T*m*np.log10(R/d*10)+(300-c0)/1.5) #pocet bodu casove site
n=int(8/600*T*m*np.log10(R/d*10)) #pocet bodu casove site
n=1000
tau=T/n #casovy krok
h=R/m #prostorovy krok
sigma=tau/h**2

j_d=round((R-d)/R*m)
j_array = np.linspace(j_d, m, m-j_d+1)

def vypocet_CN(c,D,c_u=0,c_v=c0,theta=1/2):
    '''
    c_u(float): Pocatecni koncentrace uvnitr koule (vyviji se s casem, proto se iniciuje pole c_u_vysl)
    c_v(float): Konstantni koncentrace vne koule
    '''
    start=time.time()
    print("\nBEH CRANK NICOLSONA")
    print("-------------------------------------------")
    print("Koeficient difuze: "+str(D)+" m^2/s")
    print("-------------------------------------------")
    print("Pocet pouzitych bodu prostorove site: "+str(m-j_d+1))
    print("Pocet bodu casove site: "+str(n))

    # uloziste vysledku
    vysledek=np.zeros((n+1,m-j_d+1))
    c_u_vysl=np.zeros(n+1)
    #DO VYSLEDKU SE OKR. PODM. UKLADAJI!!!
    vysledek[0]=c[:] #ulozeni pocatecni podminky
    c_u_vysl[0]=c_u

    pom1 = D*sigma*theta
    pom2 = D*(1. - theta)

    def make_matrix():
        subDiag = np.zeros(m-j_d)
        superDiag = np.zeros(m-j_d)
        mainDiag = np.zeros(m-j_d+1)

        #VNITRNI OKRAJOVA PODMINKA
        # mainDiag[0] = 1 + 6*pom1 + tau*prem_konst*theta
        # superDiag[0] = -6*pom1
        #pred derivaci neni minus
        mainDiag[0]= -1 - h*k_u/D
        superDiag[0]= 1

        subDiag[:-1] = -pom1*(1 - 1/j_array[1:-1])
        # mainDiag[1:-1] = np.ones(m-1)*(1 + 2*pom1)
        mainDiag[1:-1] = np.ones(m-j_d-1)*(1 + 2*pom1 + tau*prem_konst*theta)
        superDiag[1:] = -pom1*(1 + 1/j_array[1:-1])

        #VNEJSI OKRAJOVA PODMINKA
        # mainDiag[-1]=1 #puvodni
        #pred derivaci neni minus
        subDiag[-1]= -1
        mainDiag[-1]= 1 + h*k_v/D

        return scipy.sparse.diags([subDiag, mainDiag, superDiag], [-1, 0, 1], format='csc')

    A=make_matrix()

    def jeden_krok(c_uOld,cOld):
        """ Args:
                c_uOld(float): Concentration inside the sphere for the previous timestep, n.
                cOld(array): The entire solution vector for the previous timestep, n.

            Returns:
                c_uNew(float): concentration inside the sphere at timestep n+1
                cNew(array): solution at timestep n+1
        """
        c_uNew=c_uOld*np.exp(-prem_konst*tau)+(k_u*(cOld[0]-c_uOld)*3)/((R-d)*prem_konst)*(1-np.exp(-prem_konst*tau))

        PS = np.zeros(m-j_d+1)
        #VNITRNI OKRAJOVA PODMINKA
        # PS[0] = (1 - 6*pom2*sigma - tau*prem_konst*(1-theta))*cOld[0] + 6*pom2*sigma*cOld[1]
        #pred derivaci neni minus
        PS[0]= -h*k_u/D*c_uNew

        a = pom2*sigma*(1 - 1./(j_array[1:-1]))*cOld[:-2]
        b = (1 - 2*pom2*sigma - tau*prem_konst*(1-theta))*cOld[1:-1]
        c = pom2*sigma*(1 + 1/(j_array[1:-1]))*cOld[2:]
        PS[1:-1] = a + b + c

        #VNEJSI OKRAJOVA PODMINKA
        # PS[-1]=cOld[-1] #puvodni
        PS[-1]= h*k_v/D*c_v

        cNew = scipy.sparse.linalg.spsolve(A, PS)
        return c_uNew, cNew

    for j in np.arange(1,n+1):
        # print(len(c))
        c_u_vysl[j], c=jeden_krok(c_u_vysl[j-1], c)
        # print(len(c))
        vysledek[j]=c[0:]
        # ulozeni_obrazku(c,j)

    stop=time.time()
    print("-------------------------------------------")
    print("Spotrebovany cas pro nalezeni numerickeho reseni: "+str(stop-start))
    print("-------------------------------------------")
    return c_u_vysl,vysledek

def animace(vysledek,op,interval=1,ulozit=False):
    '''
    OPRAVIT!!! (kvuli zmenam, NEAKTUALNI)
    '''
    j=np.linspace(0,m,num=m+1,dtype=int)
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

def animace_obe_D(vysledek,c_u, c_v,interval=1,ulozit=False):
    # PRVNI MUSI BYT VZDY TEKUTE PROSTREDI (v promenne vysledek)
    a=vysledek[0]
    b=vysledek[1]
    c_u1=c_u[0]
    c_u2=c_u[1]
    # j=np.linspace(0,m,num=m+1,dtype=int)
    # plt.cla()
    # plt.clf()
    # plt.close()
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(R-d-d/10, R+d/10), ylim=(-c0/30,c0+c0/30))
    ax.grid()
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [Bq/m$^3$]')
    # k=np.linspace(0,n,num=n+1,dtype=int)
    ax.axvline(x=R, color='k')
    ax.axvline(x=R-d, color='k')
    line1, = ax.plot(j_array*h, a[0], 'b')
    line2, = ax.plot(j_array*h, b[0], 'g')
    ax.axhline(y=c_v,xmin=0.92,linestyle=':',linewidth=2, color='r')
    line1_u=ax.axhline(y=c_u1[0],xmax=0.08,linestyle=':',linewidth=2, color='b')
    line2_u=ax.axhline(y=c_u2[0],xmax=0.08,linestyle=':',linewidth=2, color='g')
    text_min = ax.text(0.1,0.97, "", ha="left", va="center", transform=ax.transAxes)
    text_sek = ax.text(0.1,0.94, "", ha="left", va="center", transform=ax.transAxes)

    def prepocet(i):
        s=i*tau
        return math.floor(s/60),s

    def animate(i):
        line1.set_ydata(a[i])  # update the data
        line1.set_label('tekuté prostředí')
        line2.set_ydata(b[i])  # update the data
        line2.set_label('pevné prostředí')
        line1_u.set_ydata(c_u1[i])
        line2_u.set_ydata(c_u2[i])
        # legend.remove()
        legend = plt.legend(loc=9)
        minuty,sekundy=prepocet(i)
        text_min.set_text("$t=$ %.0f min" % minuty)
        text_sek.set_text("$t=$ %.0f s" % sekundy)
        return line1, line2, line1_u, line2_u, text_min, text_sek, legend

    def init():
        line1.set_ydata('')
        line2.set_ydata('')
        line1_u.set_ydata('')
        line2_u.set_ydata('')
        text_min.set_text('')
        text_sek.set_text('')
        return line1, line2, line1_u, line2_u, text_min, text_sek

    anim = animation.FuncAnimation(fig, animate, np.arange(0, n+1), init_func=init,
                                interval=interval, blit=True, repeat=False)
    if ulozit:
        anim.save('animace.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def main():
    print("\nSPUSTENI PROGRAMU")
    start=time.time()
    vysledek_celk=0 #bude matice, pokud probehne dany vypocet
    c_u_celk=0 #bude vektor, pokud probehne dany vypocet

    #Uloha a)
    #pp
    pp_a=np.zeros(m-j_d+1)
    #op (puvodni)
    # pp_a[-1]=c0
    c_u_t, vysledek_tekute_a=vypocet_CN(pp_a,D_t)
    c_u_p, vysledek_pevne_a=vypocet_CN(pp_a,D_p)

    #Uloha b)
    #pp
    # pp_b=np.zeros(m-j_d+1)+c0
    #op (puvodni)
    # pp_b[-1]=0
    # vysledek_tekute_b=run(pp_b,D_t)
    # vysledek_pevne_b=run(pp_b,D_p)

    c_u_celk=[c_u_t, c_u_p]
    vysledek_celk=[vysledek_tekute_a, vysledek_pevne_a]

    animace_obe_D(vysledek_celk,c_u_celk,c0,interval=1)

    #CN METODA
    # vysledek_celk=vypocet_CN(pp_a,D_t)
    # animace(vysledek_celk, pp_a[-1])
    stop=time.time()
    print('\nSpotrebovany cas celkove: '+str(stop-start))
    return c_u_celk, vysledek_celk

if __name__ == "__main__":
    c_u_celk, vysledek_celk=main()
