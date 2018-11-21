import numpy as np
# from scipy.sparse import diags
# import scipy
# import scipy.linalg
# import scipy.sparse
# import scipy.sparse.linalg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from matplotlib import cm
import matplotlib.animation as animation
import time
import math
# from astropy.units import dT

# T=600*3 #cas v sekundach
T=600
R=0.1 #polomer v metrech
c0=300 #pocatecni koncentrace v Bq/m^3

# n=5*10**4
# m=200
n=1000
m=50
tau=T/n
h=R/m

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

def make_matrix_CN(D):
    sigma=tau/h**2
    # A=np.zeros((m+1,m+1))
    # A[0,0]=1-6*D*sigma
    # A[0,1]=6*D*sigma
    # for j in np.arange(1,m):
        # A[j,j-1]=D*sigma*(1-1/j)
        # A[j,j]=1-2*D*sigma
        # A[j,j+1]=D*sigma*(1+1/j)
    # A[m,m]=1
    # return A

def vypocet_CN(c,D):
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

def vykresleni(vysledek):
    plt.cla()
    plt.clf()
    plt.close()
    k, j=np.linspace(0,n,num=n+1,dtype=int), np.linspace(0,m-1,num=m,dtype=int)
    # print(len(vysledek[0,:]))
    # print(len(j))
    # print(j)
    t, r=k*tau, j*h
    t, r=np.meshgrid(t, r)
    fig=plt.figure()
    ax=fig.gca(projection='3d')

    # surf=ax.plot_surface(t,r,np.transpose(vysledek), cmap=cm.coolwarm)
    plt.ion()
    surf=ax.plot_surface(t,r,np.transpose(vysledek))
    plt.pause(0.0001)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('r [m]')
    ax.set_zlabel('c [Bq/m^3]')

    # fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.ioff()

def animace(vysledek,op,interval=10**(-n),ulozit=False):
    j=np.linspace(0,m-1,num=m,dtype=int)
    # plt.cla()
    # plt.clf()
    # plt.close()
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(0, 0.11), ylim=(-5,c0+50))
    ax.grid()
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [Bq/m$^3$]')
    # k=np.linspace(0,n,num=n+1,dtype=int)
    line, = ax.plot(j*h, vysledek[0],)
    ax.axvline(x=0.1, color='k')
    ax.axhline(y=op,xmin=0.1/0.11,linestyle=':',linewidth=2)
    text_min = ax.text(0.02,0.97, "", ha="left", va="center", transform=ax.transAxes)
    text_sek = ax.text(0.02,0.94, "", ha="left", va="center", transform=ax.transAxes)

    def prepocet(i):
        s=np.array(i)*tau
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

def animace_obe_D(vysledek,op,interval=10**(-n),ulozit=False):
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
    ax = plt.axes(xlim=(0, 0.11), ylim=(-5,c0+50))
    ax.grid()
    plt.xlabel('$r$ [m]')
    plt.ylabel('$c$ [Bq/m$^3$]')
    # k=np.linspace(0,n,num=n+1,dtype=int)
    ax.axvline(x=0.1, color='k')
    line1, = ax.plot(j*h, a[0])
    line2, = ax.plot(j*h, b[0])
    ax.axhline(y=op,xmin=0.1/0.11,linestyle=':',linewidth=2)
    text_min = ax.text(0.02,0.97, "", ha="left", va="center", transform=ax.transAxes)
    text_sek = ax.text(0.02,0.94, "", ha="left", va="center", transform=ax.transAxes)

    def prepocet(i):
        s=np.array(i)*tau
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

def run(pp,D,interval_animace=10**(-n),ulozit_animace=False):
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
        return 0

def main():
    vysledek_celk=0
    D_t=2*10**(-6) #tekute prostredi v m^2/s
    D_p=3*10**(-7) #pevne prostredi v m^2/s

    #Uloha a)
    #pp
    pp_a=np.zeros(m+1)
    #op
    pp_a[-1]=c0
    vysledek_tekute_a=run(pp_a,D_t)
    vysledek_pevne_a=run(pp_a,D_p)

    #Uloha b)
    #pp
    # pp_b=np.zeros(m+1)+c0
    #op
    # pp_b[-1]=0
    # vysledek_tekute_b=run(pp_b,D_t)
    # vysledek_pevne_b=run(pp_b,D_p)

    # vysledek=run(pp,D,ulozit_animace=True)
    vysledek_celk=[vysledek_tekute_a, vysledek_pevne_a]
    animace_obe_D(vysledek_celk,pp_a[-1])
    return vysledek_celk

#TO DO: make_matrix_CN, vypocet_CN

if __name__ == "__main__":
    vysledek_celk=main()
