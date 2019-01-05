def main():
    '''
    animace_obe_D NEPOUZIVAT!!!!!!!
    '''
    print("\nSPUSTENI PROGRAMU")
    start=time.time()

    print("\nRelaxacni doby:")
    t_rel_t=relaxacni_doba(D_t)
    t_rel_p=relaxacni_doba(D_p)
    print("t_rel_tekute="+str(t_rel_t))
    print("t_rel_pevne="+str(t_rel_p))

    T_t=round(30*t_rel_t) #cas v sekundach
    T_p=round(35*t_rel_p)

    vysledek_celk1=0 #bude matice, pokud probehne dany vypocet
    c_u_celk1=0 #bude vektor, pokud probehne dany vypocet

    #Uloha a)
    #pp
    pp_a1=np.zeros(m-j_d+1)
    #op (puvodni)
    # pp_a[-1]=c0
    k_t1,tau_t1,c_u_t1, vysledek_t1=vypocet_CN(T_t,pp_a1,D_t)
    k_p1,tau_p1,c_u_p1, vysledek_p1=vypocet_CN(T_p,pp_a1,D_p)
    # k_p1,c_u_p1, vysledek_p1,tau_p1=vypocet_CN(T_t,pp_a1,D_p) #pro animace_obe_D

    #Uloha b)
    #pp
    # pp_b=np.zeros(m-j_d+1)+c0
    #op (puvodni)
    # pp_b[-1]=0
    # vysledek_tekute_b=run(pp_b,D_t)
    # vysledek_pevne_b=run(pp_b,D_p)

    c_u_celk=[c_u_t1, c_u_p1]
    vysledek_celk=[vysledek_t1, vysledek_p1]

    #Prvni cyklus=NAPLNOVANI
    animace(vysledek_t1,c_u_t1,c0,tau_t1,k_t1,'tekute', interval=1)
    animace(vysledek_p1,c_u_p1,c0,tau_p1,k_p1,'pevne', interval=1)

    #SOUHRNA ANIMACE
    # animace_obe_D(vysledek_celk,c_u_celk,c0,tau_t1, k_t1,interval=1)

    #CN METODA
    # vysledek_celk=vypocet_CN(pp_a,D_t)
    # animace(vysledek_celk, pp_a[-1])
    k=[k_t1, k_p1]
    t=[k_t1*tau_t1,k_p1*tau_p1]
    stop=time.time()
    print('\nSpotrebovany cas celkove: '+str(stop-start))
    return k,t,c_u_celk, vysledek_celk

