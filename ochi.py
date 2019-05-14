import matplotlib
from matplotlib import rc
from numpy import *
from pylab import *
from scipy.integrate import *
from scipy.interpolate import interp1d
import time
# ffmpeg -framerate 15 -pattern_type glob -i 'mfdump*.png' -b 4096k mfevol.mp4
#Uncomment the following if you want to use LaTeX in figures
rc('font',**{'family':'serif','serif':['Times']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)

tol=1e-3
eqzero=1e-6 # angle indistinguishable from 0 
# parameters of  pulsar losses  (Philippov et al., 2014)
k0=1. #
k1=1. # Sasha has 1.2
k2=1.
m0=1.4 # initial NS mass, Msun units
mmax=2.0 # final NS mass
q1=0.0 # deformation factor with magnetic field 
qsat=1e-8 # saturation value of deformation

rstar = 6.5 # radius (fixed! in GMsun/c**2 units; 10GMsun/c**2 units \simeq 15km )
beta=10. # modulation factor

# switches for physical effects:
parfrey = False
mburial=True # magnetic field burial 
ifmudec=False # magnetic field decay
deathline = False # death line
ifIK = False # turns on IK90 scaling for spin-down (Ghosh-Lamb 78/79 instead)

death_muos = 13.1 # critical value of mu30 * Omega(s)^2 
pafrac=0.0 # in propeller stage, pafrac of matter and angular momentum is actually accreted
pslope=14./11. # mu \propto (Delta M/Delta M0)^-p during burial
deltam0=0.00563863/rstar**3  # mu \propto (Delta M/Delta M0)^-p during burial
mufloor=0. # undecayable part of the field 
xi=0.5 # Alfven radius modifier
kIK=1./3. #k/2\pi from Illarionov-Kompaneets 1990
xlosses=True # if we turn on X-ray pulsar losses (IK90-style; or is it Ghosh-Lamb-style?)
psrlosses=True # pulsar losses 
gwlosses=True # GW losses 
tohm=1e7 # Ohmic dissipation time
thall=1e9 # Hall decay for mu = 10^30

def izero(m, r):
    return 23.0528 * m  # this corresponds to 10^45 for one Msun
# in general, I is in Msun * (G Msun/c**2)**2

def qfun(x, q1, qsat): # x = dm/m_0
    return -q1 * x  /(1.+q1*x/qsat)

def dqfun(x, q1, qsat):
    return -q1 /(1.+q1*x/qsat)**2 

def afun(beta, sina, cosa, sichi, cochi):
    return 1. + beta * ((sina*sichi)**2/2.+(cosa*cochi)**2)

def mufun(mu0, deltam, deltam0, pslope, mufloor):
    # magnetic field burial, according to Melatos& Phinney (2001)
    return mu0*(1.+deltam/deltam0)**(-pslope)+mufloor

def mudec(mu0, t, tohm, thall):
    # Aguilera et al.(2008)
    return mu0*exp(-t/tohm)/(1.+tohm/thall/mu0*(1.-exp(-t/tohm)))

def fastnessfactor(ofast):
    if(ifIK): # Illarionov-Kompaneets case
        return 1./ofast
    else:
        return 1.

def torques(omega, mu, mdot, sichi, cochi, q, m):
    '''
    calculates all the torque components acting on a NS
    output: spin-up moment, X-ray pulsar magnitospheric losses, pulsar losses (x and z), GW losses (x and z)
    '''
    ral = 1577.74 * mu**(4./7.)/mdot**(2./7.)/m**(1./7.) # alfven radius in GMsun/c^2
    jrat= 2.03007e5 * maximum(sqrt(xi * ral), sqrt(rstar)) # either half Alfven radius or the radius of the star
    #    8.036e6*sqrt(xi)*mu**(2./7.)*m**(3./7.)*mdot**(-1./7.) # all re-calculated to domega/dM, M in Msun
    
    jIK=2.11e6*mu**2.*omega**2./mdot/m # X-ray pulsar losses according to Illarionov-Kompaneets, 1990 (or Ghosh & Lamb?)
    jp=10.4*mu**2*omega**3/mdot 
    jpz=jp*(k0+k1*sichi**2) # spin-down law, see Philippov et al. (2014), eq 15
    jpx=jp*k2*sichi*cochi  # aligning torque, see Philippov et al. (2014), eq 16
    jgw0=22.*q**2*rstar**4*omega**5 # hope I still have got this derivation
    jgwz=2.*sichi**2*(3.+cochi**2)*jgw0
    jgwx=4.*cochi*sichi**3*jgw0

    return jrat, jIK, jpx, jpz, jgwx, jgwz

#######################################################################
def aochi(omega0=1., chi0=pi/4., alpha0=pi/2., mu30=1., mdot0=1.0, verbose=True):
    # with omega-aligned losses, j_- \propto \mu^2/R_A^3 or \mu^2/R_C^3
    # mdot in 4piGMsun/ckappa
    
    m=m0 ; md=m0 ;  dm=1e-4 ; dm0=dm 
    mueff=mu30 # effective magnetic moment 
    
    chi=chi0 # magnetic angle

    a=alpha0 # inclination
    omega=omega0 # rotation freq.
    mdot=mdot0 # mass accretion rate

    acur=alpha0
    l=omega0*izero(m0, rstar) # L= I_0 Omega
    
    mstore=m0
    ostore=omega0
    alphastore=alpha0
    chistore=chi0

    mtr=[] ;  mdtr=[] ;  atr=[] ;  otr=[] ;  ctr=[] ;  qtr=[] ; fasttr=[]  ;  pultr=[]  
    jplus=[] ;  jx=[] ;   jjpz=[] ; jjgw=[] ; jjpx=[] ;  pulin=[] ; propin=[] ;  accrin=[]
    ctar=0

    if(verbose):
        otrace=open('aotrace.dat', 'w')
        otrace.write('# md(Msun) -- m(Msun) -- Omega(s) -- a -- chi\n')
        start = time.time()
    
    while(md<mmax):
        if (mburial):
            if(ifmudec):
                mueff=mudec(mu30, (md-m0)/mdot, tohm/4e8, thall/mu30/4e8) # time scales in c kappa / 4 / pi / G units
            else:
                mueff=mufun(mu30, m-m0, deltam0, pslope, mufloor) 
        ofast=0.31*mueff**(6./7.)/m**(5./7.)/mdot**(3./7.)*omega # fastness parameter
        opul=127.6/xi*(mdot*sqrt(m)/mueff**2.)**(2./7.) # period when the light cylinder coincides with the radius of the magnetosphere
        if(((ofast>=1.)&xlosses)|(omega>opul)):
            afrac=pafrac # propeller state
        else:
            afrac=1. # accreting
        if (omega<opul) & xlosses: 
            bfrac=1.
        else:
            bfrac=0.  # ejector
        pfrac=1.
        if(parfrey):
            pfrac = maximum(1., (opul/omega)**2) # Parfrey's modifier
        if(deathline & ((omega**2 * mueff) < death_muos)):
            pfrac = 0. # death line
        if(abs(chi)>eqzero): # accurate treatment of the cases chi\to 0, alpha \to 0 (do we need it?)
            sichi=sin(chi)
            cochi=cos(chi)
        if(abs(a)>eqzero):
            sina=sin(a)
            cosa=cos(a)
        b=sina*sichi*cosa*cochi*beta/afun(beta, sina, cosa, sichi, cochi)
        if (mburial):
            if(ifmudec):
                mueff=mudec(mu30, (md-m0)/mdot, tohm/4e8, thall/mu30/4e8) # time scales in c kappa / 4 / pi / G units
            else:
                mueff=mufun(mu30, m-m0, deltam0, pslope, mufloor) 
        imid=izero(m, rstar) # moment of inertia
        q=qfun(1.-m0/m, q1, qsat) # deformation as a function of m
        dq=dqfun(1.-m0/m, q1, qsat)/m # dq/dt

        jrat, jIK, jpx, jpz, jgwx, jgwz  = torques(omega, mueff, mdot, sichi, cochi, q, m)
        if(ofast>0.):jIK *= fastnessfactor(ofast)
        jrat *= afrac ; jIK *= bfrac ; jpx *= pfrac ; jpz *= pfrac
        # we want the time step to adjust to the evolutionary time scales
        dm = l/sqrt((jpx**2+jpz**2)*pfrac+(jrat-jIK)**2*afrac+jIK**2*bfrac)*dm0

        # we need to estimate the working omega:
        domega=((1.+q*sichi**2)*(jrat*cosa-jIK-jpz-jgwz)-dq*l*cochi**2 - b * q *jrat * sina - q * jpx * sichi * cochi)/(q+1.)
        dchi=((q*(jrat*cosa-jIK-jpz-jgwz)+dq)*sichi*cochi -  b * jrat * sina *(1.+q*cochi**2)-(1.+q*cochi**2)*(jpx+jgwx))/(q+1.)/l
        da=((-sina*(1.+q*(1.+cochi**2)/2.)*jrat + b *q *jrat * cosa * sichi * cochi))/l / (q+1.)
        l1=l+domega*dm/2.
        chi1 = chi + dchi*dm/2.
        sichi=sin(chi1)
        cochi=cos(chi1)
        a1 = a + da*dm/2.
        sina=sin(a1)
        cosa=cos(a1)
        b=sina*sichi*cosa*cochi*beta/afun(beta, sina, cosa, sichi, cochi)
        omega1=l1/imid # about the mean omega (moment of inertial included, meaning we actually evolve l not omega)
        #        jlim=1.48e5*sqrt(m*rstar)
        ofast=0.31*mueff**(6./7.)/m**(5./7.)/mdot**(3./7.)*omega1 # fastness parameter
        opul=127.6/xi*(mdot*sqrt(m)/mueff**2.)**(2./7.) # period when the light cylinder coincides with the radius of the magnetosphere
        if(((ofast>=1.)&xlosses)|(omega1>opul)):
            afrac=pafrac # propeller state
        else:
            afrac=1. # accreting
        if (omega1<opul) & xlosses: 
            bfrac=1.
        else:
            bfrac=0.  # ejector
        pfrac=1.
        if(parfrey):
            pfrac = maximum(1., (opul/omega)**2) # Parfrey's modifier
        if(deathline & ((omega1**2 * mueff) < death_muos)):
            pfrac = 0. # death line
        jrat, jIK, jpx, jpz, jgwx, jgwz  = torques(omega1, mueff, mdot, sichi, cochi, q, m)
        jrat *= afrac ; jIK *= bfrac ; jpx *= pfrac ; jpz *= pfrac
        domega=((1.+q*sichi**2)*(jrat*cosa-jIK-jpz-jgwz)-dq*l1*cochi**2- b * q *jrat * sina- q * jpx * sichi * cochi)/(q+1.)
        dchi=((q*(jrat*cosa-jIK-jpz-jgwz)+dq)*sichi*cochi -  b * jrat * sina *(1.+q*cochi**2)-(1.+q*cochi**2)*(jpx+jgwx))/(q+1.)/l1
        da=((-sina*(1.+q*(1.+cochi**2)/2.)*jrat + b *q *jrat * cosa * sichi * cochi))/l1 / (q+1.)
        chi += dchi*dm
        sichi=sin(chi)
        cochi=cos(chi)
        a+=da*dm
        sina=sin(a)
        cosa=cos(a)

        l+=domega*dm
        omega=l/izero(m,rstar)
        m+=dm*afrac
        md+=dm
        
        if((domega*dm/l)>0.1):
            print("warning! dO = "+str(domega*dm))
            print("M = "+str(m)+";  Omega = "+str(omega)+"; alpha = "+str(a)+";  chi = "+str(chi))
            dm0/=2.
            print("dm = "+str(dm))

        if(((m>(mstore*1.01))|(abs(omega-ostore)>(0.1*ostore))|(abs(a-alphastore)>(0.1*abs(a)))|(abs(chi-chistore)>(0.1*abs(chi))))&verbose): # if any of the parameters change by 10%, we make an output
            print("M = "+str(m)+";  Omega = "+str(omega)+"; alpha = "+str(a)+";  chi = "+str(chi))
            print("o/opul = "+str(omega/opul)+" = "+str(omega)+" / "+str(opul))
            mtr.append(m)
            mdtr.append(md)
            otr.append(omega)
            atr.append(a)
            ctr.append(chi)
            qtr.append(q)
            otrace.write(str(md)+' '+str(m)+' '+str(omega)+' '+str(a)+' '+str(chi)+'\n')
            otrace.flush()
            fasttr.append(ofast)
            pultr.append(omega/opul)
            jplus.append(jrat*afrac)
            jx.append(-jIK*bfrac)
            jjpz.append(-jpz*pfrac)
            jjpx.append(jpx*pfrac)
            jjgw.append(jgwz)
            ostore=omega
            mstore=m
            alphastore=a
            chistore=chi
            if(omega>opul):
                pulin.append(ctar)
                print("pulsar regime")
            else:
                if((ofast>=1.)&xlosses):
                    propin.append(ctar)
                    print("propeller regime")
                else:
                    accrin.append(ctar)
                    print("accretor regime")
            ctar=ctar+1
    if(verbose):
        otrace.close()
        mdar=asarray(mdtr, dtype='double')
        mar=asarray(mtr, dtype='double')
        oar=asarray(otr, dtype='double')
        aar=asarray(atr, dtype='double')
        chir=asarray(ctr, dtype='double')
        qr=asarray(qtr, dtype='double')
        jplus=asarray(jplus, dtype='double')
        jx=asarray(jx, dtype='double')
        jjpx=asarray(jjpx, dtype='double')
        jjpz=asarray(jjpz, dtype='double')
        jjgw=asarray(jjgw, dtype='double')
        pulin=asarray(pulin)
        propin=asarray(propin)
        accrin=asarray(accrin)
        end = time.time()
        print("calculation took "+str(end-start)+"s = "+str((end-start)/60.)+"min")
        #        o0=369.*sqrt(xi)*mar**(6./7.)*mu30**(2./7.)/mdot**(1./7.)/izero(mar, rstar)
        #        oo=o0*((mar-m0)/deltam0)**(1.-2./7.*pslope)
        if(mburial):
            oeq=1.9*mdot**(3./7.)*mar**(5./7.)/mufun(mu30, mar-m0, deltam0, pslope, mufloor)**(6./7.)
        else:
            oeq=1.9*mdot**(3./7.)*mar**(5./7.)/mu30**(6./7.)
            #mu30**(6./7.)*(1.+(mar-m0)/deltam0)**(pslope*6./7.)

        kest=2./(10./7.*pslope+1.)/(1.-2.*pslope/7.)**2*xi*mu30**(18./7.)*m0**(6./7.)*mdot**(-9./7.)/izero(m0,rstar)**3*(deltam0/4.65e-5)**3
        print("kest="+str(kest))

        plt.clf()
        fig=figure()
        subplot(3,1,1)
        plot(mdar-m0, otr,color='k',linewidth=2.)
        plot(mdar-m0, oar*sin(aar),color='r')
        plot(mdar-m0, oar*cos(aar),color='b')
        plot(mdar-m0, oeq, color='r', linestyle='dotted')
        yscale('log')
        xscale('log')
        ylabel('$\Omega$, s$^{-1}$', fontsize=20)
        tick_params(labelsize='x-large')
        ylim([oar.min()/2., oar.max()*2.])
        subplot(3,1,2)
        plot(mdar-m0, chir,color='k')
        xscale('log')
        tick_params(labelsize='x-large')
        ylabel('$\chi$', fontsize=20)
        subplot(3,1,3)
        plot(mdar-m0, aar,color='k')
        ylabel(r'$\alpha$', fontsize=20)
        xscale('log')
        xlabel(r'$\Delta M$, M$_\odot$', fontsize=20)
        fig.set_size_inches(4, 6)
        tick_params(labelsize='x-large')
        tight_layout()
        savefig('aotrace.eps')
        savefig('aotrace.png')
        close()
        plt.clf()
        fig=figure()
        plot(mdar-m0, jplus+jx+jjpz,label='total',color='k')
        plot(mdar-m0, -(jplus+jx+jjpz),label='total',color='k',linestyle='dotted')
        plot(mdar-m0, jplus,label='$j_+$',color='g')
        plot(mdar-m0, -jplus,label='$j_+$',color='g',linestyle='dotted')
        #    plot(mdar[accrin]-m0, jplus[accrin],'o', color='g')
        plot(mdar-m0, jx, label='$j_X$', color='r')
        plot(mdar-m0, -jx, label='$j_X$', color='r',linestyle='dotted')
        #    plot(mdar[accrin]-m0, jx[accrin], 'o', color='k')
        #    plot(mdar[propin]-m0, jx[propin], 'x', color='r')
        plot(mdar-m0, jjpz, label='$K_z$', color='m',linewidth=2)
        plot(mdar-m0, -jjpz, label='$K_z$', color='m',linestyle='dotted',linewidth=2)
        plot(mdar-m0, jjgw, label='$G_z$', color='c',linestyle='dashed',linewidth=2)
        #    plot(mdar[pulin]-m0, jjpz[pulin], '*', color='b')
        #        plot(mdar-m0, jjpx, label='$K_x$', color='m')
        #        plot(mdar-m0, -jjpx, label='$K_x$', color='m',linestyle='dotted')
        #    plot(mdar[pulin]-m0, jjpx[pulin], '*', color='m')
        #        ylim([(fabs(jplus)*0.+fabs(jjgw)+fabs(jx)*0.+fabs(jjpz)).min()/5.,(fabs(jplus)+fabs(jjgw)+fabs(jx)+fabs(jjpz)).max()])
        #    legend()
        xscale('log')
        yscale('log')
        xlabel(r'$\Delta M, $M$_\odot$',fontsize=20)
        ylabel(r'$N/I\Omega$',fontsize=20)
        tick_params(labelsize='x-large')
        fig.set_size_inches(4, 4)
        tight_layout()
        savefig('aotorqs.eps')
        savefig('aotorqs.png')
        close()    
    return m, omega, chi, a

##################################################################
################################################################################################
# maps 
def aomap_mm():
    '''
    calculates a model grid with identical alpha0, chi0 but variable mdot and mu
    '''
    recalc=False
    nmag=10
    nmdot=10

    a0=pi/3.
    chi0=pi/4.
    omega0=1.

    mag0=1e-2
    mag1=100.
    mu=(mag1/mag0)**(arange(nmag)/double(nmag-1))*mag0
    
    mdot0=1e-2
    mdot1=100.
    mdot=(mdot1/mdot0)**(arange(nmdot)/double(nmdot-1))*mdot0
    
    chifin=zeros([nmag, nmdot], dtype=double)
    ofin=zeros([nmag, nmdot], dtype=double)

    if(recalc):
        aomap=open('aomap.dat', 'w')
    else:
        aomap=open('aomap.dat', 'r')

    for j in arange(nmag):
        for k in arange(nmdot):
            if(recalc):
                print("aochi( alpha0="+str(a0)+", chi0="+str(chi0)+", mdot0="+str(mdot[k])+", mu30="+str(mu[j])+", omega0="+str(omega0)+")")
                mtmp, otmp, ctmp, atmp = aochi(alpha0=a0, chi0=chi0, mdot0=mdot[k], mu30=mu[j], verbose=False,omega0=omega0)
                chifin[j,k]=ctmp
                ofin[j,k]=otmp
                aomap.write(str(mu[j])+' '+str(mdot[k])+' '+str(otmp)+' '+str(ctmp)+'\n')
            else:
                s=str.split(str.strip(aomap.readline()))
                ctmp=double(s[3])
                otmp=double(s[2])
                chifin[j,k]=ctmp
                ofin[j,k]=otmp
            print("mdot = "+str(mdot[k]))
            print("mu30 = "+str(mu[j]))
            print("ctmp = "+str(ctmp))
            print("otmp = "+str(otmp))

    aomap.close()
    mdot2,mu2=meshgrid(mdot,mu)

    plt.clf()
    contourf(mu2, mdot2, chifin,20)
    colorbar()
    contour(mu2, mdot2, mu2-1.7e-5*mdot2**(9./4.)*1.4**0.25,20, levels=[0.], linestyles='dotted')
    tick_params(labelsize='x-large')
    xscale('log')
    yscale('log')
    xlabel(r'$\mu_{30}$',fontsize=16)
    ylabel(r'$\dot{m}$',fontsize=16)
    savefig('aocfin.eps')
    plt.clf()
    contourf(mu2, mdot2, log(ofin),20)
    colorbar()
    contour(mu2, mdot2, mu2-1.7e-5*mdot2**(9./4.)*1.4**0.25,20, levels=[0.], linestyles='dotted')
    xscale('log')
    yscale('log')
    tick_params(labelsize='x-large')
    xlabel(r'$\mu_{30}$',fontsize=16)
    ylabel(r'$\dot{m}$',fontsize=16)
    savefig('aologomega.eps')
    plt.clf()
    plot(ofin, chifin, '.')
    xscale('log')
    #    yscale('log')
    tick_params(labelsize='x-large')
    xlabel(r'$\Omega_{\rm fin}$',fontsize=16)
    ylabel(r'$\chi_{\rm fin}$',fontsize=16)
    savefig('aoochi.eps')

def aomap_achi():

    recalc=True
    nchi=20
    na=21

    mu0=1.
    mdot0=0.1
    chi0=pi/4.
    omega0=1.

    chi1=1.e-2
    chi2=pi/2.-1.e-2
    chi=(chi2/chi1)**(arange(nchi)/double(nchi-1))*chi1
    
    alpha1=1.e-2
    alpha2=pi-1.e-2
    alpha=(alpha2/alpha1)**(arange(na)/double(na-1))*alpha1
    
    chifin=zeros([na, nchi], dtype=double)
    ofin=zeros([na, nchi], dtype=double)

    c2,a2=meshgrid(chi,alpha)

    if(recalc):
        aomap=open('caomap.dat', 'w')
    else:
        aomap=open('caomap.dat', 'r')

    for j in arange(na):
        for k in arange(nchi):
            if(recalc):
                print("aochi( alpha0="+str(alpha[j])+", chi0="+str(chi[k])+", mdot0="+str(mdot0)+", mu30="+str(mu0)+", omega0="+str(omega0)+")")
                mtmp, otmp, ctmp, atmp = aochi(alpha0=alpha[j], chi0=chi[k], mdot0=mdot0, mu30=mu0, verbose=False,omega0=omega0)
                chifin[j,k]=ctmp
                ofin[j,k]=otmp
                aomap.write(str(alpha[j])+' '+str(chi[k])+' '+str(otmp)+' '+str(ctmp)+'\n')
            else:
                s=str.split(str.strip(aomap.readline()))
                c0tmp=double(s[1])
                a0tmp=double(s[0])
                ctmp=double(s[3])
                otmp=double(s[2])
                c2[j,k]=c0tmp
                a2[j,k]=a0tmp
                chifin[j,k]=ctmp
                ofin[j,k]=otmp
            print("alpha = "+str(alpha[j]))
            print("chi = "+str(chi[k]))
#            print("mdot = "+str(mdot0))
#            print("mu30 = "+str(mu0))
            print("ctmp = "+str(ctmp))
            print("otmp = "+str(otmp))

    aomap.close()

    plt.clf()
    contourf(c2*180./pi, a2*180./pi, (chifin-c2)*180./pi,20)
    colorbar()
#    contour(c2, a2, mu2-1.7e-5*mdot2**(9./4.)*1.4**0.25,20, levels=[0.], linestyles='dotted')
    tick_params(labelsize='x-large')
#    xscale('log')
#    yscale('log')
    xlabel(r'$\chi_0$, deg',fontsize=16)
    ylabel(r'$\alpha_0$, deg',fontsize=16)
    savefig('dcaocfin.eps')
    plt.clf()
    contourf(c2*180./pi, a2*180./pi, chifin*180./pi,20)
    colorbar()
#    contour(c2, a2, mu2-1.7e-5*mdot2**(9./4.)*1.4**0.25,20, levels=[0.], linestyles='dotted')
    tick_params(labelsize='x-large')
#    xscale('log')
#    yscale('log')
    xlabel(r'$\chi_0$, deg',fontsize=16)
    ylabel(r'$\alpha_0$, deg',fontsize=16)
    savefig('caocfin.eps')
    plt.clf()
    contourf(c2, a2, log(ofin),20)
    colorbar()
#    contour(c2, a2, mu2-1.7e-5*mdot2**(9./4.)*1.4**0.25,20, levels=[0.], linestyles='dotted')
#    xscale('log')
#    yscale('log')
    tick_params(labelsize='x-large')
    xlabel(r'$\chi_{0}$',fontsize=16)
    ylabel(r'$\alpha_0$',fontsize=16)
    savefig('caologomega.eps')
    plt.clf()
    plot(ofin, chifin, '.')
    xscale('log')
    #    yscale('log')
    tick_params(labelsize='x-large')
    xlabel(r'$\Omega_{\rm fin}$',fontsize=16)
    ylabel(r'$\chi_{\rm fin}$',fontsize=16)
    savefig('caoochi.eps')

def compare(infile1 = 'aotrace.dat', infile2 = 'aotrace1.dat'):
    data1 = loadtxt(infile1)  ;  data2 = loadtxt(infile2)
    md1 = data1[:,0] ; m1 = data1[:,1] ; omega1 = data1[:,2] ; a1 = data1[:,3] ; chi1 = data1[:,4]
    md2 = data2[:,0] ; m2 = data2[:,1] ; omega2 = data2[:,2] ; a2 = data2[:,3] ; chi2 = data2[:,4]

    plt.clf()
    plot(a1, chi1, 'k-')
    plot(a2, chi2, 'r-')
    ylabel(r'$\chi$',fontsize=16)
    xlabel(r'$\alpha$',fontsize=16)
    savefig('compare_achi.png')
    plt.clf()
    plot(m1, omega1, 'k-')
    plot(m2, omega2, 'r-')
    xscale('log') ;   yscale('log')
    xlabel(r'$M$',fontsize=16)
    ylabel(r'$\Omega, \, {\rm s^{-1}}$',fontsize=16)
    savefig('compare_momega.png')

    # interpolation and difference:
    mint = interp1d(md2, m2, bounds_error = False)
    aint = interp1d(md2, a2, bounds_error = False)
    chint = interp1d(md2, chi2, bounds_error = False)
    oint = interp1d(md2, omega2, bounds_error = False)
    plt.clf()
    plot(aint(md1)-a1, chint(md1)-chi1, 'k-')
    ylabel(r'$\Delta\chi$',fontsize=16)
    xlabel(r'$\Delta\alpha$',fontsize=16)
    savefig('compare_dachi.png')
    plt.clf()
    plot(md1, oint(md1)/omega1-1., 'k-')
    ylabel(r'$M-M_0$',fontsize=16)
    xlabel(r'$\Delta \Omega/\Omega$',fontsize=16)
    savefig('compare_dome.png')
    
aochi(omega0=2.*pi/0.05, chi0=pi*(0.5-1./36.), alpha0=0.75*pi, mu30=1e-2, mdot0=1.0, verbose=True)
