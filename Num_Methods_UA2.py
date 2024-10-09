import autograd.numpy as np
import numpy as npp
import numpy as npp
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec
from autograd import jacobian
import scipy.optimize
import matplotlib.pyplot as plt
import time
import statistics


# Micro-Macro methods for Highly oscillatory ODE's

# Maths parameters

dyn_syst = "VDP"            # Dynamical system: "Logistic", "VDP" or "Henon-Heiles"
num_meth = "Integral_Euler_Modified_AppInt"        # Numerical method: "Forward Euler" or "Integral_order1" or "Integral_RK2" or "Integral_MidPoint" or "Integral_Euler_AppInt" or "Integral_Euler_Modified_AppInt" or "Integral_Symp_Euler_AppInt" or "Integral_MidPoint_AppInt (Integral Euler/MidPoint + Approximation of integrals involved in the scheme)
step_h = [1e-3 , 0.1]           # Time step interval for convergence curves
step_eps = [1e-3,1]           # High oscillation parameter interval for the convergence curves
n_Gauss = 10                   # Number of points in Gauss quadrature

# Curves parameters

n_HH = 7 # Number of time steps used for convergence curves
n_EPS = 7 # Number of high oscillation parameters used for convergence curves

print(" ")
print(100*"-")
print(" Uniformly Accurate methods for highly oscillatory ODE's - Micro-Macro method")
print(100*"-")
print(" ")
print("Maths parameters:")
print(" - Dynamical system:" , dyn_syst)
print(" - Numerical method:" , num_meth)
print(" - Time step inteval for convergence curves:" , step_h)
print(" - High oscillation parameter interval for the convergence curves:" , step_eps)
print(" - Number of points in Gauss quadrature:", n_Gauss)
print("Curves parameters")
print(" - Number of time steps used for convergence curves:", n_HH)
print(" - Number of high oscillation parameters used for convergence curves:", n_EPS)


if dyn_syst == "Logistic":
    d = 1
    Y0 = np.array([1.5])
if dyn_syst == "VDP":
    d = 2
    Y0 = np.array([1,1])
if dyn_syst == "Henon-Heiles":
    d = 4
    #Y0 = np.array([0,0,np.sqrt(eps_simul/3)*np.sin(5*np.pi/4),np.sqrt(1/3)*np.cos(5*np.pi/4)])
    Y0 = np.array([0,0,0.7,0.7])


class ODE:
    def f(tau,y):
        """Vector field associated to the dynamics of the ODE.
        Inputs:
        - tau: Float - Time variable
        - y: Array of shape (d,) - Space variable"""
        if dyn_syst == "Logistic":
            y = y.reshape(d,)
            z = y*(1-y) + np.array([np.sin(tau)])
            return z
        if dyn_syst == "VDP":
            y = y.reshape(d,)
            y1 , y2 = y[0] , y[1]
            #z = 0.0*y
            #z = ( 1/4 - (y1*np.cos(tau) + y2*np.sin(tau))**2)*(-y1*np.sin(tau)+y2*np.cos(tau))*np.array([-np.sin(tau),np.cos(tau)])
            #z[0] = -np.sin(tau)*( 1/4 - (y1*np.cos(tau) + y2*np.sin(tau))**2)*(-y1*np.sin(tau)+y2*np.cos(tau))
            #z[1] = np.cos(tau)*( 1/4 - (y1*np.cos(tau) + y2*np.sin(tau))**2)*(-y1*np.sin(tau)+y2*np.cos(tau))
            #z = 0*y
            #z[0] = - np.sin(tau) * (1 / 4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            #z[1] = np.cos(tau) * (1 / 4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            #z[0] = - np.sin(tau) * (1 / 4 - (np.cos(tau) * y1 + np.sin(tau) * y2) ** 2) * (-np.sin(tau) * y1 + np.cos(tau) * y2)
            #z[1] = np.cos(tau) * (1 / 4 - (np.cos(tau) * y1 + np.sin(tau) * y2) ** 2) * (-np.sin(tau) * y1 + np.cos(tau) * y2)
            #c = np.cos(tau)
            #s = np.sin(tau)
            #z = (1 / 4 - (c * y1 + s * y2) ** 2) * (-s * y1 + c * y2) * np.array([-s, c])
            z = (1/8)*np.array([1-y1**2-y2**2 , 1-y1**2-y2**2])*y + (1/8)*np.array([-y1*(1-4*y2**2) , y2*(1-4*y1**2)])*np.cos(2*tau)  +  (1/8)*np.array([-y2*(1+2*y1**2-2*y2**2) , -y1*(1+2*y2**2-2*y1**2)])*np.sin(2*tau) + (1/8)*np.array([y1*(y1**2-3*y2**2) , y2*(y2**2-3*y1**2)])*np.cos(4*tau)  + (1/8)*np.array([y2*(3*y1**2-y2**2) , y1*(y1**2-3*y2**2)])*np.sin(4*tau)
            return z
        if dyn_syst == "Henon-Heiles":
            y = y.reshape(d,)
            y1 , y2 , y3 , y4 = y[0] , y[1] , y[2] , y[3]
            z = y
            z = np.array([y2*y3 , y4 , -y1*y2 , -(1/2)*(y1**2+y3**2-3*y2**2+2*y2)]) + np.array([-y2*y3 , 0 , -y1*y2 , (1/2)*(y3**2-y1**2)])*np.cos(2*tau) + np.array([y1*y2 , 0 , -y2*y3 ,-y1*y3])*np.sin(2*tau)
            return z

    def f1(tau,sigma,y):
        """Vector field used in the second integral for Integral Euler Modified.
        Inputs:
         - tau: Float - First time variable
         - sigma: Float - Second time variable
         - y: Array of shape (d,) - Space variable"""
        eta = 1e-5 # Small parameter for derivative approximation
        if dyn_syst == "Logistic":
            ff = (1 - 2*y)*ODE.f(sigma,y)
        else:
            ff = (ODE.f(tau,y+eta*ODE.f(sigma,y)) - ODE.f(tau,y-eta*ODE.f(sigma,y)))/(2*eta)
        return ff

    def H(y,eps):
        """Hamiltonian function of the system (only if the system is Hamiltonian);
        Inputs:
        - y: Array of shape (d,n) - Space variable
        - eps: Float - High oscillation parameter"""
        if dyn_syst == "Henon-Heiles":
            q1 , q2 , p1 , p2 = y[0,:] , y[1,:] , y[2,:] , y[3,:]

            return p1**2/(2*eps) + p2**2/2 + q1**2/(2*eps) + q2**2/2 + q1**2*q2 -  (q2**3)/2

    def f_avg(y):
        """Average vector field.
        Inputs:
        - y: Array of shape (d,) - Space variable"""
        if dyn_syst == "Logistic":
            y = y.reshape(d,)
            z = y*(1-y)
            return z
        if dyn_syst == "VDP":
            y = y.reshape(d, )
            y1 , y2 = y[0] , y[1]
            #z = np.zeros_like(y)
            z=y
            z = (1/8)*(1-y1**2-y2**2)*np.array([y1,y2])
            return z
        if dyn_syst == "Henon-Heiles":
            y = y.reshape(d, )
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            z = y
            z = np.array([y2 * y3, y4, -y1 * y2, -(1 / 2) * (y1 ** 2 + y3 ** 2 - 3 * y2 ** 2 + 2 * y2)])
            return z

    def f_1_avg(y):
        """Average vector field used in the second integral.
        Inputs:
        - y: Array of shape (d,) - Space variable"""
        eta = 1e-5 # Small parameter for derivative approximation
        if dyn_syst == "Logistic":
            ff = (1 - 2*y)*ODE.f_avg(y)
        else:
            ff = (ODE.f_avg(y + eta * ODE.f_avg(y)) - ODE.f_avg(y - eta * ODE.f_avg(y)))/(2*eta)
        return ff

    def phi1(tau,y,eps):
        """Variable change associated to the dynamics of the ODE.
        Inputs:
        - tau: Float - Time variable
        - y: Array of shape (d,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = y.reshape(d, )
        if dyn_syst == "Logistic":
            z , F = y , y
            F = np.array([1-np.cos(tau)])
            z = z + eps*F
        if dyn_syst == "Henon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            z , F = y , y
            F = (1/2)*np.array([y1*y2 , 0 , -y2*y3 , -y1*y3]) + (1/2)*np.array([-y1*y2 , 0 , y2*y3 , y1*y3])*np.cos(2*tau) + (1/2)*np.array([-y2*y3 , 0 , -y1*y2 ,(1/2)*(y3**2-y1**2)])*np.sin(2*tau)
            z = z + eps*F
        if dyn_syst == "VDP":
            y1 , y2 = y[0] , y[1]
            z, F = y, y
            #F = (np.array([-(1/32)*y1**2*y2+(3/32)*y2**3-(1/16)*y2 , (5/32)*y1**3-(7/32)*y1*y2**2-(1/16)*y1])    +     (1 / 16) * np.array([y2 * (1 + 2 * y1 ** 2 - 2 * y2 ** 2), y1 * (1 + 2 * y2 ** 2 - 2 * y1 ** 2)]) * np.cos(2 * tau) + (1 / 16) * np.array([-y1 * (1 - 4 * y2 ** 2), y2 * (1 - 4 * y1 ** 2)]) * np.sin(2 * tau) + (1 / 32) * np.array([-y2 * (3 * y1 ** 2 - y2 ** 2), -y1 * (y1 ** 2 - 3 * y2 ** 2)]) * np.cos(4 * tau) + (1 / 16) * np.array([y1 * (y1 ** 2 - 3 * y2 ** 2), y2 * (y2 ** 2 - 3 * y1 ** 2)]) * np.sin(4 * tau))
            F = np.array([-(1/32)*y1**2*y2+(3/32)*y2**3-(1/16)*y2 , (5/32)*y1**3-(7/32)*y1*y2**2-(1/16)*y1])    +     (1 / 16) * np.array([y2 * (1 + 2 * y1 ** 2 - 2 * y2 ** 2), y1 * (1 + 2 * y2 ** 2 - 2 * y1 ** 2)]) * np.cos(2 * tau)     +    (1 / 16) * np.array([-y1 * (1 - 4 * y2 ** 2), y2 * (1 - 4 * y1 ** 2)]) * np.sin(2 * tau)     +     (1 / 32) * np.array([-y2 * (3 * y1 ** 2 - y2 ** 2), -y1 * (y1 ** 2 - 3 * y2 ** 2)]) * np.cos(4 * tau)     +     (1 / 32) * np.array([y1 * (y1 ** 2 - 3 * y2 ** 2), y2 * (y2 ** 2 - 3 * y1 ** 2)]) * np.sin(4 * tau)
            z = z + eps*F
        return z

    def dy_phi1(tau,y,eps):
        """Derivative w.r.t. y of the variable change associated to the dynamics of the ODE.
        Inputs:
        - tau: Float - Time variable
        - y: Array of shape (d,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = y.reshape(d, )
        z, F = np.eye(d, d), np.zeros((d, d))
        if dyn_syst == "Logistic":
            z = z + eps*F
        if dyn_syst == "Henon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            F = (1/2)*np.array([[y2 , y1 , 0 , 0] , [0 , 0 , 0 , 0] , [0 , -y3 , -y2 , 0] , [-y3 , 0 , -y1 , 0]])*(1-np.cos(2*tau)) + (1/2)*np.array([[0 , -y3 , -y2 , 0] , [0 , 0 , 0 , 0] , [-y2 , -y1 , 0 , 0] , [-y1 , 0 , y3 , 0]])*np.sin(2*tau)
            z = z + eps*F
        if dyn_syst == "VDP":
            y1, y2 = y[0], y[1]
            e1 , e2 = np.array([1,0]) , np.array([0,1])
            eta = 1e-6
            #F[:,0] = (ODE.phi1(tau,y+eta*e1,eps) - ODE.phi1(tau,y-eta*e1,eps))/(2*eta)
            #F[:,1] = (ODE.phi1(tau,y+eta*e2,eps) - ODE.phi1(tau,y-eta*e2,eps))/(2*eta)
            z1 = F
            #F = np.array([[-y1*y2/16 ,-y1**2/32 + 9*y2**2/32 - 1/16 ],[15*y1**2/32-7*y2**2/32-1/16 , -7*y1*y2/16]])   +    np.array([[y1*y2/4 ,y1**2/8-3*y2**2/8+1/16 ],[y2**2/8-3*y1**2/8+1/16 , y1*y2/4]])*np.cos(2*tau)   +   np.array([[y2**2/4-1/16 ,y1*y2/2 ],[-y1*y2/2 , -y1**2/4+1/16]])*np.sin(2*tau)   +   np.array([[-3*y1*y2/16 ,(y2**2-y1**2)*3/32 ],[(y2**2-y1**2)*3/32 , 3*y1*y2/16]])*np.cos(4*tau)   +   np.array([[(y1**2-y2**2)*3/32 , -3*y1*y2/16],[-3*y1*y2/16 , (y2**2-y1**2)*3/32]])*np.sin(4*tau)
            F = np.array([[-y1*y2/16 ,-y1**2/32 + 9*y2**2/32 - 1/16 ],[15*y1**2/32-7*y2**2/32-1/16 , -7*y1*y2/16]])   +    np.array([[y1*y2/4 ,y1**2/8-3*y2**2/8+1/16 ],[y2**2/8-3*y1**2/8+1/16 , y1*y2/4]])*np.cos(2*tau)   +   np.array([[y2**2/4-1/16 ,y1*y2/2 ],[-y1*y2/2 , -y1**2/4+1/16]])*np.sin(2*tau)    +   np.array([[-3*y1*y2/16 ,(y2**2-y1**2)*3/32 ],[(y2**2-y1**2)*3/32 , 3*y1*y2/16]])*np.cos(4*tau)    +   np.array([[(y1**2-y2**2)*3/32 , -3*y1*y2/16],[-3*y1*y2/16 , (y2**2-y1**2)*3/32]])*np.sin(4*tau)
            z = z + eps*F
            #print(np.linalg.norm(z1-z))
        return z

    def dtau_phi1(tau,y,eps):
        """Derivative w.r.t. tau of the variable change associated to the dynamics of the ODE.
        Inputs:
        - tau: Float - Time variable
        - y: Array of shape (d,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = y.reshape(d, )
        if dyn_syst == "Logistic":
            F = np.array([np.sin(tau)])
        if dyn_syst == "Henon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            F = np.array([-y2*y3 , 0 , -y1*y2 , (1/2)*(y3**2-y1**2)])*np.cos(2*tau) + np.array([y1*y2 , 0 , -y2*y3 ,-y1*y3])*np.sin(2*tau)
        if dyn_syst == "VDP":
            y1, y2 = y[0], y[1]
            F = (-1 / 8) * np.array([y2 * (1 + 2 * y1 ** 2 - 2 * y2 ** 2), y1 * (1 + 2 * y2 ** 2 - 2 * y1 ** 2)]) * np.sin(2 * tau)    +    (1 / 8) * np.array([-y1 * (1 - 4 * y2 ** 2), y2 * (1 - 4 * y1 ** 2)]) * np.cos(2 * tau)     +     (-1 / 8) * np.array([-y2 * (3 * y1 ** 2 - y2 ** 2), -y1 * (y1 ** 2 - 3 * y2 ** 2)]) * np.sin(4 * tau)   +   (1 / 8) * np.array([y1 * (y1 ** 2 - 3 * y2 ** 2), y2 * (y2 ** 2 - 3 * y1 ** 2)]) * np.cos(4 * tau)
        return eps*F

    def F1(y,eps):
        """First-order approximation of the averaged field associated to the ODE.
        Inputs:
        - y: Array of shape (d,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = y.reshape(d, )
        if dyn_syst == "Logistic":
            z = y
            F0 = y*(1-y)
            F1 = (1-2*y)
        if dyn_syst == "Henon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            #z = y
            F0 = np.array([y2*y3 , y4 , -y1*y2 , -(1/2)*(y1**2+y3**2-3*y2**2+2*y2)])
            F1 = (1/2)*np.array([-y1*y4 - 3*y2**2*y3 , -y1*y3 , y3*y4 - y1*y2**2 , -y1**2*y2 - 3*y2*y3**2])
        if dyn_syst == "VDP":
            y1, y2 = y[0], y[1]
            F0 = (1/8)*np.array([y1*(1-y1**2-y2**2) , y2*(1-y1**2-y2**2)])
            F1 = (1/256)*np.array([ -21*y2*y1**4 - 10*y1**2*y2**3 - 5*y2**5 + 22*y1**2*y2 + 6*y2**3-2*y2  , -14*y2**5 - 21*y2**4*y1 + (-16*y1**2+17)*y2**3 + (-6*y1**3+19*y1)*y2**2 + (-18*y1**4+21*y1**2-4)*y2 + 15*y1**5 - 17*y1**3 + 2*y1])
            #F1_1 = (1/256)*np.array([ -21*y2*y1**4 - 10*y1**2*y2**3 - 5*y2**5 + 22*y1**2*y2 + 6*y2**3-2*y2  , 21*y1**5 + (10*y2**2-22)*y1**3 + (5*y2**4+10*y2**2+2)*y1 ])
            #print(F1_1-F1)
        return F0 + eps*F1

    def I(ti,tf,w,v,eps):

        """Integral for integral numerical integrators.
        Inputs:
        - ti: Float - Initial integration time
        - tf: Float - Final integration time
        - w: Array of shape (d,) - Space variable
        - v: Array of shape (d,) - Source term
        - eps: Float - High oscillation parameter"""
        #I = v.reshape(d,)
        I = np.zeros_like(v).reshape(d,)

        #I = y.reshape(d,)
        if dyn_syst == "Logistic":
            #I = (ti-tf)*(1+eps*(1-2*y)) + np.array([np.sin(tf/eps) - np.sin(ti/eps) + np.cos(ti/eps) - np.cos(tf/eps)])
            I = (w*(1-w) - 2*w*(v+eps) - (3/2)*np.array([eps**2]))*(tf-ti)/eps + (eps*(2*w-1) + 2*eps*(v+eps))*(np.sin(tf/eps)-np.sin(ti/eps)) - (1/4)*np.array([eps**2])*(np.sin(2*tf/eps)-np.sin(2*ti/eps))
        if dyn_syst == "Henon-Heiles":
            v1, v2, v3, v4 = v[0], v[1], v[2], v[3]
            w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
            #z = v
            W1, W2, W3, W4 = v1 + w1, v2 + w2, v3 + w3, v4 + w4
            V1, V2, V3 = v1 * v2, v2 * v3, v3 * v1

            I[0] = W2 * ((W3 - eps * V2)*(tf-ti)/eps     -      (W3 - eps * V2) * (np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2     +      W1 * (np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2)
            I[1] = (v4 + w4 - (eps / 2) * v1 * v3)*(tf-ti)/eps      +      (eps / 2) * v1 * v3 * (np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2    +     (eps / 4) * (v3 ** 2 - v1 ** 2) * (np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2
            I[2] = -W2 * (W1*(tf-ti)/eps    +    W1 * (np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2    +    (W3 - eps * V2) * (np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2)
            I[3] = (-(1 / 2) * (W1 ** 2 + W3 ** 2 - 3 * W2 ** 2 + 2 * W2) + eps * V2 * W3 - (eps ** 2 / 2) * V2 ** 2)*(tf-ti)/eps    +      ((1 / 2) * (W3 ** 2 - W1 ** 2) - eps * V2 * W3 + (eps ** 2 / 2) * V2 ** 2) * (np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2      +      (-W1 * W3 + eps * V2 * W1) * (np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2

            I = I - np.array([-v2*v3 , 0 , -v1*v2 , (1/2)*(v3**2-v1**2)])*(np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2 - np.array([v1*v2 , 0 , -v2*v3 ,-v1*v3])*(np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2
            I = I - (np.eye(d,d)*(tf-ti)/eps + eps*(1/2)*np.array([[v2 , v1 , 0 , 0] , [0 , 0 , 0 , 0] , [0 , -v3 , -v2 , 0] , [-v3 , 0 , -v1 , 0]])*((tf-ti)/eps   -    (np.sin(2 * tf / eps) - np.sin(2 * ti / eps))/2) + eps*(1/2)*np.array([[0 , -v3 , -v2 , 0] , [0 , 0 , 0 , 0] , [-v2 , -v1 , 0 , 0] , [-v1 , 0 , v3 , 0]])*(np.cos(2 * ti/eps) - np.cos(2 * tf/eps))/2)@ODE.F1(v,eps)

            #I[0] = W2 * ((W3 - eps * V2) * (tf - ti) / eps - (W3 - eps * V2) * (np.sin(2 * tf / eps) - np.sin(2 * ti / eps)) / 2 + W1 * (np.cos(2 * ti / eps) - np.cos(2 * tf / eps)) / 2)

        return eps*I

    def Exact_solve(T , h, eps):
        """Exact resolution of the ODE by using a very accurate Python Integrator DOP853
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter"""

        def func(t , y):
            """Function for exact solving of the ODE:
            Inputs:
            - t: Float - Time
            - y: Array of shape (1,) - Space variable"""
            return ODE.f(t/eps , y)


        Y = solve_ivp(fun = func , t_span=(0,T) , y0 = Y0 , t_eval=np.arange(0,T,h) , atol = 1e-13 , rtol = 1e-13 , method = "DOP853").y
        return Y

    def Num_Solve_MM(T , h , eps , num_meth = "Forward Euler" , print_step = True):
        """Numerical resolution of the ODE - Micro-Macro method.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - print_step: Boolean - Print the steps of computation. Default: True"""

        TT = np.arange(0, T, h)

        if num_meth != "Integral_Euler_Modified_AppInt":

            if print_step == True:
                print(" -> Resolution of the first ODE with averaged field...")
            VV = np.zeros((d, np.size(TT)))
            VV[:,0] = Y0
            for n in range(np.size(TT)-1):
                if num_meth == "Forward Euler":
                    VV[:, n + 1] = VV[:, n] + h * ODE.F1(VV[:, n], eps)
                if num_meth == "Integral_order1":
                    VV[:, n + 1] = VV[:, n] + h * ODE.F1(VV[:, n], eps)
                if num_meth == "Integral_Euler_AppInt" or num_meth == "Integral_Symp_Euler_AppInt":
                    VV[:, n + 1] = VV[:, n] + h * ODE.f_avg(VV[:, n])
                if num_meth == "Integral_RK2":
                    VV[:, n + 1] = VV[:, n] + h * ODE.F1(VV[:, n] + (h / 2) * ODE.F1(VV[:, n], eps), eps)
                if num_meth == "Integral_MidPoint" or num_meth == "Integral_MidPoint_AppInt":
                    def F_iter(y):
                        """Iteration function for the MidPoint Rule.
                        Inputs:
                        - y:  Space variable"""
                        f = VV[:, n] + h*ODE.F1((VV[:, n]+y)/2, eps)
                        return f

                    N_iter = 5
                    yy = VV[:, n]
                    for i in range(N_iter):
                        yy = F_iter(yy)
                    VV[:, n + 1] = yy


            if print_step == True:
                print(" -> Resolution of the second ODE...")
            WW = np.zeros_like(VV)
            WW[:,0] = np.zeros_like(Y0)

            def g(tau,w,v):
                """Dynamics associated to the resolution of the second equation.
                Inputs:
                - tau: Float - Time variable
                - w: Array of shape (d,) - Space variable
                - v: Array of shape (d,) - Source term"""
                z = ODE.f(tau , ODE.phi1(tau , v , eps) + w) - (1/eps)*ODE.dtau_phi1(tau,v,eps) - ODE.dy_phi1(tau,v,eps)@ODE.F1(v,eps)
                return z

            def g_avg(w,v):
                """Average dynamics associated to the resolution of the second equation.
                Inputs:
                - w: Array of shape (d,) - Space variable
                - v: Array of shape (d,) - Source term"""
                if dyn_syst == "Logistic":
                    z = w*(1-w) - 2*w*(v+eps) - (3/2)*np.array([eps**2])

                if dyn_syst == "VDP":
                    v1,v2,w1,w2 = v[0],v[1],w[0],w[1]
                    z = np.zeros(2)
                    #z[0] = (1/131072)*(-11*v2**9+(-178*v1**2+730)*v2**7-160*v1*v2**6+(244*v1**4+1234*v1**2-1024)*v2**5+(-320*v1**3+192*v1)*v2**4+(-14*v1**6+2678*v1**4-3248*v1**2+480)*v2**3+(-672*v1**5+704*v1**3-64*v1)*v2**2+(-41*v1**8-66*v1**6-624*v1**4+768*v1**2-64)*v2)*eps**3+(1/131072)*((1968*v1-304*w1)*v2**6+(2272*v1*w2-896)*v2**5+(-240*v1**3+848*v1**2*w1+768*v1+256*w1)*v2**4+(-1088*v1**3*w2+6656*v1**2-256*v1*w2)*v2**3+(-496*v1**5+304*v1**4*w1+1792*v1**3-768*v1**2*w1-2048*v1)*v2**2+(-800*v1**5*w2+10624*v1**4+2816*v1**3*w2-11776*v1**2-1536*v1*w2+512)*v2-(1360*(v1**2-12/17))*(v1+w1)*v1**2*(v1**2-4/5))*eps**2+(1/131072)*(-2560*v2**5-14336*v2**4*w2+(-5120*v1**2-8192*v1*w1-3072*w1**2-7168*w2**2-9216)*v2**3-8192*w2*(v1**2+(1/4)*v1*w1-3/2)*v2**2+(-10752*v1**4-8192*v1**3*w1+(1024*w1**2-3072*w2**2+15360)*v1**2+12288*v1*w1+2048*w1**2+6144*w2**2+7168)*v2-10240*w2*(v1+w1)*v1*(v1**2-4/5))*eps-(1/8)*v2**2*w1-(1/4)*w2*(v1+w1)*v2-3*v1**2*w1*(1/8)+(1/131072)*(-49152*w1**2-16384*w2**2)*v1-(1/8)*w1*(w1**2+w2**2-1)
                    #z[1] = (1/131072)*(-11*v1*v2**8-560*v2**7+(166*v1**3-1120*v1)*v2**6+(80*v1**2+512)*v2**5+(-308*v1**5-2084*v1**3+1376*v1)*v2**4+(48*v1**4+704*v1**2-32)*v2**3+(282*v1**7-4968*v1**5+4976*v1**3-448*v1)*v2**2+(5040*v1**6-5952*v1**4+1184*v1**2-64)*v2-129*v1**9+156*v1**7-48*v1**5)*eps**3+(1/131072)*(176*v2**7+176*v2**6*w2+(-2032*v1**2+224*v1*w1+4992)*v2**5+(-2256*v1**2*w2+5376*v1-256*w2)*v2**4+(2320*v1**4-1088*v1**3*w1-3584*v1**2-256*v1*w1-4096)*v2**3+(3408*v1**4*w2-1792*v1**3-1792*v1**2*w2-3584*v1)*v2**2+(-592*v1**6+1248*v1**5*w1+12928*v1**4-1280*v1**3*w1-7424*v1**2+512*v1*w1+512)*v2-1840*w2*v1**2*(v1**4-96/115*(v1**2)+16/115))*eps**2+(1/131072)*((2560*v1+2048*w1)*v2**4+(24576*v1*w2+2048*w1*w2-12288)*v2**3+(5120*v1**3+24576*v1**2*w1+(7168*w1**2+11264*w2**2+5120)*v1-4096*w1)*v2**2+(-8192*v1**3*w2+(10240*w1*w2+4096)*v1**2+4096*v1*w2-4096*w1*w2+8192)*v2+(10752*(v1**4+4*v1**3*w1*(1/7)+(2*w1**2*(1/7)-6*w2**2*(1/7)-22/21)*v1**2-16*v1*w1*(1/21)-8*w1**2*(1/21)+8*w2**2*(1/21)+2/21))*v1)*eps-3*v2**2*w2*(1/8)+(1/131072)*(-32768*v1*w1-16384*w1**2-49152*w2**2)*v2-(1/8)*w2*(v1**2+2*v1*w1+w1**2+w2**2-1)

                    #v1,v2,w1,w2 = np.random.uniform(low=-1,high=1,size=(4,))

                    z[0] = (1/131072)*(-41*v1**8*v2+(-14*v2**3+270*v2)*v1**6+(244*v2**5-186*v2**3-304*v2)*v1**4+(-178*v2**7-126*v2**5+144*v2**3+96*v2)*v1**2-11*v2**9+10*v2**7)*eps**3 + (1/131072)*(-1024*v1**7-1360*v1**6*w1+(-4032*v2**2-800*v2*w2+2368)*v1**5+(304*v2**2*w1+2048*w1)*v1**4+(-1920*v2**4-1088*v2**3*w2+6400*v2**2+2816*v2*w2-1440)*v1**3+(848*v2**4*w1-768*v2**2*w1-768*w1)*v1**2+(1088*v2**6+2272*v2**5*w2-1088*v2**4-256*v2**3*w2-1568*v2**2-1536*v2*w2+64)*v1-304*v2**6*w1+256*v2**4*w1)*eps**2+(1/131072)*(-10240*v1**4*w2-8192*w1*(v2+5*w2*(1/4))*v1**3+(-8192*v2**2*w2+(1024*w1**2-3072*w2**2)*v2+8192*w2)*v1**2-8192*w1*(v2**3+(1/4)*v2**2*w2-3/2*v2-w2)*v1-14336*v2*(v2**3*w2+(3*w1**2*(1/14)+(1/2)*w2**2)*v2**2-6*v2*w2*(1/7)-(1/7)*w1**2-3*w2**2*(1/7)))*eps-3*v1**2*w1*(1/8)+(1/131072)*(-32768*v2*w2-49152*w1**2-16384*w2**2)*v1-(1/8)*w1*(v2**2+2*v2*w2+w1**2+w2**2-1)
                    z[1] = (1/131072)*(-129*v1**9+(282*v2**2+156)*v1**7+(-308*v2**4-264*v2**2-48)*v1**5+(166*v2**6+156*v2**4+48*v2**2)*v1**3+(-11*v2**8+32*v2**4)*v1)*eps**3  +  (1/131072)*((9152*v2-1840*w2)*v1**6+1248*v1**5*v2*w1+(4608*v2**3+3408*v2**2*w2-10624*v2+1536*w2)*v1**4+(-1088*v2**3*w1-1280*v2*w1)*v1**3+(-832*v2**5-2256*v2**4*w2+896*v2**3-1792*v2**2*w2+1888*v2-256*w2)*v1**2+224*v2*(v2**4-8/7*(v2**2)+16/7)*w1*v1-384*v2**7+176*v2**6*w2+256*v2**5-256*v2**4*w2-32*v2**3-64*v2)*eps**2+(1/131072)*(6144*v1**4*w1+(-8192*v2*w2+3072*w1**2-9216*w2**2)*v1**3+24576*w1*(v2**2+(5/12*v2)*w2-1/3)*v1**2+(24576*v2**3*w2+(7168*w1**2+11264*w2**2)*v2**2+4096*v2*w2-4096*w1**2+4096*w2**2)*v1+2048*v2*w1*(v2**2-2)*(v2+w2))*eps-(1/8)*v1**2*w2-(1/4)*w1*(v2+w2)*v1-3*v2**2*w2*(1/8)+(1/131072)*(-16384*w1**2-49152*w2**2)*v2-(1/8)*w2*(w1**2+w2**2-1)


                    # Test to check the result

                    #delta_tau = 1e-1
                    #g_avg_integral = np.array([0.0,0.0])
                    #g_avg_integral_gauss = np.array([0.0,0.0])
                    #tt = np.arange(0,2*np.pi,delta_tau)

                    #xi_0, omega_0 = npp.polynomial.legendre.leggauss(10)
                    #xi_0 = [np.pi + np.pi * ksi for ksi in xi_0]

                    # for k in range(np.size(tt)-1):
                    #     g_avg_integral += delta_tau*(g(tt[k],w,v)+g(tt[k+1],w,v))/2
                    # g_avg_integral = g_avg_integral/(2*np.pi)
                    #for k in range(len(xi_0)):
                    #   #g_avg_integral_gauss = g_avg_integral_gauss + np.pi*omega_0[k]*g(xi_0[k],np.array([w1,w2]),np.array([v1,v2]))
                    #   g_avg_integral_gauss += np.pi*omega_0[k]*g(xi_0[k],np.array([w1,w2]),np.array([v1,v2]))
                    #g_avg_integral_gauss = g_avg_integral_gauss/(2*np.pi)

                    #print("g_avg:",z)
                    #print("g_avg_integral:",g_avg_integral)
                    #print("g_avg_integral_gauss:",g_avg_integral_gauss)
                    #print("Diff=", np.linalg.norm(z-g_avg_integral))
                    #print("Diff_gauss=", np.linalg.norm(z-g_avg_integral_gauss)/np.linalg.norm(g_avg_integral_gauss))
                #return g_avg_integral_gauss
                return z

            def rg(tau,w ,v):
                """Remainder, difference between g and its average field.
                Inputs:
                - tau: Float - Time variable
                - w: Array of shape (d,) - Space variable
                - v: Array of shape (d,) - Source term"""
                return g(tau , w , v) - g_avg(w ,v)

            def g_0(tau , w , v):
                """Micro-Macro vector field at order 0 for equation w.r.t. w.
                Inputs:
                - tau: Float - Time variable
                - w: Array of shape (d,) - Space variable
                - w: Array of shape (d,) - Source term"""
                return ODE.f(tau , w + v) - ODE.f_avg(v)

            def g_0_avg(w,v):
                """Average dynamics associated to the resolution of the second equation for Integral Euler scheme.
                Inputs:
                - w: Array of shape (d,) - Space variable
                - v: Array of shape (d,) - Source term"""
                z = ODE.f_avg(w + v) - ODE.f_avg(v)
                return z

            def rg_0(tau , w , v):
                """Remainder, difference between g_0 and its average field (adapted for Integral Euler scheme).
                Inputs:
                - tau: Float - Time variable
                - w: Array of shape (d,) - Space variable
                - v: Array of shape (d,) - Source term"""
                return g_0(tau , w , v) - g_0_avg(w,v)

            #TIME = []

            for n in range(np.size(TT)-1):

                xi, omega = npp.polynomial.legendre.leggauss(n_Gauss)

                if num_meth == "Forward Euler":
                    WW[:, n + 1] = WW[:, n] + h * g(TT[n] / eps, WW[:, n], VV[:, n])
                if num_meth == "Integral_order1":
                    #WW[:, n + 1] = WW[:, n] + ODE.I(TT[n] , TT[n] + h , WW[:, n] + ODE.phi1(TT[n]/eps , VV[:, n] , eps) , eps)
                    WW[:, n + 1] = WW[:, n] + ODE.I(TT[n] , TT[n] + h , WW[:, n] , VV[:, n] , eps)
                if num_meth == "Integral_RK2":
                    VV_n_1_2 = VV[:,n] + (h/2)*ODE.F1(VV[:,n] , eps)
                    #WW_n_1_2 = WW[:,n] + ODE.I(TT[n] , TT[n] + h/2 , WW[:,n] + ODE.phi1(TT[n]/eps , VV[:, n] + ODE.phi1(TT[n]/eps , VV[:, n] , eps) , eps) , eps)
                    WW_n_1_2 = WW[:,n] + ODE.I(TT[n] , TT[n] + h/2 , WW[:,n] , VV[:, n] , eps)
                    WW[:,n+1] = WW[:,n] + ODE.I(TT[n] , TT[n]+h , WW_n_1_2 , VV_n_1_2 , eps)
                if num_meth == "Integral_MidPoint":
                    def F_iter(y):
                        """Iteration function for the Integral MidPoint Rule.
                        Inputs:
                        - y:  Space variable"""
                        f = WW[:, n] + ODE.I(TT[n], TT[n] + h, (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2, eps)
                        #if dyn_syst == "Logistic":
                        #    f = WW[:,n] + ODE.I(TT[n] , TT[n]+h , (WW[:,n]+y)/2 , (VV[:,n]+VV[:,n+1])/2 , eps)
                        #if dyn_syst == "Henon-Heiles":
                        #    f = WW[:, n] + ODE.I(TT[n], TT[n] + h, (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2, eps)# - ODE.phi1((n+1)*h/eps , VV[:, n + 1] , eps) + ODE.phi1(n*h/eps , VV[:, n] , eps)
                        return f

                    N_iter = 10
                    yy = WW[:,n]
                    for i in range(N_iter):
                        yy = F_iter(yy)

                    WW[:,n+1] = yy# - ODE.phi1((n+1)*h/eps , VV[:,n+1] , eps) + ODE.phi1(n*h/eps , VV[:,n] , eps)
                if num_meth == "Integral_Euler_AppInt":
                    #xi = [np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)) , -np.sqrt(3 / 7 - (2 / 7) * np.sqrt(6 / 5)) , np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5)) , -np.sqrt(3 / 7 + (2 / 7) * np.sqrt(6 / 5))]
                    #omega = [(18 + np.sqrt(30)) / 36 , (18 + np.sqrt(30)) / 36 , (18 - np.sqrt(30)) / 36 , (18 - np.sqrt(30)) / 36]
                    #xi = [-0.1488743389816312, 0.1488743389816312, -0.4333953941292472, 0.4333953941292472,-0.6794095682990244, 0.6794095682990244, -0.8650633666889845, 0.8650633666889845,-0.9739065285171717, 0.9739065285171717]
                    #omega = [0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2692667193099963,0.2190863625159820, 0.2190863625159820, 0.1494513491505806, 0.1494513491505806,0.0666713443086881, 0.0666713443086881]
                    t_n = TT[n]
                    t_n_1 = (TT[n] + h)
                    tau_n = t_n / eps + np.round(h / (2 * np.pi * eps)) * 2 * np.pi
                    tau_n_1 = t_n_1 / eps
                    tau = [(tau_n_1 - tau_n) / 2 * ksi + (tau_n + tau_n_1) / 2 for ksi in xi]

                    #tau_1 , tau_2 , tau_3 , tau_4 = (tau_n_1-tau_n)*xi_1/2 + (tau_n+tau_n_1)/2 , (tau_n_1-tau_n)*xi_2/2 + (tau_n+tau_n_1)/2 , (tau_n_1-tau_n)*xi_3/2 + (tau_n+tau_n_1)/2 , (tau_n_1-tau_n)*xi_4/2 + (tau_n+tau_n_1)/2
                    #ww = WW[:, n] + h*g_0_avg(WW[:,n] , VV[:,n])
                    #ww = ww + eps*(tau_n_1-tau_n)*(rg_0(tau_n_1, WW[:, n] , VV[:,n])+rg_0(tau_n, WW[:, n] , VV[:,n]))/2
                    #

                    #time_0 = time.time()

                    #F = [eps*(tau_n_1 - tau_n) / 2 * omega[i] * rg_0(tau[i], WW[:, n] , VV[:,n]) for i in range(len(xi))]
                    #F = [eps * (tau_n_1 - tau_n) / 2 * omega[i] for i in range(len(xi))]


                    #ww = WW[:, n] + h*g_avg(WW[:,n] , VV[:,n]) + np.sum(F)

                    ww = WW[:, n] + h*g_0_avg(WW[:,n] , VV[:,n]) # + np.sum(F)
                    for i in range(len(xi)):
                        ww = ww + eps*(tau_n_1 - tau_n) / 2 * omega[i] * rg_0(tau[i], WW[:, n] , VV[:,n])
                        #ww = ww + F[i]



                    ################

                    #TIME.append(time.time()-time_0)
                    #WW[:, n + 1] = WW[:, n] + h*g_0_avg(WW[:,n] , VV[:,n]) + eps*(tau_n_1-tau_n)/2*(omega_1*rg_0(tau_1 , WW[:, n] , VV[:, n]) + omega_2*rg_0(tau_2 , WW[:, n] , VV[:, n]) + omega_3*rg_0(tau_3 , WW[:, n] , VV[:, n]) + omega_4*rg_0(tau_4 , WW[:, n] , VV[:, n]) )

                    #WW[:,n+1] = WW[:,n] + (h/2)*(g_0(t_n/eps , WW[:,n] , VV[:,n]) + g_0((t_n+h)/eps , WW[:,n] , VV[:,n]))
                    # N_int = 100
                    # ww = WW[:,n]
                    # for j in range(N_int):
                    #     tau_n = t_n/eps + np.round(h/(2*np.pi*eps))*2*np.pi + (j/N_int)*(h/eps - np.round(h/(2*np.pi*eps))*2*np.pi)
                    #     tau_n_1 = t_n/eps + np.round(h/(2*np.pi*eps))*2*np.pi + ((j+1)/N_int)*(h/eps - np.round(h/(2*np.pi*eps))*2*np.pi)
                    #     ww = ww + eps*(tau_n_1-tau_n)/2*(g_0(tau_n , WW[:,n] , VV[:,n]) + g_0(tau_n_1 , WW[:,n] , VV[:,n]))
                    # WW[:, n + 1] = ww
                    WW[: , n+1] = ww
                if num_meth == "Integral_Symp_Euler_AppInt":
                    t_n = TT[n]
                    t_n_1 = (TT[n] + h)
                    tau_n = t_n / eps + np.round(h / (2 * np.pi * eps)) * 2 * np.pi
                    tau_n_1 = t_n_1 / eps
                    tau = [(tau_n_1 - tau_n) / 2 * ksi + (tau_n + tau_n_1) / 2 for ksi in xi]

                    ww = WW[:, n] + h * g_0_avg(WW[:, n], VV[:, n])
                    # ww = ww + eps*(tau_n_1-tau_n)*(rg_0(tau_n_1, WW[:, n] , VV[:,n])+rg_0(tau_n, WW[:, n] , VV[:,n]))/2
                    for i in range(len(xi)):
                        ww = ww + eps * (tau_n_1 - tau_n) / 2 * omega[i] * rg_0(tau[i], WW[:, n], VV[:, n])

                    WW[:, n + 1] = ww

                if num_meth == "Integral_MidPoint_AppInt":
                    def F_iter(y):
                        """Iteration function for the Integral MidPoint Rule with Approximations of Integrals.
                        Inputs:
                        - y:  Space variable"""
                        #xi_4_1 = np.sqrt(3/7 - (2/7)*np.sqrt(6/5))
                        #xi_4_2 = np.sqrt(3/7 + (2/7)*np.sqrt(6/5))
                        #omega_4_1 = (18 + np.sqrt(30))/36
                        #omega_4_2 = (18 - np.sqrt(30))/36
                        #t_n = TT[n]
                        #tau_n = t_n/eps + np.round(h / (2 * np.pi * eps)) * 2 * np.pi
                        #tau_n_1 = (t_n+h)/eps
                        #f = WW[:, n]  + h*g_avg((WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)    +   (tau_n_1 - tau_n)*eps*( rg(tau_n , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) + rg(tau_n_1 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )/2
                        #f = WW[:, n]  + h*g_avg((WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)    +   (tau_n_1 - tau_n)/2*eps*( omega_4_1*rg((tau_n_1-tau_n)/2*xi_4_1 + (tau_n+tau_n_1)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)  +  omega_4_1*rg(-(tau_n_1-tau_n)/2*xi_4_1 + (tau_n+tau_n_1)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)  +  omega_4_2*rg((tau_n_1-tau_n)/2*xi_4_2 + (tau_n+tau_n_1)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)  +  omega_4_2*rg(-(tau_n_1-tau_n)/2*xi_4_2 + (tau_n+tau_n_1)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )
                        #f = WW[:, n]  + h*g_avg((WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)    +   (tau_n_1 - tau_n)*eps*( rg(tau_n , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) + 4*rg((tau_n_1+tau_n)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) + rg(tau_n_1 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )/6
                        #f = WW[:, n]  + h*g_avg((WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)    +   (tau_n_1 - tau_n)*eps*( rg((t_n+h)/eps , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) + rg(t_n/eps + np.round(h/(2*np.pi*eps))*2*np.pi , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )/6
                        #N_int = int(1/h**0.25)
                        #N_int = 1
                        # for j in range(N_int):
                        #     tau_n = t_n/eps + np.round(h/(2*np.pi*eps))*2*np.pi + (j/N_int)*(h/eps - np.round(h/(2*np.pi*eps))*2*np.pi)
                        #     tau_n_1 = t_n/eps + np.round(h/(2*np.pi*eps))*2*np.pi + ((j+1)/N_int)*(h/eps - np.round(h/(2*np.pi*eps))*2*np.pi)
                        #     f = f + (tau_n_1-tau_n)*(eps/2)*( rg(tau_n , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) + rg(tau_n_1 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )
                        #     #f = f + (tau_n_1-tau_n)*(eps/6)*( rg(tau_n , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)  +  4*rg((tau_n+tau_n_1)/2 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)  + rg(tau_n_1 , (WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2) )

                        #xi = [-0.1488743389816312, 0.1488743389816312, -0.4333953941292472, 0.4333953941292472,-0.6794095682990244, 0.6794095682990244, -0.8650633666889845, 0.8650633666889845,-0.9739065285171717, 0.9739065285171717]
                        #omega = [0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2692667193099963,0.2190863625159820, 0.2190863625159820, 0.1494513491505806, 0.1494513491505806,0.0666713443086881, 0.0666713443086881]

                        t_n = TT[n]
                        t_n_1 = (TT[n] + h)
                        tau_n = t_n / eps# + np.round(h / (2 * np.pi * eps)) * 2 * np.pi
                        tau_n_1 = t_n_1 / eps
                        #tau_n = t_n / (2*np.pi*eps) + np.round(h / (2 * np.pi * eps))
                        #tau_n_1 = t_n_1 / (2*np.pi*eps)
                        #tau = [(tau_n_1 - tau_n) / 2 * ksi + (tau_n + tau_n_1) / 2 for ksi in xi]
                        tau = [(t_n_1 - t_n) / 2 * ksi + (t_n + t_n_1) / 2 for ksi in xi]

                        #print(np.abs(tau_n_1-tau_n))
                        ww = WW[:, n]# + h * g_avg((WW[:, n] + y) / 2, (VV[:, n] + VV[:, n + 1]) / 2)
                        for i in range(len(xi)):
                            #ww = ww + eps * (tau_n_1 - tau_n) / 2 * omega[i] * g(tau[i] , (WW[:, n] + y) / 2 , (VV[:, n] + VV[:, n + 1]) / 2)
                            ww = ww + h/ 2 * omega[i] * g(tau[i]/eps , (WW[:, n] + y) / 2 , (VV[:, n] + VV[:, n + 1]) / 2)
                            #ww = ww + 2*np.pi*eps * (tau_n_1 - tau_n) / 2 * omega[i] * rg(2*np.pi*tau[i] , (WW[:, n] + y) / 2 , (VV[:, n] + VV[:, n + 1]) / 2)

                        #tau = [tau_n + (i/20)*(tau_n_1-tau_n) for i in range(21)]

                        #for i in range(len(tau)-1):
                        #    ww = ww + 2*np.pi*eps * (tau_n_1 - tau_n) / 2 * (rg(2*np.pi*tau[i+1] , (WW[:, n] + y) / 2 , (VV[:, n] + VV[:, n + 1]) / 2) + rg(2*np.pi*tau[i] , (WW[:, n] + y) / 2 , (VV[:, n] + VV[:, n + 1]) / 2))

                        return ww

                    N_iter = 5
                    yy = WW[:,n]
                    for i in range(N_iter):
                        yy = F_iter(yy)

                    WW[:,n+1] = yy

            #print("Average time for an iteration (s):", statistics.mean(TIME))

            if print_step == True:
                print(" -> Final computation...")
            YY = np.zeros_like(VV)
            for n in range(np.size(TT)):
                if num_meth == "Integral_Euler_AppInt" or num_meth == "Integral_Symp_Euler_AppInt":
                    YY[:, n] = VV[:, n] + WW[:, n]
                else:
                    YY[:,n] = ODE.phi1(TT[n]/eps , VV[:,n] , eps) + WW[:,n]

        if num_meth == "Integral_Euler_Modified_AppInt":
            YY = np.zeros((d, np.size(TT)))
            YY[:, 0] = Y0
            for n in range(np.size(TT) - 1):
                xi, omega = npp.polynomial.legendre.leggauss(n_Gauss)
                t_n = TT[n]
                t_n_1 = TT[n+1]
                tau_n = t_n/eps# + 2*np.pi*np.round(h/(2*np.pi*eps))
                tau_n_1 = t_n_1/eps
                ttau = [(tau_n_1 -tau_n) / 2 * ksi + (tau_n + tau_n_1) / 2 for ksi in xi]
                xi = [(t_n_1 - t_n) / 2 * ksi + (t_n + t_n_1) / 2 for ksi in xi]

                # xi = [(t_n_1 - t_n) / 2 * ksi + (tau_n + tau_n_1) / 2 for ksi in xi]
                # tt = [h / 2 * ksi + (t_n + t_n_1) / 2 for ksi in xi]
                # F = [h / 2 * omega[i] * ODE.f(tt[i]/eps, YY[:, n]) for i in range(len(xi))]
                # yy = YY[:, n]
                # for i in range(len(xi)):
                #     yy = yy + F[i]
                #
                # def I_f1(t,y):
                #     """Integral function w.r.t. second variable of f1, which is approximated (as all integrals here)
                #     Inputs:
                #     - t: Float - Time variable
                #     - y: Array of shape (d,) - Space variable"""
                #     ss = [(t-t_n)/2 * ksi + (t+t_n)/2 for ksi in xi]
                #     F1 = [(t-t_n)/2 * omega[j]*ODE.f1(t/eps , ss[j]/eps , y) for j in range(len(xi))]
                #     ff1 = np.zeros_like(y)
                #     for j in range(len(xi)):
                #         ff1 = ff1 + F1[j]
                #     return ff1
                #
                # FF = [h / 2 * omega[i] * I_f1(tt[i], YY[:, n]) for i in range(len(xi))]
                #
                # for i in range(len(xi)):
                #     yy = yy + FF[i]

                yy = YY[:, n] + h * ODE.f_avg(YY[:, n])

                F = [(tau_n_1 - tau_n) / 2 * omega[i] * (ODE.f(ttau[i], YY[:, n]) - ODE.f_avg(YY[:, n])) for i in range(len(xi))]
                for i in range(len(xi)):
                    yy = yy + eps*F[i]


                def I_f1(tau,y):
                    """Integral function w.r.t. second variable of f1, which is approximated (as all integrals here)
                    Inputs:
                    - tay: Float - Time variable
                    - y: Array of shape (d,) - Space variable"""
                    ssigma = [(tau-tau_n)/2 * ksi + (tau+tau_n)/2 for ksi in xi]
                    F1 = [(tau-tau_n)/2 * omega[j]*(ODE.f1(tau , ssigma[j] , y) - ODE.f_1_avg(y)) for j in range(len(xi))]
                    ff1 = np.zeros_like(y)
                    for j in range(len(xi)):
                        ff1 += F1[j]
                    return ff1



                yy += (h**2/2)*ODE.f_1_avg(YY[:,n])

                FF = [(tau_n_1-tau_n) / 2 * omega[i] * I_f1(ttau[i], YY[:, n]) for i in range(len(xi))]
                for i in range(len(xi)):
                    yy += eps**2*FF[i]

                YY[:, n + 1] = yy

        return YY

    def Plot_Solve(T , h , eps , num_meth = "Forward Euler" , save = False):
        """Numerical resolution of the ODE vs exact solution ploted.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - save: Boolean - Saves the figure or not. Default: False"""
        TT = np.arange(0,T,h)
        TT = TT.reshape(1,np.size(TT))
        Y_Exact , Y_Num  = ODE.Exact_solve(T,h,eps) , ODE.Num_Solve_MM(T,h,eps,num_meth)
        if dyn_syst == "Logistic":
            plt.figure()
            plt.scatter(TT , Y_Num , label = "Num solution" , color = "green")
            plt.plot(np.squeeze(TT) , np.squeeze(Y_Exact) , label = "Exact solution" , color = "red")
            plt.grid()
            plt.legend()
            plt.xlabel("t")
            plt.ylabel("y")
            plt.title("Exact solution vs Numerical solution - "+"$\epsilon = $"+str(eps))
            if save == True:
                plt.savefig("Integration_Micro-Macro_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()
        if dyn_syst == "VDP":
            Y_Num_bis , Y_Exact_bis = np.zeros_like(Y_Exact) , np.zeros_like(Y_Num)
            for n in range(np.size(TT)):
                VC = np.array([[np.cos(TT[0, n] / eps), np.sin(TT[0, n] / eps)],[-np.sin(TT[0, n] / eps), np.cos(TT[0, n] / eps)]])
                Y_Exact_bis[:, n], Y_Num_bis[:, n] = VC @ Y_Exact[:, n], VC @ Y_Num[:, n]

            plt.figure()
            plt.scatter(Y_Num_bis[0,:] , Y_Num_bis[1,:], label="Num solution", color="green")
            plt.plot(Y_Exact_bis[0,:] , Y_Exact_bis[1,:], label="Exact solution", color="red")
            plt.grid()
            plt.legend(loc = "upper left")
            plt.axis("equal")
            plt.axis("square")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.title("Exact solution vs Numerical solution - " + "$\epsilon = $" + str(eps))
            if save == True:
                plt.savefig("Integration_Micro-Macro_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()

            plt.figure()
            Error = npp.linalg.norm(Y_Num - Y_Exact, 2, axis=0)
            plt.semilogy(npp.squeeze(TT), Error, label="Num solution", color="green")
            plt.grid()
            plt.show()
        if dyn_syst == "Henon-Heiles":

            Y_Exact_VC , Y_Num_VC = np.zeros_like(Y_Exact) , np.zeros_like(Y_Num)
            for n in range(np.size(TT)):
                VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                Y_Exact_VC[:,n] , Y_Num_VC[:,n] = VC@Y_Exact[:,n] , VC@Y_Num[:,n]

            Ham = ODE.H(Y_Num_VC , eps)
            Ham_0 = ODE.H(Y0.reshape(d,1)@np.ones_like(TT) , eps)

            plt.figure(figsize = (12,5))
            plt.subplot(1, 2, 1)
            plt.scatter(Y_Num_VC[1, :], Y_Num_VC[3, :], s=5, label="Num solution", color="green")
            plt.plot(np.squeeze(Y_Exact_VC[1, :]), np.squeeze(Y_Exact_VC[3, :]), label="Exact solution", color="red")
            plt.grid()
            plt.legend()
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.title("$\epsilon = $" + str(eps))
            plt.axis("equal")
            plt.subplot(1, 2, 2)
            plt.plot(TT.reshape(np.size(TT), ), (Ham - Ham_0), label="Error on $H$")
            plt.xlabel("$t$")
            plt.ylabel("$H(y_n) - H(y_0)$")
            plt.legend()
            plt.title("Hamiltonian error")
            plt.grid()
            if save == True:
                num_meth_bis = "Forward_Euler"
                dyn_syst_bis = "Henon-Heiles"
                plt.savefig("Integration_Micro-Macro_"+dyn_syst_bis+"_"+num_meth_bis+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()
        pass

class Convergence(ODE):
    def Error(T , h , eps , num_meth):
        """Computes the relative error between exact solution an numerical approximation of the solution
        w.r.t. a selected numerical method.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - num_meth: Str - Name of the numerical method"""

        YY_exact = ODE.Exact_solve(T , h , eps)
        YY_app = ODE.Num_Solve_MM(T , h , eps , num_meth , print_step = False)

        norm_exact = np.max(np.linalg.norm(YY_exact , 2 , axis = 0))
        norm_error = np.max(np.linalg.norm(YY_exact - YY_app, 2 , axis = 0))

        error = norm_error/norm_exact

        return error

    def Curve(T , num_meth = "Forward Euler" ,save = False):
        """Plots a curve of convergence w.r.t. various numerical methods
        Inputs:
        - T: Float - Time for ODE simulations
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - save: Boolean - Saves the figure or not. Default: False"""
        Num_Meths = [num_meth]
        cmap = plt.get_cmap("jet")
        Colors = [cmap(k/n_EPS) for k in range(n_EPS)]
        HH = np.exp(np.linspace(np.log(step_h[0]),np.log(step_h[1]),n_HH))
        EPS = np.exp(np.linspace(np.log(step_eps[0]),np.log(step_eps[1]),n_EPS))
        E = np.zeros((len(Num_Meths),np.size(HH),np.size(EPS)))

        print(50 * "-")
        print("Loading...")
        print(50 * "-")
        for k in range(np.size(EPS)):
            for j in range(np.size(HH)):
                print(" - eps =  {}  \r".format(str(format(EPS[k], '.4E'))).rjust(3)," h = ",format(str(format(HH[j],'.4E'))).rjust(3), end=" ")

                for i in range(len(Num_Meths)):
                    E[i,j,k] = Convergence.Error(T , HH[j] , EPS[k] , Num_Meths[i])

        plt.figure()
        for k in range(np.size(EPS)):
            for i in range(len(Num_Meths)):
                plt.loglog(HH, E[i,:,k], marker="s" , linestyle="dashed" , color=Colors[k] , label = "$\epsilon = $"+str(format(EPS[k],'.2E')) , markersize = 5)
        plt.legend()
        plt.title("Integration errors - "+num_meth+" - "+dyn_syst)
        plt.xlabel("h")
        plt.ylabel("Rel. Error")
        plt.grid()
        if save == True:
            if num_meth == "Forward Euler":
                num_meth_bis = "Forward_Euler"
            if num_meth != "Forward Euler":
                num_meth_bis = num_meth
            if dyn_syst == "Henon-Heiles":
                dyn_syst_bis = "Henon-Heiles"
            if dyn_syst == "Logistic":
                dyn_syst_bis = dyn_syst
            plt.savefig("Convergence_Curve_Micro-Macro_"+dyn_syst+"_"+num_meth_bis+"_T="+str(T)+"_h="+str(step_h)+"_epsilon="+str(step_eps)+".pdf")
        plt.show()


        pass
