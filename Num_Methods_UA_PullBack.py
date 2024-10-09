import autograd.numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.integrate import quad_vec
from autograd import jacobian
import scipy.optimize
import matplotlib.pyplot as plt

# Numerical methods tested

# Parameters

T_simul = 1                  # Time for ODE simulations
h_simul = 0.01                # Time step for ODE simulations
eps_simul = 0.1              # High oscillation parameter for ODE simulations
N_iter = 10                  # Number of iterations for implicit methods
dyn_syst = "Hénon-Heiles"    # Dynamical system: "Logistic" or "Hénon-Heiles"
step_h = [0.001 , 0.5]       # Time step interval for convergence curves
step_eps = [0.001 , 0.5]     # High oscillation parameter interval for the convergence curves

if dyn_syst == "Logistic":
    d = 1
    Y0 = np.array([1.5])
if dyn_syst == "Hénon-Heiles":
    d = 4
    #Y0 = np.array([0,0,np.sqrt(eps_simul/3)*np.sin(5*np.pi/4),np.sqrt(1/3)*np.cos(5*np.pi/4)])
    Y0 = np.array([0,0,0.7,0.7])

print(" ")
print(60*"-")
print(" Uniformly Accurate methods for highly oscillatory ODE's - Pullback method")
print(60*"-")
print(" ")
print(" - Time for ODE simulations:" , T_simul)
print(" - Time step for ODE simulations:" , h_simul)
print(" - High oscillation parameter for ODE simulations" , eps_simul)
print(" - Number of iterations for implicit methods:", N_iter)
print(" - Dynamical system:" , dyn_syst)
print(" - Time step inteval for convergence curves:" , step_h)
print(" - High oscillation parameter interval for the convergence curves:" , step_eps)

class Tools:
    def Id():
        """Identity mapping"""
        if dyn_syst == "Logistic":
            y = sp.symbols("y")
            f = sp.Matrix([y])
        if dyn_syst == "Hénon-Heiles":
            y1 , y2 , y3 , y4 = sp.symbols("y1 y2 y3 y4")
            f = sp.Matrix([y1 , y2 , y3 , y4])
        return f

    def VF():
        """Vector field associated to the corresponding dynamical system"""
        eps , tau = sp.symbols("eps , tau")
        if dyn_syst == "Logistic":
            y = sp.symbols("y")
            f = sp.Matrix([y*(1-y) + sp.sin(tau)])
        if dyn_syst == "Hénon-Heiles":
            y1 , y2 , y3 , y4 = sp.symbols("y1 y2 y3 y4")
            f = sp.Matrix([2*sp.sin(tau)*(y1*sp.cos(tau) + y3*sp.sin(tau))*y2 , y4 , -2*sp.cos(tau)*(y1*sp.cos(tau) + y3*sp.sin(tau))*y2 , -(y1*sp.cos(tau) + y3*sp.sin(tau))**2 + (3/2)*y2**2 - y2 ])
        return f

    def Avg(f):
        """Computes the formal average of 2*pi-periodic w.r.t. tau vector field"""
        eps, tau = sp.symbols("eps , tau")
        return sp.simplify(sp.integrate(f , (tau,0,2*sp.pi))/(2*sp.pi))

    def Phi(g):
        """Computes the iteration Phi^{[k+1]} for a vector field g inetead of Phi^{[k]}"""
        eps, tau = sp.symbols("eps , tau")
        if dyn_syst == "Logistic":
            y = sp.symbols("y")
            F = Tools.Avg(g.jacobian([y])).inv()@Tools.Avg(Tools.VF().subs(y,g[0]))
            ff = Tools.VF().subs(y,g[0]) - g.jacobian([y])@F
            # print(F+ff)
            #print(ff)
            f = sp.Matrix([y]) + eps*sp.integrate(ff,(tau,0,tau))
            return sp.Matrix([sp.series(sp.simplify(f)[0],x = eps)])
        if dyn_syst == "Hénon-Heiles":
            y1 , y2 , y3 , y4 = sp.symbols("y1 y2 y3 y4")
            order = 4
            F = Tools.Avg(g.jacobian([y1 , y2 , y3 , y4])).inv()@Tools.Avg(Tools.VF().subs([y1 , y2 , y3 , y4],g[0]))
            ff = Tools.VF().subs([y1 , y2 , y3 , y4],g[0]) - g.jacobian([y1 , y2 , y3 , y4])@F
            f = sp.Matrix([y1 , y2 , y3 , y4]) + eps*sp.integrate(ff,(tau,0,tau))
            return sp.Matrix([sp.series(sp.simplify(f)[0],x = eps , n = order) , sp.series(sp.simplify(f)[1],x = eps , n = order) , sp.series(sp.simplify(f)[2],x = eps , n = order) , sp.series(sp.simplify(f)[3],x = eps , n = order)])

    def F(g):
        """Computes the iteration F^{[k]} for a vector field g inetead of Phi^{[k]}"""
        eps, tau = sp.symbols("eps , tau")
        if dyn_syst == "Logistic":
            y = sp.symbols("y")
            F = Tools.Avg(g.jacobian([y])).inv() @ Tools.Avg(Tools.VF().subs(y, g[0]))
            return sp.Matrix([sp.series(sp.simplify(F)[0],x = eps)])
        if dyn_syst == "Hénon-Heiles":
            y1 , y2 , y3 , y4 = sp.symbols("y1 y2 y3 y4")
            F = Tools.Avg(g.jacobian([y1 , y2 , y3 , y4])).inv() @ Tools.Avg(Tools.VF().subs([y1 , y2 , y3 , y4], g[0]))
            return sp.Matrix([sp.series(sp.simplify(F)[0], x=eps), sp.series(sp.simplify(F)[1], x=eps), sp.series(sp.simplify(F)[2], x=eps), sp.series(sp.simplify(F)[3], x=eps)])

    def h():
        """Computes the function h, used for MidPoint method for second order variable change"""
        eps, tau = sp.symbols("eps , tau")
        if dyn_syst == "Logistic":
            y = sp.symbols("y")
            f = Tools.VF()
            g = sp.integrate(f - Tools.Avg(f), (tau, 0, tau))
            gg = f.jacobian([y]) @ g - Tools.Avg(f.jacobian([y]) @ g) - (g.jacobian([y]) - Tools.Avg(g.jacobian([y]))) @ Tools.Avg(f)
            h = g + eps * sp.integrate(gg, (tau, 0, tau)) - (eps / 2) * g.jacobian([y]) @ g
            return sp.simplify(h)
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = sp.symbols("y1 y2 y3 y4")
            f = Tools.VF()
            g = sp.integrate(f - Tools.Avg(f) , (tau , 0 , tau))
            gg = f.jacobian([y1, y2, y3, y4])@g - Tools.Avg( f.jacobian([y1, y2, y3, y4])@g ) - ( g.jacobian([y1, y2, y3, y4]) - Tools.Avg(g.jacobian([y1, y2, y3, y4])) )@Tools.Avg(f)
            h = g + eps*sp.integrate(gg , (tau , 0 , tau)) - (eps/2)*g.jacobian([y1, y2, y3, y4])@g
            return sp.simplify(h)

class ODE:
    def f(tau , y):
        """Function involved in the averaging process:
         Inputs:
         - tau: Float - Time
         - y: Array of shape (1,) - Space variable"""

        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            z = y * (1 - y) + np.sin(tau)
        if dyn_syst == "Hénon-Heiles":
            z[0] = 2*np.sin(tau)*(y[0]*np.cos(tau)+y[2]*np.sin(tau))*y[1]
            z[1] = y[3]
            z[2] = -2*np.cos(tau)*(y[0]*np.cos(tau)+y[2]*np.sin(tau))*y[1]
            z[3] = -1*(y[0]*np.cos(tau)+y[2]*np.sin(tau))**2 + (3/2)*y[1]**2 - y[1]
        return z

    def H(y,eps):
        """Hamiltonian function of the system (only if the system is Hamiltonian);
        Inputs:
        - y: Array of shape (d,n) - Space variable
        - eps: Float - High oscillation parameter"""
        if dyn_syst == "Hénon-Heiles":
            q1 , q2 , p1 , p2 = y[0,:] , y[1,:] , y[2,:] , y[3,:]

            return p1**2/(2*eps) + p2**2/2 + q1**2/(2*eps) + q2**2/2 + q1**2*q2 -  (q2**3)/2

    def dyn(t, y , eps):
        """Dynamics of the ODE: Logistic equation:
        Inputs:
        - t: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""

        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        z = ODE.f(t/eps,y)
        return z


    def F_0(y,eps):
        """Approximation of the averaged field:
        Input:
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            z = y*(1-y)
            return z

    def F_1(y,eps):
        """Approximation of the averaged field:
        Input:
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            z = y*(1-y) + eps*(1-2*y) - np.array([(3/2)*eps**2])
            return z

    def F(y, eps, n_approx):
        """Approximation of the mapping F which is the Averaged field:
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations in order to approximate mapping F"""
        if n_approx == 0:
            return ODE.F_0(y , eps)
        if n_approx == 1:
            return ODE.F_1(y , eps)


    def Phi(tau , y , eps , n_approx):
        """Approximation of the mapping Phi :
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations in order to approximate mapping Phi"""

        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            if n_approx == 0:
                return y
            if n_approx == 1:
                z = y + eps * np.array([(1 - np.cos(tau))])
                return z
            if n_approx == 2:
                z = y + eps * np.array([(1 - np.cos(tau))]) + eps ** 2 * (2 * y - 1) * np.array([(np.sin(tau))]) + eps ** 3*np.array([2 * np.sin(tau) - np.sin(2 * tau) / 4])
                return z
        if dyn_syst == "Hénon-Heiles":
            y1 , y2 , y3 , y4 = y[0] , y[1] , y[2] , y[3]
            if n_approx == 0:
                return y
            if n_approx == 1:
                #z = np.array([eps * (-y1 * y2 * np.cos(2 * tau) / 2 + y1 * y2 / 2 - y2 * y3 * np.sin(2 * tau) / 2) + y1 , y2 , eps * (-y1 * y2 * np.sin(2 * tau) / 2 + y2 * y3 * np.cos(2 * tau) / 2 - y2 * y3 / 2) + y3,
                #    eps * (-y1 ** 2 * np.sin(2 * tau) / 2 + y1 * y3 * np.cos(2 * tau) - y1 * y3 + y3 ** 2 * np.sin(2 * tau) / 2) + y4])
                z = np.array([
                [                     eps*(-y1*y2*cos(2*tau)/2 + y1*y2/2 - y2*y3*sin(2*tau)/2) + y1],
                [                                                                                y2],
                [                     eps*(-y1*y2*sin(2*tau)/2 + y2*y3*cos(2*tau)/2 - y2*y3/2) + y3],
                [eps*(-y1**2*sin(2*tau)/4 + y1*y3*cos(2*tau)/2 - y1*y3/2 + y3**2*sin(2*tau)/4) + y4]]).reshape(d,)
                return z
            if n_approx == 2:

                # z =  np.array([y1 + eps * (-y1 * y2 * np.cos(2 * tau) / 2 + y1 * y2 / 2 - y2 * y3 * np.sin(2 * tau) / 2) + eps ** 2 * ( y1 * y2 ** 2 * np.cos(2 * tau) / 4 - y1 * y2 ** 2 / 4 + y1 * y4 * np.sin(2 * tau) / 4 + y2 ** 2 * y3 * np.sin(2 * tau) / 4 - y3 * y4 * np.cos(2 * tau) / 4 + y3 * y4 / 4) + eps ** 3 * (y1 * y2 ** 3 * np.cos(2 * tau) / 8 - y1 * y2 ** 3 / 8 - y1 * y2 * y4 * np.sin(2 * tau) / 8 - y2 ** 3 * y3 * np.sin(2 * tau) / 8 - y2 * y3 * y4 * np.cos(2 * tau) / 8 + y2 * y3 * y4 / 8),
                #      y2,
                #      y3 + eps * (-y1 * y2 * np.sin(2 * tau) / 2 + y2 * y3 * np.cos(2 * tau) / 2 - y2 * y3 / 2) + eps ** 2 * (y1 * y2 ** 2 * np.sin(2 * tau) / 4 - y1 * y4 * np.cos(2 * tau) / 4 + y1 * y4 / 4 + y2 ** 2 * y3 * np.sin(tau) ** 2 / 2 - y3 * y4 * np.sin(2 * tau) / 4) + eps ** 3 * (y1 * y2 ** 3 * np.sin(2 * tau) / 8 - y1 * y2 * y4 * np.sin(tau) ** 2 / 4 - y2 ** 3 * y3 * np.sin(tau) ** 2 / 4 - y2 * y3 * y4 * np.sin(2 * tau) / 8),
                #      y4 + eps * (-y1 ** 2 * np.sin(2 * tau) / 2 + y1 * y3 * np.cos(2 * tau) - y1 * y3 + y3 ** 2 * np.sin(2 * tau) / 2) + eps ** 2 * (y1 ** 2 * y2 * np.sin(2 * tau) / 2 - y1 * y2 * y3 * np.cos(2 * tau) + y1 * y2 * y3 - y2 * y3 ** 2 * np.sin(2 * tau) / 2) + eps ** 3 * (y1 ** 2 * y2 ** 2 * np.sin(2 * tau) / 4 + y1 ** 2 * y4 * np.cos(2 * tau) / 4 - y1 ** 2 * y4 / 4 + y2 ** 2 * y3 ** 2 * np.sin(2 * tau) / 4 + y3 ** 2 * y4 * np.cos(2 * tau) / 4 - y3 ** 2 * y4 / 4)])
                z = np.array([
                    [y1 + eps * (-y1 * y2 * cos(2 * tau) / 2 + y1 * y2 / 2 - y2 * y3 * sin(2 * tau) / 2) + eps ** 2 * (
                                y1 * y2 ** 2 * cos(2 * tau) / 4 - y1 * y2 ** 2 / 4 + y1 * y4 * sin(
                            2 * tau) / 4 + y2 ** 2 * y3 * sin(2 * tau) / 4 - y3 * y4 * cos(
                            2 * tau) / 4 + y3 * y4 / 4) + eps ** 3 * (
                                 y1 * y2 ** 3 * cos(2 * tau) / 8 - y1 * y2 ** 3 / 8 - y1 * y2 * y4 * sin(
                             2 * tau) / 8 - y2 ** 3 * y3 * sin(2 * tau) / 8 - y2 * y3 * y4 * cos(
                             2 * tau) / 8 + y2 * y3 * y4 / 8)],
                    [y2],
                    [y3 + eps * (-y1 * y2 * sin(2 * tau) / 2 + y2 * y3 * cos(2 * tau) / 2 - y2 * y3 / 2) + eps ** 2 * (
                                y1 * y2 ** 2 * sin(2 * tau) / 4 - y1 * y4 * cos(
                            2 * tau) / 4 + y1 * y4 / 4 + y2 ** 2 * y3 * sin(tau) ** 2 / 2 - y3 * y4 * sin(
                            2 * tau) / 4) + eps ** 3 * (y1 * y2 ** 3 * sin(2 * tau) / 8 - y1 * y2 * y4 * sin(
                        tau) ** 2 / 4 - y2 ** 3 * y3 * sin(tau) ** 2 / 4 - y2 * y3 * y4 * sin(2 * tau) / 8)],
                    [y4 + eps * (-y1 ** 2 * sin(2 * tau) / 4 + y1 * y3 * cos(2 * tau) / 2 - y1 * y3 / 2 + y3 ** 2 * sin(
                        2 * tau) / 4) + eps ** 2 * (y1 ** 2 * y2 * sin(2 * tau) / 4 - y1 * y2 * y3 * cos(
                        2 * tau) / 2 + y1 * y2 * y3 / 2 - y2 * y3 ** 2 * sin(2 * tau) / 4) + eps ** 3 * (
                                 y1 ** 2 * y2 ** 2 * sin(2 * tau) / 8 + y1 ** 2 * y4 * cos(
                             2 * tau) / 8 - y1 ** 2 * y4 / 8 + y2 ** 2 * y3 ** 2 * sin(
                             2 * tau) / 8 + y3 ** 2 * y4 * cos(2 * tau) / 8 - y3 ** 2 * y4 / 8)]]).reshape(d,)
                return z

    def d_y_Phi(tau , y , eps , n_approx):
        """derivative w.r.t. y of the approximation of the mapping Phi :
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations in order to approximate mapping Phi"""
        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            if n_approx == 0:
                return np.array([[1]])
            if n_approx == 1:
                return np.array([[1]])
            if n_approx == 2:
                return np.array([[1 + 2*eps**2*np.sin(tau)]])
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            if n_approx == 0:
                return np.eye(d)
            if n_approx == 1:
                # return np.array([[        eps*(-y2*np.cos(2*tau)/2 + y2/2) + 1, eps*(-y1*np.cos(2*tau)/2 + y1/2 - y3*np.sin(2*tau)/2),-eps*y2*np.sin(2*tau)/2, 0],[0,1,0,0],
                # [-eps*y2*np.sin(2*tau)/2, eps*(-y1*np.sin(2*tau)/2 + y3*np.cos(2*tau)/2 - y3/2),         eps*(y2*np.cos(2*tau)/2 - y2/2) + 1, 0],
                # [eps*(-y1*np.sin(2*tau) + y3*np.cos(2*tau) - y3),0, eps*(y1*np.cos(2*tau) - y1 + y3*np.sin(2*tau)), 1]])
                z = np.array([
                [              eps*(-y2*cos(2*tau)/2 + y2/2) + 1, eps*(-y1*cos(2*tau)/2 + y1/2 - y3*sin(2*tau)/2),                           -eps*y2*sin(2*tau)/2, 0],
                [                                              0,                                               1,                                              0, 0],
                [                           -eps*y2*sin(2*tau)/2, eps*(-y1*sin(2*tau)/2 + y3*cos(2*tau)/2 - y3/2),               eps*(y2*cos(2*tau)/2 - y2/2) + 1, 0],
                [eps*(-y1*sin(2*tau)/2 + y3*cos(2*tau)/2 - y3/2),                                               0, eps*(y1*cos(2*tau)/2 - y1/2 + y3*sin(2*tau)/2), 1]])
                return z
            if n_approx == 2:
                # return np.array([[        eps**3*(y2**3*np.cos(2*tau)/8 - y2**3/8 - y2*y4*np.sin(2*tau)/8) + eps**2*(y2**2*np.cos(2*tau)/4 - y2**2/4 + y4*np.sin(2*tau)/4) + eps*(-y2*np.cos(2*tau)/2 + y2/2) + 1, eps**3*(3*y1*y2**2*np.cos(2*tau)/8 - 3*y1*y2**2/8 - y1*y4*np.sin(2*tau)/8 - 3*y2**2*y3*np.sin(2*tau)/8 - y3*y4*np.cos(2*tau)/8 + y3*y4/8) + eps**2*(y1*y2*np.cos(2*tau)/2 - y1*y2/2 + y2*y3*np.sin(2*tau)/2) + eps*(-y1*np.cos(2*tau)/2 + y1/2 - y3*np.sin(2*tau)/2),                         eps**3*(-y2**3*np.sin(2*tau)/8 - y2*y4*np.cos(2*tau)/8 + y2*y4/8) + eps**2*(y2**2*np.sin(2*tau)/4 - y4*np.cos(2*tau)/4 + y4/4) - eps*y2*np.sin(2*tau)/2, eps**3*(-y1*y2*np.sin(2*tau)/8 - y2*y3*np.cos(2*tau)/8 + y2*y3/8) + eps**2*(y1*np.sin(2*tau)/4 - y3*np.cos(2*tau)/4 + y3/4)],[0,1,0,0],
                # [eps**3*(y2**3*np.sin(2*tau)/8 - y2*y4*np.sin(tau)**2/4) + eps**2*(y2**2*np.sin(2*tau)/4 - y4*np.cos(2*tau)/4 + y4/4) - eps*y2*np.sin(2*tau)/2,eps**3*(3*y1*y2**2*np.sin(2*tau)/8 - y1*y4*np.sin(tau)**2/4 - 3*y2**2*y3*np.sin(tau)**2/4 - y3*y4*np.sin(2*tau)/8) + eps**2*(y1*y2*np.sin(2*tau)/2 + y2*y3*np.sin(tau)**2) + eps*(-y1*np.sin(2*tau)/2 + y3*np.cos(2*tau)/2 - y3/2),                           eps**3*(-y2**3*np.sin(tau)**2/4 - y2*y4*np.sin(2*tau)/8) + eps**2*(y2**2*np.sin(tau)**2/2 - y4*np.sin(2*tau)/4) + eps*(y2*np.cos(2*tau)/2 - y2/2) + 1,         eps**3*(-y1*y2*np.sin(tau)**2/4 - y2*y3*np.sin(2*tau)/8) + eps**2*(-y1*np.cos(2*tau)/4 + y1/4 - y3*np.sin(2*tau)/4)],
                # [eps**3*(y1*y2**2*np.sin(2*tau)/2 + y1*y4*np.cos(2*tau)/2 - y1*y4/2) + eps**2*(y1*y2*np.sin(2*tau) - y2*y3*np.cos(2*tau) + y2*y3) + eps*(-y1*np.sin(2*tau) + y3*np.cos(2*tau) - y3),  eps**3*(y1**2*y2*np.sin(2*tau)/2 + y2*y3**2*np.sin(2*tau)/2) + eps**2*(y1**2*np.sin(2*tau)/2 - y1*y3*np.cos(2*tau) + y1*y3 - y3**2*np.sin(2*tau)/2), eps**3*(y2**2*y3*np.sin(2*tau)/2 + y3*y4*np.cos(2*tau)/2 - y3*y4/2) + eps**2*(-y1*y2*np.cos(2*tau) + y1*y2 - y2*y3*np.sin(2*tau)) + eps*(y1*np.cos(2*tau) - y1 + y3*np.sin(2*tau)),                                        eps**3*(y1**2*np.cos(2*tau)/4 - y1**2/4 + y3**2*np.cos(2*tau)/4 - y3**2/4) + 1]])
                z = np.array([
                    [eps ** 3 * (y2 ** 3 * cos(2 * tau) / 8 - y2 ** 3 / 8 - y2 * y4 * sin(2 * tau) / 8) + eps ** 2 * (
                                y2 ** 2 * cos(2 * tau) / 4 - y2 ** 2 / 4 + y4 * sin(2 * tau) / 4) + eps * (
                                 -y2 * cos(2 * tau) / 2 + y2 / 2) + 1, eps ** 3 * (
                                 3 * y1 * y2 ** 2 * cos(2 * tau) / 8 - 3 * y1 * y2 ** 2 / 8 - y1 * y4 * sin(
                             2 * tau) / 8 - 3 * y2 ** 2 * y3 * sin(2 * tau) / 8 - y3 * y4 * cos(
                             2 * tau) / 8 + y3 * y4 / 8) + eps ** 2 * (
                                 y1 * y2 * cos(2 * tau) / 2 - y1 * y2 / 2 + y2 * y3 * sin(2 * tau) / 2) + eps * (
                                 -y1 * cos(2 * tau) / 2 + y1 / 2 - y3 * sin(2 * tau) / 2),
                     eps ** 3 * (-y2 ** 3 * sin(2 * tau) / 8 - y2 * y4 * cos(2 * tau) / 8 + y2 * y4 / 8) + eps ** 2 * (
                                 y2 ** 2 * sin(2 * tau) / 4 - y4 * cos(2 * tau) / 4 + y4 / 4) - eps * y2 * sin(
                         2 * tau) / 2,
                     eps ** 3 * (-y1 * y2 * sin(2 * tau) / 8 - y2 * y3 * cos(2 * tau) / 8 + y2 * y3 / 8) + eps ** 2 * (
                                 y1 * sin(2 * tau) / 4 - y3 * cos(2 * tau) / 4 + y3 / 4)],
                    [0, 1, 0, 0],
                    [eps ** 3 * (y2 ** 3 * sin(2 * tau) / 8 - y2 * y4 * sin(tau) ** 2 / 4) + eps ** 2 * (
                                y2 ** 2 * sin(2 * tau) / 4 - y4 * cos(2 * tau) / 4 + y4 / 4) - eps * y2 * sin(
                        2 * tau) / 2, eps ** 3 * (3 * y1 * y2 ** 2 * sin(2 * tau) / 8 - y1 * y4 * sin(
                        tau) ** 2 / 4 - 3 * y2 ** 2 * y3 * sin(tau) ** 2 / 4 - y3 * y4 * sin(
                        2 * tau) / 8) + eps ** 2 * (y1 * y2 * sin(2 * tau) / 2 + y2 * y3 * sin(tau) ** 2) + eps * (
                                 -y1 * sin(2 * tau) / 2 + y3 * cos(2 * tau) / 2 - y3 / 2),
                     eps ** 3 * (-y2 ** 3 * sin(tau) ** 2 / 4 - y2 * y4 * sin(2 * tau) / 8) + eps ** 2 * (
                                 y2 ** 2 * sin(tau) ** 2 / 2 - y4 * sin(2 * tau) / 4) + eps * (
                                 y2 * cos(2 * tau) / 2 - y2 / 2) + 1,
                     eps ** 3 * (-y1 * y2 * sin(tau) ** 2 / 4 - y2 * y3 * sin(2 * tau) / 8) + eps ** 2 * (
                                 -y1 * cos(2 * tau) / 4 + y1 / 4 - y3 * sin(2 * tau) / 4)],
                    [eps ** 3 * (y1 * y2 ** 2 * sin(2 * tau) / 4 + y1 * y4 * cos(
                        2 * tau) / 4 - y1 * y4 / 4) + eps ** 2 * (
                                 y1 * y2 * sin(2 * tau) / 2 - y2 * y3 * cos(2 * tau) / 2 + y2 * y3 / 2) + eps * (
                                 -y1 * sin(2 * tau) / 2 + y3 * cos(2 * tau) / 2 - y3 / 2),
                     eps ** 3 * (y1 ** 2 * y2 * sin(2 * tau) / 4 + y2 * y3 ** 2 * sin(2 * tau) / 4) + eps ** 2 * (
                                 y1 ** 2 * sin(2 * tau) / 4 - y1 * y3 * cos(2 * tau) / 2 + y1 * y3 / 2 - y3 ** 2 * sin(
                             2 * tau) / 4), eps ** 3 * (y2 ** 2 * y3 * sin(2 * tau) / 4 + y3 * y4 * cos(
                        2 * tau) / 4 - y3 * y4 / 4) + eps ** 2 * (
                                 -y1 * y2 * cos(2 * tau) / 2 + y1 * y2 / 2 - y2 * y3 * sin(2 * tau) / 2) + eps * (
                                 y1 * cos(2 * tau) / 2 - y1 / 2 + y3 * sin(2 * tau) / 2), eps ** 3 * (
                                 y1 ** 2 * cos(2 * tau) / 8 - y1 ** 2 / 8 + y3 ** 2 * cos(
                             2 * tau) / 8 - y3 ** 2 / 8) + 1]])
                return z

    def d_tau_Phi(tau , y , eps , n_approx):
        """derivative w.r.t. tau of the approximation of the mapping Phi :
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations in order to approximate mapping Phi"""
        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        y = np.array(y).reshape(d, )
        z = np.zeros_like(y)
        if dyn_syst == "Logistic":
            if n_approx == 0:
                return z
            if n_approx == 1:
                z = eps * np.array([np.sin(tau)])
                return z
            if n_approx == 2:
                z = eps*np.array([np.sin(tau)]) + eps ** 2 * (2 * y - 1) * np.cos(tau) + eps ** 3 * np.array([2 * np.cos(tau) - np.cos(2 * tau) / 2])
                return z
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            if n_approx == 0:
                return z
            if n_approx == 1:
                #return np.array([eps*(y1*y2*np.sin(2*tau) - y2*y3*np.cos(2*tau)),0,eps*(-y1*y2*np.cos(2*tau) - y2*y3*np.sin(2*tau)),eps*(-y1**2*np.cos(2*tau) - 2*y1*y3*np.sin(2*tau) + y3**2*np.cos(2*tau))])
                z = np.array([
                [                        eps*(y1*y2*sin(2*tau) - y2*y3*cos(2*tau))],
                [                                                                0],
                [                       eps*(-y1*y2*cos(2*tau) - y2*y3*sin(2*tau))],
                [eps*(-y1**2*cos(2*tau)/2 - y1*y3*sin(2*tau) + y3**2*cos(2*tau)/2)]]).reshape(d,)
                return z
            if n_approx == 2:
                # return np.array([eps**3*(-y1*y2**3*np.sin(2*tau)/4 - y1*y2*y4*np.cos(2*tau)/4 - y2**3*y3*np.cos(2*tau)/4 + y2*y3*y4*np.sin(2*tau)/4) + eps**2*(-y1*y2**2*np.sin(2*tau)/2 + y1*y4*np.cos(2*tau)/2 + y2**2*y3*np.cos(2*tau)/2 + y3*y4*np.sin(2*tau)/2) + eps*(y1*y2*np.sin(2*tau) - y2*y3*np.cos(2*tau)),0,
                # eps**3*(y1*y2**3*np.cos(2*tau)/4 - y1*y2*y4*np.sin(tau)*np.cos(tau)/2 - y2**3*y3*np.sin(tau)*np.cos(tau)/2 - y2*y3*y4*np.cos(2*tau)/4) + eps**2*(y1*y2**2*np.cos(2*tau)/2 + y1*y4*np.sin(2*tau)/2 + y2**2*y3*np.sin(tau)*np.cos(tau) - y3*y4*np.cos(2*tau)/2) + eps*(-y1*y2*np.cos(2*tau) - y2*y3*np.sin(2*tau)),
                # eps**3*(y1**2*y2**2*np.cos(2*tau)/2 - y1**2*y4*np.sin(2*tau)/2 + y2**2*y3**2*np.cos(2*tau)/2 - y3**2*y4*np.sin(2*tau)/2) + eps**2*(y1**2*y2*np.cos(2*tau) + 2*y1*y2*y3*np.sin(2*tau) - y2*y3**2*np.cos(2*tau)) + eps*(-y1**2*np.cos(2*tau) - 2*y1*y3*np.sin(2*tau) + y3**2*np.cos(2*tau))])

                z = np.array([
                    [eps ** 3 * (
                                -y1 * y2 ** 3 * sin(2 * tau) / 4 - y1 * y2 * y4 * cos(2 * tau) / 4 - y2 ** 3 * y3 * cos(
                            2 * tau) / 4 + y2 * y3 * y4 * sin(2 * tau) / 4) + eps ** 2 * (
                                 -y1 * y2 ** 2 * sin(2 * tau) / 2 + y1 * y4 * cos(2 * tau) / 2 + y2 ** 2 * y3 * cos(
                             2 * tau) / 2 + y3 * y4 * sin(2 * tau) / 2) + eps * (
                                 y1 * y2 * sin(2 * tau) - y2 * y3 * cos(2 * tau))],
                    [0],
                    [eps ** 3 * (y1 * y2 ** 3 * cos(2 * tau) / 4 - y1 * y2 * y4 * sin(tau) * cos(
                        tau) / 2 - y2 ** 3 * y3 * sin(tau) * cos(tau) / 2 - y2 * y3 * y4 * cos(
                        2 * tau) / 4) + eps ** 2 * (
                                 y1 * y2 ** 2 * cos(2 * tau) / 2 + y1 * y4 * sin(2 * tau) / 2 + y2 ** 2 * y3 * sin(
                             tau) * cos(tau) - y3 * y4 * cos(2 * tau) / 2) + eps * (
                                 -y1 * y2 * cos(2 * tau) - y2 * y3 * sin(2 * tau))],
                    [eps ** 3 * (y1 ** 2 * y2 ** 2 * cos(2 * tau) / 4 - y1 ** 2 * y4 * sin(
                        2 * tau) / 4 + y2 ** 2 * y3 ** 2 * cos(2 * tau) / 4 - y3 ** 2 * y4 * sin(
                        2 * tau) / 4) + eps ** 2 * (
                                 y1 ** 2 * y2 * cos(2 * tau) / 2 + y1 * y2 * y3 * sin(2 * tau) - y2 * y3 ** 2 * cos(
                             2 * tau) / 2) + eps * (
                                 -y1 ** 2 * cos(2 * tau) / 2 - y1 * y3 * sin(2 * tau) + y3 ** 2 * cos(2 * tau) / 2)]]).reshape(d,)
                return z


    def Phi_MP(tau , y ,eps):
        """Approximation of the mapping Phi for MidPoint method:
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations in order to approximate mapping Phi"""
        y = np.array(y).reshape(d,)
        y_i = y
        for i in range(N_iter):
            y_i = y + eps*ODE.h_MP(tau , (y_i + y)/2 , eps)
            #print(i,y_i)
        return y_i

    def h_MP(tau , y ,eps):
        """Second order variable change for the MidPoint method.
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        if dyn_syst == "Logistic":
            return np.array([1-np.cos(tau)]) - (1-2*y)*np.sin(tau)
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            # return np.array([eps * y1 * y2 ** 2 * np.cos(2 * tau) / 2 - eps * y1 * y2 ** 2 / 2 + eps * y1 * y4 * np.sin(2 * tau) / 4 + 3 * eps * y2 ** 2 * y3 * np.sin(2 * tau) / 4 - eps * y3 * y4 * np.cos(2 * tau) / 4 + eps * y3 * y4 / 4 - y1 * y2 * np.cos(2 * tau) / 2 + y1 * y2 / 2 - y2 * y3 * np.sin(
            #         2 * tau) / 2 , eps * (y1 ** 2 * np.cos(2 * tau) / 2 - y1 ** 2 / 2 + y1 * y3 * np.sin(2 * tau) - y3 ** 2 * np.cos(
            #         2 * tau) / 2 + y3 ** 2 / 2) / 2 , eps * y1 * y2 ** 2 * np.sin(2 * tau) / 4 - eps * y1 * y4 * np.cos(
            #         2 * tau) / 4 + eps * y1 * y4 / 4 - eps * y2 ** 2 * y3 * np.cos(
            #         2 * tau) / 2 + eps * y2 ** 2 * y3 / 2 - eps * y3 * y4 * np.sin(2 * tau) / 4 - y1 * y2 * np.sin(
            #         2 * tau) / 2 + y2 * y3 * np.cos(2 * tau) / 2 - y2 * y3 / 2 , eps * y1 ** 2 * y2 * np.sin(2 * tau) / 2 - 2 * eps * y1 * y2 * y3 * np.cos(
            #         2 * tau) + 2 * eps * y1 * y2 * y3 - 3 * eps * y2 * y3 ** 2 * np.sin(2 * tau) / 2 - y1 ** 2 * np.sin(
            #         2 * tau) / 2 + y1 * y3 * np.cos(2 * tau) - y1 * y3 + y3 ** 2 * np.sin(2 * tau) / 2])
            z = np.array([
            [eps*y1*y2**2*cos(2*tau)/2 - eps*y1*y2**2/2 + eps*y1*y4*sin(2*tau)/4 + 3*eps*y2**2*y3*sin(2*tau)/4 - eps*y3*y4*cos(2*tau)/4 + eps*y3*y4/4 - y1*y2*cos(2*tau)/2 + y1*y2/2 - y2*y3*sin(2*tau)/2],
            [                                                                                                      eps*(y1**2*cos(2*tau)/2 - y1**2/2 + y1*y3*sin(2*tau) - y3**2*cos(2*tau)/2 + y3**2/2)/4],
            [  eps*y1*y2**2*sin(2*tau)/4 - eps*y1*y4*cos(2*tau)/4 + eps*y1*y4/4 - eps*y2**2*y3*cos(2*tau)/2 + eps*y2**2*y3/2 - eps*y3*y4*sin(2*tau)/4 - y1*y2*sin(2*tau)/2 + y2*y3*cos(2*tau)/2 - y2*y3/2],
            [                   eps*y1**2*y2*sin(2*tau)/4 - eps*y1*y2*y3*cos(2*tau) + eps*y1*y2*y3 - 3*eps*y2*y3**2*sin(2*tau)/4 - y1**2*sin(2*tau)/4 + y1*y3*cos(2*tau)/2 - y1*y3/2 + y3**2*sin(2*tau)/4]]).reshape(d,)
            return z

    def d_y_h_MP(tau , y , eps):
        """Derivative w.r.t. y of the second order variable change for the MidPoint method.
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        if dyn_syst == "Logistic":
            return np.array([2*np.sin(tau)])
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            # return np.array([[eps*y2**2*np.cos(2*tau)/2 - eps*y2**2/2 + eps*y4*np.sin(2*tau)/4 - y2*np.cos(2*tau)/2 + y2/2, eps*y1*y2*np.cos(2*tau) - eps*y1*y2 + 3*eps*y2*y3*np.sin(2*tau)/2 - y1*np.cos(2*tau)/2 + y1/2 - y3*np.sin(2*tau)/2,                         3*eps*y2**2*np.sin(2*tau)/4 - eps*y4*np.cos(2*tau)/4 + eps*y4/4 - y2*np.sin(2*tau)/2,  eps*y1*np.sin(2*tau)/4 - eps*y3*np.cos(2*tau)/4 + eps*y3/4],
            #         [eps*(y1*np.cos(2*tau) - y1 + y3*np.sin(2*tau))/2, 0,                                                          eps*(y1*np.sin(2*tau) - y3*np.cos(2*tau) + y3)/2,                                                     0],
            #         [eps*y2**2*np.sin(2*tau)/4 - eps*y4*np.cos(2*tau)/4 + eps*y4/4 - y2*np.sin(2*tau)/2,   eps*y1*y2*np.sin(2*tau)/2 - eps*y2*y3*np.cos(2*tau) + eps*y2*y3 - y1*np.sin(2*tau)/2 + y3*np.cos(2*tau)/2 - y3/2,                -eps*y2**2*np.cos(2*tau)/2 + eps*y2**2/2 - eps*y4*np.sin(2*tau)/4 + y2*np.cos(2*tau)/2 - y2/2, -eps*y1*np.cos(2*tau)/4 + eps*y1/4 - eps*y3*np.sin(2*tau)/4],
            #         [eps*y1*y2*np.sin(2*tau) - 2*eps*y2*y3*np.cos(2*tau) + 2*eps*y2*y3 - y1*np.sin(2*tau) + y3*np.cos(2*tau) - y3,               eps*y1**2*np.sin(2*tau)/2 - 2*eps*y1*y3*np.cos(2*tau) + 2*eps*y1*y3 - 3*eps*y3**2*np.sin(2*tau)/2, -2*eps*y1*y2*np.cos(2*tau) + 2*eps*y1*y2 - 3*eps*y2*y3*np.sin(2*tau) + y1*np.cos(2*tau) - y1 + y3*np.sin(2*tau),                                                     0]])
            z = np.array([
            [                 eps*y2**2*cos(2*tau)/2 - eps*y2**2/2 + eps*y4*sin(2*tau)/4 - y2*cos(2*tau)/2 + y2/2, eps*y1*y2*cos(2*tau) - eps*y1*y2 + 3*eps*y2*y3*sin(2*tau)/2 - y1*cos(2*tau)/2 + y1/2 - y3*sin(2*tau)/2,                             3*eps*y2**2*sin(2*tau)/4 - eps*y4*cos(2*tau)/4 + eps*y4/4 - y2*sin(2*tau)/2,  eps*y1*sin(2*tau)/4 - eps*y3*cos(2*tau)/4 + eps*y3/4],
            [                                                          eps*(y1*cos(2*tau) - y1 + y3*sin(2*tau))/4,                                                                                                      0,                                                              eps*(y1*sin(2*tau) - y3*cos(2*tau) + y3)/4,                                                     0],
            [                           eps*y2**2*sin(2*tau)/4 - eps*y4*cos(2*tau)/4 + eps*y4/4 - y2*sin(2*tau)/2,   eps*y1*y2*sin(2*tau)/2 - eps*y2*y3*cos(2*tau) + eps*y2*y3 - y1*sin(2*tau)/2 + y3*cos(2*tau)/2 - y3/2,                    -eps*y2**2*cos(2*tau)/2 + eps*y2**2/2 - eps*y4*sin(2*tau)/4 + y2*cos(2*tau)/2 - y2/2, -eps*y1*cos(2*tau)/4 + eps*y1/4 - eps*y3*sin(2*tau)/4],
            [eps*y1*y2*sin(2*tau)/2 - eps*y2*y3*cos(2*tau) + eps*y2*y3 - y1*sin(2*tau)/2 + y3*cos(2*tau)/2 - y3/2,                   eps*y1**2*sin(2*tau)/4 - eps*y1*y3*cos(2*tau) + eps*y1*y3 - 3*eps*y3**2*sin(2*tau)/4, -eps*y1*y2*cos(2*tau) + eps*y1*y2 - 3*eps*y2*y3*sin(2*tau)/2 + y1*cos(2*tau)/2 - y1/2 + y3*sin(2*tau)/2,                                                     0]])
            return z

    def d_tau_h_MP(tau , y , eps):
        """Derivative w.r.t. tau of the second order variable change for the MidPoint method.
        Inputs:
        - tau: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        def cos(tau):
            return np.cos(tau)
        def sin(tau):
            return np.sin(tau)
        if dyn_syst == "Logistic":
            return np.array([np.sin(tau)]) - (1-2*y)*np.cos(tau)
        if dyn_syst == "Hénon-Heiles":
            y1, y2, y3, y4 = y[0], y[1], y[2], y[3]
            # return np.array([
            #         [-eps*y1*y2**2*np.sin(2*tau) + eps*y1*y4*np.cos(2*tau)/2 + 3*eps*y2**2*y3*np.cos(2*tau)/2 + eps*y3*y4*np.sin(2*tau)/2 + y1*y2*np.sin(2*tau) - y2*y3*np.cos(2*tau)],
            #         [                                                                             eps*(-y1**2*np.sin(2*tau) + 2*y1*y3*np.cos(2*tau) + y3**2*np.sin(2*tau))/2],
            #         [   eps*y1*y2**2*np.cos(2*tau)/2 + eps*y1*y4*np.sin(2*tau)/2 + eps*y2**2*y3*np.sin(2*tau) - eps*y3*y4*np.cos(2*tau)/2 - y1*y2*np.cos(2*tau) - y2*y3*np.sin(2*tau)],
            #         [    eps*y1**2*y2*np.cos(2*tau) + 4*eps*y1*y2*y3*np.sin(2*tau) - 3*eps*y2*y3**2*np.cos(2*tau) - y1**2*np.cos(2*tau) - 2*y1*y3*np.sin(2*tau) + y3**2*np.cos(2*tau)]]).reshape(d,)
            z = np.array([
            [  -eps*y1*y2**2*sin(2*tau) + eps*y1*y4*cos(2*tau)/2 + 3*eps*y2**2*y3*cos(2*tau)/2 + eps*y3*y4*sin(2*tau)/2 + y1*y2*sin(2*tau) - y2*y3*cos(2*tau)],
            [                                                                               eps*(-y1**2*sin(2*tau) + 2*y1*y3*cos(2*tau) + y3**2*sin(2*tau))/4],
            [     eps*y1*y2**2*cos(2*tau)/2 + eps*y1*y4*sin(2*tau)/2 + eps*y2**2*y3*sin(2*tau) - eps*y3*y4*cos(2*tau)/2 - y1*y2*cos(2*tau) - y2*y3*sin(2*tau)],
            [eps*y1**2*y2*cos(2*tau)/2 + 2*eps*y1*y2*y3*sin(2*tau) - 3*eps*y2*y3**2*cos(2*tau)/2 - y1**2*cos(2*tau)/2 - y1*y3*sin(2*tau) + y3**2*cos(2*tau)/2]]).reshape(d,)
            return z


    def Pull_back(t , y , eps , n_approx):
        """Field for Pull-back method for non-stiff integrator:
        Inputs:
        - t: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iterations of the approximation of the mapping Phi"""
        y = np.array(y).reshape(d,)
        z = np.zeros_like(y)
        z = np.linalg.inv(ODE.d_y_Phi(t/eps , y , eps , n_approx)) @ (ODE.f(t/eps , ODE.Phi(t/eps , y , eps , n_approx)) - (1/eps)*ODE.d_tau_Phi(t/eps , y , eps , n_approx))
        return z

    def Pull_back_MP(t , y , eps ):
        """Field for Pull-back method for non-stiff integrator, specified to MidPoint method:
        Inputs:
        - t: Float - Time
        - y: Array of shape (1,) - Space variable
        - eps: Float - High oscillation parameter"""
        y = np.array(y).reshape(d, )
        P = y
        S,Q = np.zeros_like(y) , np.zeros_like(y)
        #M = np.zeros((d,d))
        for i in range(N_iter):
            U = (y+P)/2
            P = y + eps*ODE.h_MP(t/eps , U , eps)
            Q = ODE.d_tau_h_MP(t/eps , U , eps) + (eps/2)*ODE.d_y_h_MP(t/eps , U , eps)@Q
            K = ODE.f(t/eps , P) - Q
            S = K - (eps/2)*ODE.d_y_h_MP(t/eps , U , eps)@(S + K)
            #print(i,S)
            #M = np.eye(d,d) - (eps/2)*ODE.d_tau_h_MP(t/eps , U , eps)@(np.eye(d,d) + M)
            #S = np.linalg.inv(M)@K
        return S


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
            return ODE.dyn(t , y , eps)


        Y = solve_ivp(fun = func , t_span=(0,T) , y0 = Y0 , t_eval=np.arange(0,T,h) , atol = 1e-13 , rtol = 1e-13 , method = "DOP853").y
        return Y

    def Exact_solve_bis(T , h , eps , n_approx = 1):
        """Exact resolution of the ODE by using a very accurate Python Integrator DOP853 for PullBack equation and mapping Phi
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - n_approx: Int - Iteration for approximation of averaged field and mapping Phi. Default: 1"""

        def func(t , y):
            """Function for exact solving of the ODE:
            Inputs:
            - t: Float - Time
            - y: Array of shape (1,) - Space variable"""
            return ODE.Pull_back(t , y , eps , n_approx)

        Y = solve_ivp(fun=func, t_span=(0, T), y0=Y0, t_eval=np.arange(0, T, h), atol=1e-13, rtol=1e-13, method="DOP853").y

        TT = np.arange(0, T, h)
        for n in range(np.size(TT)):
                Y[: , n] = ODE.Phi(TT[n]/eps , Y[:,n] , eps , n_approx)
        return Y

    def Num_solve(T , h , eps , num_meth = "Forward Euler" , n_approx = 1 , time_print = False):
        """Numerical resolution of the ODE - Pullback method.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - n_approx: Int - Iteration for approximation of averaged field and mapping Phi. Default: 1
        - time_print: Boolean - Print pr not the time for numerical approximation of the solution. Default: False"""

        TT = np.arange(0,T,h)
        YY = np.zeros((d,np.size(TT)))
        YY[:,0] = Y0

        for n in range(np.size(TT)-1):
            if time_print == True:
                print(" t =  {}  \r".format(str(format(TT[n], '.4E'))).rjust(3),  end=" ")
            def F_iter(y, num_meth):
                """Function which solves the equation for each iteration of an implicit method by Fixed Point iterations.
                Inputs:
                - y: Array of shape (2, ) - Space variable
                - num_meth: Str - Name of the implicit method"""
                if num_meth == "MidPoint":
                    return YY[:, n] + h * ODE.Pull_back_MP(TT[n] + h/2, (YY[:, n] + y)/2 , eps)

            if num_meth == "Forward Euler":
                YY[:, n + 1] = YY[:, n] + h * ODE.Pull_back(TT[n] , YY[:, n] , eps , n_approx)
            if num_meth == "RK2":
                YY[:, n + 1] = YY[:, n] + h * ODE.Pull_back(TT[n] + h/2, YY[:, n] + (h/2) * ODE.Pull_back(TT[n] , YY[:, n] , eps , n_approx), eps, n_approx)
            if num_meth == "MidPoint":
                y_i = YY[:, n]
                for i in range(N_iter):
                    y_i = F_iter(y_i, "MidPoint")
                    #print(i,y_i)
                YY[:, n + 1] = y_i


        for n in range(np.size(TT)):
            if num_meth == "MidPoint":
                YY[: , n] = ODE.Phi_MP(TT[n]/eps, YY[:, n], eps)
                #YY[: , n] = YY[: , n]
            else:
                YY[: , n] = ODE.Phi(TT[n]/eps , YY[:,n] , eps , n_approx)

        return YY

    def Plot_Solve(T , h , eps , num_meth = "Forward Euler" , n_approx = 1 , save = False):
        """Numerical resolution of the ODE vs exact solution ploted.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - n_approx: Int- Iteration for approximation of averaged field and mapping Phi. Default: 1
        - save: Boolean - Saves the figure or not. Default: False"""
        TT = np.arange(0,T,h)
        TT = TT.reshape(1,np.size(TT))
        Y_Exact , Y_Num  = ODE.Exact_solve(T,h,eps) , ODE.Num_solve(T,h,eps,num_meth,n_approx,time_print=True)
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
                plt.savefig("Integration_Pullback_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()

        if dyn_syst == "Hénon-Heiles":

            Y_Exact_VC , Y_Num_VC = np.zeros_like(Y_Exact) , np.zeros_like(Y_Num)
            for n in range(np.size(TT)):
                #VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                Y_Exact_VC[:,n] , Y_Num_VC[:,n] = VC@Y_Exact[:,n] , VC@Y_Num[:,n]

            Ham = ODE.H(Y_Num_VC , eps)
            Ham_0 = ODE.H(Y0.reshape(d,1)@np.ones_like(TT) , eps)

            plt.figure(figsize = (12,5))
            # plt.subplot(1,3,1)
            # plt.scatter(Y_Num_VC[0,:], Y_Num_VC[2,:], s = 5 , label="Num solution", color="green")
            # plt.plot(np.squeeze(Y_Exact_VC[0,:]), np.squeeze(Y_Exact_VC[2,:]), label="Exact solution", color="red")
            # plt.grid()
            # plt.legend()
            # plt.xlabel("$q_1$")
            # plt.ylabel("$p_1$")
            # plt.title("Exact solution vs Numerical solution - ")
            # plt.axis("equal")
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
                plt.savefig("Integration_Pullback_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
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
        YY_app = ODE.Num_solve(T , h , eps , num_meth)

        norm_exact = np.max(np.linalg.norm(YY_exact , 2 , axis = 0))
        norm_error = np.max(np.linalg.norm(YY_exact - YY_app, 2 , axis = 0))

        error = norm_error/norm_exact

        return error

    def Curve(T , num_meth = "Forward Euler" , n_approx = 1 ,save = False):
        """Plots a curve of convergence w.r.t. various numerical methods
        Inputs:
        - T: Float - Time for ODE simulations
        - num_meth: Str - Name of the numerical method. Default: "Forward Euler"
        - n_approx: Int- Iteration for approximation of averaged field and mapping Phi. Default: 1
        - save: Boolean - Saves the figure or not. Default: False"""
        Num_Meths = [num_meth]
        cmap = plt.get_cmap("jet")
        Colors = [cmap(k/10) for k in range(10)]
        HH = np.exp(np.linspace(np.log(step_h[0]),np.log(step_h[1]),11))
        EPS = np.exp(np.linspace(np.log(step_eps[0]),np.log(step_eps[1]),10))
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
                plt.loglog(HH, E[i,:,k], "s" , color=Colors[k] , label = "$\epsilon = $"+str(format(EPS[k],'.2E')) , markersize = 5)
        plt.legend()
        plt.title("Integration errors - "+num_meth+" - "+dyn_syst)
        plt.xlabel("h")
        plt.ylabel("Rel. Error")
        plt.grid()
        if save == True:
            plt.savefig("Convergence_Curve_Pullback_"+num_meth+"_"+dyn_syst+"_T="+str(T)+"_h="+str(step_h)+"_epsilon="+str(step_eps)+".pdf")
        plt.show()


        pass
