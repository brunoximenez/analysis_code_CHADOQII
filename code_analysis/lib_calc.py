import numpy as np 
import random
import matplotlib.pyplot as plt
	
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})

def sim_vary_gate(T, t_init, t_end, steps, u0):
    # constants
    pi = np.pi
    kb = 1.38e-23
    m = 1.45e-25

    # tweezers parameters
    lamb = 850e-9
    w = 1.05e-6
    zr = pi*w**2/lamb
    u0 *= kb * 1e-3
    om_perp=np.sqrt(4*u0/m/w**2)
    om_z=np.sqrt(2*u0/m/zr**2) 

    # atomic distribution parameters
    recap_list = np.array([])
    natoms = 10000

    free_flight_time_list = np.linspace(t_init, t_end, steps)
    for tau in free_flight_time_list:
        E_m = kb * T / m
        sigma_v = np.sqrt(E_m)
        sigma_perp = np.sqrt(E_m / om_perp ** 2)
        sigma_z = np.sqrt(E_m / om_z ** 2)

        # simulation params
        nrecaps = 0

        
        for atom in range(natoms):
            v0 = np.array([random.normalvariate(0,sigma_v), random.normalvariate(0,sigma_v), random.normalvariate(0,sigma_v)])
            r0 = np.array([random.normalvariate(0,sigma_perp), random.normalvariate(0,sigma_perp), random.normalvariate(0,sigma_z)])

            Ek=m/2.*(np.sum(v0**2))

            rf = r0 + v0 * tau

            u=-u0/(1+rf[2]**2/zr**2)
            u=u*np.exp(-2*(rf[0]**2+rf[1]**2)/w**2)
            if Ek<-u:
                nrecaps+=1

        recap_list = np.append(recap_list, nrecaps / natoms)


    return free_flight_time_list, recap_list


def sim_vary_temp(free_flight_time, T_init, T_end, steps, u0):
	# constants
    pi = np.pi
    kb = 1.38e-23
    m = 1.45e-25

    # tweezers parameters
    lamb = 850e-9
    w = 1.05e-6
    zr = pi*w**2/lamb
    u0 *= kb * 1e-3
    om_perp=np.sqrt(4*u0/m/w**2)
    om_z=np.sqrt(2*u0/m/zr**2) 

    # atomic distribution parameters
    recap_list = np.array([])
    natoms = 10000

    T_list = np.linspace(T_init, T_end, steps)
    for T in T_list:
        E_m = kb * T / m
        sigma_v = np.sqrt(E_m)
        sigma_perp = np.sqrt(E_m / om_perp ** 2)
        sigma_z = np.sqrt(E_m / om_z ** 2)

        # simulation params
        nrecaps = 0

        
        for atom in range(natoms):
            v0 = np.array([random.normalvariate(0,sigma_v), random.normalvariate(0,sigma_v), random.normalvariate(0,sigma_v)])
            r0 = np.array([random.normalvariate(0,sigma_perp), random.normalvariate(0,sigma_perp), random.normalvariate(0,sigma_z)])

            Ek=m/2.*(np.sum(v0**2))

            rf = r0 + v0 * free_flight_time

            u=-u0/(1+rf[2]**2/zr**2)
            u=u*np.exp(-2*(rf[0]**2+rf[1]**2)/w**2)
            if Ek<-u:
                nrecaps+=1

        recap_list = np.append(recap_list, nrecaps / natoms)


    return T_list, recap_list


def plot(x, y, parameter, vary_time, plot_title='', xlabel='x'):
	# # TeX settings
	# plt.rcParams['text.usetex'] = True
	# plt.rcParams['text.latex.preamble'] = [
	#     r'\usepackage{amsmath}']  # for \text command
	fig = plt.figure(plot_title)

	if vary_time == True:
		# plt.title("%d uK"%int(parameter * 1e6))
		label = 'T: ' + str(parameter * 1e6) + ' uK'
	else:
		# plt.title("%d us"%int(parameter * 1e6))
		label = 'Gate time: ' + str(parameter * 1e6) + ' us'

	plt.plot(
	    x * 1e6,
	    y,
	    # marker = '',
	    # markeredgewidth = 2,
	    # markersize = 8,
	    # markerfacecolor = color_face_list[0],
	    # markeredgecolor = color_edge_list[0],
	    linestyle = '-',
	    linewidth = 2,
	    # color = color_edge_list[0],
	    label = label
	    )
	    
	#ax.plot(xnew,ynew,'k-')
	plt.ylim(0,1)
	plt.xlim(0, np.amax(x)*1e6)
	# ax.set_title("%d us"%(free_flight_time*1e6))
	plt.xlabel(xlabel)
	plt.ylabel('Recapture probability')
	plt.legend()
	#plt.savefig('free_flight_10us.pdf')
