from lmfit import Parameters, Minimizer


# ## Physics functions

# In[25]:


def fit_rabi_spatial(user_function, x, y, f0=10):
    # we define the parameters for the fit
    # require : name (ex : 'y0')
    # optionnal : value, min, max
    # it helps to give good initial value to ensure that the fit succeeds
    params = Parameters()
    # params.add('y0', value = 0., min = 0, max = 1)
    params.add('B', value=50, min=-1, max=100)
    params.add('x0', value=f0, min=f0-10, max=f0+10)
    # params.add('x0', value = f0, min = 1080, max = 1090.)

    params.add('w', value=15., min=0, max=140.)

    # we define the minimize
    mini = Minimizer(
        userfcn=user_function,  # fitting function
        params=params,
        fcn_args=(x,),  # positional arguments for the fitting function
        fcn_kws={'data': y, },  # keywords arguments for the fitting function
    )

    # FIT
    result = mini.minimize(method='leastsq')

    return result


def func_rabi_spatial(params, x, data=None):

    parvals = params.valuesdict()
    B = parvals['B']
    w = parvals['w']
    x0 = parvals['x0']

    model = B*np.exp(-((x-x0)/w)**2)

    # return the model
    if data is None:
        return model

    # return the residuals
    return model-data


def fit_autler_townes(user_function, x, y, f01, f02):
    # we define the parameters for the fit
    # require : name (ex : 'y0')
    # optionnal : value, min, max
    # it helps to give good initial value to ensure that the fit succeeds
    params = Parameters()
    # params.add('y0', value = 0., min = 0, max = 1)
    params.add('A', value=1, min=0)
    params.add('B1', value=-0.3, min=-1.0, max=0.0)
    params.add('x01', value=f01, min=f01-2, max=f01+2)
    # params.add('x0', value = f0, min = 1080, max = 1090.)

    params.add('w1', value=0.5, min=0, max=2)

    # we define the minimize
    mini = Minimizer(
        userfcn=user_function,  # fitting function
        params=params,
        fcn_args=(x,),  # positional arguments for the fitting function
        fcn_kws={'data': y, },  # keywords arguments for the fitting function
    )

    # FIT
    result = mini.minimize(method='leastsq')

    return result


def func_autler_townes(params, x, data=None):

    parvals = params.valuesdict()
    A = parvals['A']
    B1 = parvals['B1']
    w1 = parvals['w1']
    x01 = parvals['x01']

    model = A + B1*np.exp(-1./2*((x-x01)/w1)**2)

    # return the model
    if data is None:
        return model

    # return the residuals
