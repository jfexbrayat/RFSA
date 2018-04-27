"""
27/04/2018 - JFE
This script contains a function that performs the Random Forest based
sensitivity analysis similar to the one in Lopez-Blanco et al. (2017) 
and Moore et al. (2018). 

Lopez-Blanco E, Lund M, Williams M, Tamstorf MP, Westergaard-Nielsen A, 
Exbrayat J-F, Hansen BU, Christensen TR (2017) 
Exchange of CO2 in Arctic tundra: impacts of meteorological variations 
and biological disturbance. Biogeosciences 14:4467-4483. 
doi: 10.5194/bg-14-4467-2017

Moore CE, Beringer J, Donohue RJ, Evans B, Exbrayat J-F, Hutley LB, 
Tapper NJ (2018) 
Seasonal, inter-annual and decadal drivers of tree and grass productivity 
in an Australian tropical savanna. Global Change Biology early view. 
doi: 10.1111/gcb.14072

"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pylab as pl


def RFSA(df,independent,dependent,resample='none',bins='single',n_jobs=1):
    """
    The function returns the importances of predictors to explain the target.
    Data input is:
    - df: a pandas DataFrame holding the data. Index must be a date

    - independent: a list or tuple holding the names of the independent 
        variables stored in df

    - dependent: a string holding the name of the dependent variable in df

    - resample: pandas-compatible string indicating the temporal resampling 
        of the data
        default 'none' will not assume any resampling

    - bins: string indicating the temporal scale of the analysis, 
        the code currently supports
            default 'single' will run the analysis for the whole data once
            'M' will produce a seasonality based on values for each month
            'A' will produce annual time series
        

    - climatology: a boolean indicating whether the analyses should be performed 
        to derive a climatology / seasonal cycle, or for time series

    - n_jobs: number of cores to use (default = 1)
    """

    # resample if needed
    if resample != 'none':
        df = df.resample(resample)

    #create the forest object
    forest = RandomForestRegressor(1000, bootstrap = True, oob_score = True, n_jobs = n_jobs, criterion='mse')

    # get dependent and independent variables in correct format
    # 1st case, run the RF once over the whole dataset
    if bins == 'single':
        target = np.array(df[dependent])
        predictors = np.zeros([target.size,len(independent)])
        for nn, name in enumerate(independent):
            predictors[:,nn] = df[name]

        forest.fit(predictors,target)
        imp = forest.feature_importances_
        impstd = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # 2nd case, get a monthly climatology
    elif bins == 'M':
        imp = np.zeros([12,len(independent)])
        impstd = np.zeros([12,len(independent)])
        for mm in range(1,13):
            target = np.array(df[dependent][df.index.month==mm])
            predictors = np.zeros([target.size,len(independent)])
            for nn, name in enumerate(independent):
                predictors[:,nn] = df[name][df.index.month==mm]
            forest.fit(predictors,target)
            imp[mm-1] = forest.feature_importances_
            impstd[mm-1] = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)               
        
    elif bins == 'A':            
        # find first and last year
        y0, y1 = df.index.year.min(),df.index.year.max()
        # create an array storing year numbers
        years = np.linspace(y0,y1,y1-y0+1,dtype='i')
        imp = np.zeros([years.size,len(independent)])
        impstd = np.zeros([years.size,len(independent)])
        for yy, year in enumerate(years):
            target = np.array(df[dependent][df.index.year==year])
            predictors = np.zeros([target.size,len(independent)])
            for nn, name in enumerate(independent):
                predictors[:,nn] = df[name][df.index.year==year]
            forest.fit(predictors,target)
            imp[yy-1] = forest.feature_importances_
            impstd[yy-1] = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)      

    # construct necessary arrays

    return imp, impstd

def plot_importances(ax_imp,xx,imp_mean,imp_std,varlist,colors,ylabel,**kwargs):
    '''
    This function plot the importances +/- the uncertainty on axis ax_imp
    '''

    for dd, drivname in enumerate(varlist):
        ax_imp.fill_between(xx,imp_mean[:,dd]+imp_std[:,dd],imp_mean[:,dd]-imp_std[:,dd],color=colors[dd],alpha=.35,edgecolor="None")
        ax_imp.plot(xx,imp_mean[:,dd],color=colors[dd],ls='-',lw=2,label = drivname)

    ax_imp.set_ylim(0,1);

    if ylabel != '':
        ax_imp.set_ylabel(ylabel)

    if 'legend_loc' in kwargs:
        ax_imp.legend(loc=kwargs['legend_loc'],fontsize='small')


if __name__ == "__main__":

    df = pd.read_csv('TAM05_daily.csv')
    df.index = pd.date_range('2009-01-01','2010-12-31')

    imp, impstd = RFSA(df,['LAI','tair','VPD','sw'],'GPP',resample='D',bins='M',n_jobs = 10)

    #plotting part

    mon_pcp = df.groupby(df.index.month).aggregate('sum')['ppt']/2.

    fig = pl.figure('importances');fig.clf()
 
    ax = fig.add_subplot(111)
    ax.bar(range(12),mon_pcp,facecolor='silver',width=1.,edgecolor='k')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

    aximp = ax.twinx()
    plot_importances(aximp,range(12),imp,impstd,['tair','VPD','sw'],pl.cm.jet(np.linspace(0,1,imp.shape[1])),'importances',legend_loc='upper center')

    fig.show()

    

