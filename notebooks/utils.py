import xarray as xr
import numpy as np
import netCDF4 as nc
import pandas as pd
import pickle
import random
import xskillscore as xs
import cartopy.crs as ccrs
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib as plt
    
def standard_plot(ds):
    """Plots a map where different columns correspond to different lead_times
    
    Args:
        ds: xarray data_array with one data var and coordinates: latitude, longitude and lead_time.
    
    Returns:
       Plots a global map in EqualEarth projection (area-preserving), centered in Greenwich and with a fixed color scale.
    """
    ds.plot(size=9, aspect=2,robust=True, col='lead_time', vmax=1, levels=np.arange(-0.15,0.20,0.05),transform=ccrs.PlateCarree(), subplot_kws={'projection': ccrs.EqualEarth()})

# Aaron's code slightly modified in order to work with already year-week split datasets
def make_probabilistic(ds, tercile_edges, member_dim='realization', mask=None):
    """Compute probabilities from ds (observations or forecasts) based on tercile_edges.
    
    Args:
        ds: xarray data_array with observations, hindcast or forecasts.
        tercile_edges: xarray with tercile edges
        member_dim: name of the dimension where you have the different members
        mask: condition to mask some values
        
    Returns:
        Dataset with probabilites of each category based on tercile edges.  
    """
    # broadcast
    if 'forecast_time' not in tercile_edges.dims and 'weekofyear' in tercile_edges.dims:
        tercile_edges = tercile_edges.sel(weekofyear=ds.forecast_time.dt.weekofyear)
    bn = ds < tercile_edges.isel(category_edge=0, drop=True)  # below normal
    n = (ds >= tercile_edges.isel(category_edge=0, drop=True)) & (ds < tercile_edges.isel(category_edge=1, drop=True))  # normal
    an = ds >= tercile_edges.isel(category_edge=1, drop=True)  # above normal
    if member_dim in ds.dims:
        bn = bn.mean(member_dim)
        an = an.mean(member_dim)
        n = n.mean(member_dim)
    ds_p = xr.concat([bn, n, an],'category').assign_coords(category=['below normal', 'near normal', 'above normal'])
    if mask is not None:
        ds_p = ds_p.where(mask)
    if 'tp' in ds_p.data_vars:
        # mask arid grid cells where category_edge are too close to 0
        # we are using a dry mask as in https://doi.org/10.1175/MWR-D-17-0092.1
        tp_arid_mask = tercile_edges.tp.isel(category_edge=0, lead_time=0, drop=True) > 0.01
        ds_p['tp'] = ds_p['tp'].where(tp_arid_mask)
    ds_p['category'].attrs = {'long_name': 'tercile category probabilities', 'units': '1',
                        'description': 'Probabilities for three tercile categories. All three tercile category probabilities must add up to 1.'}
    ds_p['tp'].attrs = {'long_name': 'Probability of total precipitation in tercile categories', 'units': '1',
                      'comment': 'All three tercile category probabilities must add up to 1.',
                      'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Total+Precipitation'
                     }
    ds_p['t2m'].attrs = {'long_name': 'Probability of 2m temperature in tercile categories', 'units': '1',
                      'comment': 'All three tercile category probabilities must add up to 1.',
                      'variable_before_categorization': 'https://confluence.ecmwf.int/display/S2S/S2S+Surface+Air+Temperature'
                      }
    if 'weekofyear' in ds_p.coords:
        ds_p = ds_p.drop('weekofyear')
    return ds_p

def year_week_split(ds):
    """Reshape a dataset/data_array with 1060 forecast_time (start_date) into 53 weeks and 20 years
    
    Args:
        ds: xarray data_array or dataset with observations, hindcast or forecasts
        
    Returns:
        Dataset with year and week dimensions  
    """
    y = list(range(2000,2019+1))
    w = list(range(53))
    ds = ds.assign_coords(
        year=y, week=w
    ).stack(
          dim=("year", "week")
    ).reset_index(
       "forecast_time", drop=True
    ).rename(
        forecast_time="dim"
    ).unstack("dim")
    return ds

def create_predictions_dataset(pred1, pred2 = None):
    """Transform data_array predictions into a dataset with all coordinates/dimensions needed in the challenge
    
    Args:
        pred1: data array with t2m predictions
        pred2: data array with tp predictions
        
    Returns:
        Dataset with all vars and coordinates/dimensions needed in the challenge
    """
    predictions_array_t2m= np.asarray(pred1.values)
    predictions_dataset_t2m= np.zeros((3,53,121,2,240))
    if pred2 is not None:
        predictions_array_tp= np.asarray(pred2.values)
        predictions_dataset_tp= np.zeros((3,53,121,2,240))    
    for i in range(53):
        for j in range(121):
            for k in range(2):
                for l in range(240):
                    try:
                        predictions_dataset_t2m[:,i,j,k,l] = predictions_array_t2m[i,j,k,l]
                        if pred2 is not None:
                            predictions_dataset_tp[:,i,j,k,l] = predictions_array_tp[i,j,k,l]
                    except:
                        pass
    latitude= np.array(pred1.latitude)
    longitude = np.array(pred1.longitude)
    category= ['below normal', 'near normal', 'above normal']
    lead_time = np.array(pred1.lead_time)
    forecast_time = np.array(pred1.week)
    if pred2 is not None:
        predictions_dataset = xr.Dataset(data_vars=dict(
                                    t2m=(["category","forecast_time","latitude", "lead_time","longitude"], predictions_dataset_t2m),
                                    tp=(["category","forecast_time","latitude", "lead_time","longitude"], predictions_dataset_tp)),
                                         #tp is just here to pass the verification test
                     coords=dict(
                                    category=(["category"], category),
                                    forecast_time=(["forecast_time"],forecast_time),
                                    latitude=(["latitude"], latitude),
                                    lead_time =(["lead_time"], lead_time),
                                    longitude= (["longitude"], longitude),
                                   ),                   
                     attrs=dict(description="Weather related data."),
                    )
    else:
        predictions_dataset = xr.Dataset(data_vars=dict(
                                    t2m=(["category","forecast_time","latitude", "lead_time","longitude"], predictions_dataset_t2m),
                                    ),
                                         #tp is just here to pass the verification test
                     coords=dict(
                                    category=(["category"], category),
                                    forecast_time=(["forecast_time"],forecast_time),
                                    latitude=(["latitude"], latitude),
                                    lead_time =(["lead_time"], lead_time),
                                    longitude= (["longitude"], longitude),
                                   ),                   

                     attrs=dict(description="Weather related data."),
                    )
    return predictions_dataset


# Function defined in 21st october (issue #50): https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge/-/issues/50
def compute_rpss(verif, ML_model, clim):
    """Compute rpss of a model.
    
    Args:
        verif: observations/ground truth in categories
        ML_model: predictions
        clim: climatology model. 1/3 probabilities for all classes
        
    Returns:
        Dataset with rpss for each latitude, longitude and lead_time
    """
    if 'forecast_time' in verif.coords:
        try:
            verif["forecast_time"] = ML_model["forecast_time"]
        except:
            pass
    #Compute RPS
    rps_clim = xs.rps(verif, clim, category_edges=None, dim=[], input_distributions='p').compute()  
    rps_ML = xs.rps(verif, ML_model, category_edges=None, dim=[], input_distributions='p').compute()
    # penalize # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/7
    expect = verif.sum('category')
    expect = expect.where(expect > 0.98).where(expect < 1.02)  # should be True if not all NaN
    # https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/issues/50
    rps_ML = rps_ML.where(expect, other=2)  # assign RPS=2 where value was expected but NaN found
    # following Weigel 2007: https://doi.org/10.1175/MWR3280.1
    rpss = 1 - (rps_ML.mean('forecast_time') / rps_clim.mean('forecast_time'))
    # clip
    rpss = rpss.clip(-10, 1)
    return rpss

def compute_cv_rpss(verification, ML_model, clim_p):
    """Compute rpss for each year.
    
    Args:
        verif: observations/ground truth in categories
        ML_model: predictions
        clim_p: climatology model. 1/3 probabilities for all classes
        
    Returns:
        Dataset with rpss for each latitude, longitude, lead_time and year.
    """
    rpss = []
    for i in range(20):
        verif = verification.isel(year=i)
        if 'week' in verif.coords:
            verif = verif.rename({'week':'forecast_time'})
        try:
            rpss_year = compute_rpss(verif, ML_model.isel(year=i), clim_p)
        except: #Case to compute climatology rpss
            rpss_year = compute_rpss(verif, clim_p, clim_p)
        rpss.append(rpss_year)
    rpss_concat = xr.concat(rpss,'year')    
    return rpss_concat
