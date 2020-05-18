import numpy as np 
import pandas as pd 
from scipy import stats
from matplotlib import pyplot as plt 
from matplotlib.ticker import MaxNLocator
from lifelines.utils import coalesce, CensoringType

"""
 Lifelines is SCiKit_Learn friendly , Built on top of Pandas
    Only focus is survival analysis
    handles left, right and interval censored data
    Estimating Hazard Rates
    Defining personal Surviving Models
    Compare two or more survival functions
         lifelines.statistics.logrank_test()

  Kaplan Meier is Linear Regression. 
  Survival regression in additional to traditional linear regression, is used to 
  explain relationship between the survival or person and 
  characteristics.         

"""
def get_distribution_name_of_lifelines_model(model):
  return model._class_name.replace("Fitter","").replace("AFT","").lower()


  
def statsdata_lifelnmodel(model):
  from lifelines.fitters import KnownModelParametricUnivariateFitter
  # Instantiate lifelines model
  first_model = isinstance(model, KnownModelParametricUnivariateFitter)
  if not (first_model):
     print("This is not univariate model. Cannot qq-plot")

  if first_model == "weibull":
    scipy_dist = "weibull_min"
    sparams = (model.rho_, 0, model.lambda_)   

  elif first_model == "lognormal":
    scipy_dist = "lognorm"
    sparams = (model.rho_, 0, np.exp(model.mu_))

  elif first_model == "loglogistic":
    scipy_dist = "fisk"
    sparams = (model.beta_, 0, model.alpha_)

  elif first_model == "exponential":
    scipy_dist = "expon"
    sparams = (0, model.lambda_)

  else:
        print("Not a Scipy Distribution")

  return getattr(stats, scipy_dist)(*sparams)          


def set_kwargs_color(kwargs):
    kwargs["color"] = coalesce(kwargs.get("c"), kwargs.get("color"), kwargs["ax"]._get_lines.get_next_color())


def set_kwargs_drawstyle(kwargs, default="steps-post"):
    kwargs["drawstyle"] = kwargs.get("drawstyle", default)


def set_kwargs_label(kwargs, cls):
    kwargs["label"] = kwargs.get("label", cls._label)  

def create_dataframe_slicer(iloc, loc, timeline):
    if (loc is not None) and (iloc is not None):
        raise ValueError("Cannot set both loc and iloc in call to .plot().")

    user_did_not_specify_certain_indexes = (iloc is None) and (loc is None)
    user_submitted_slice = slice(timeline.min(), timeline.max()) if user_did_not_specify_certain_indexes else coalesce(loc, iloc)

    get_method = "iloc" if iloc is not None else "loc"
    return lambda df: getattr(df, get_method)[user_submitted_slice]

def cdf_plot(model, timeline=None, ax=None, **plot_kwargs):
    """
         Cumulative Distribution Function
    """    
    from lifelines import KaplanMeierFitter

    #kmf = KaplanMeierFitter()
    #kmf.fit(durations = churn_data['tenure'], event_observed = churn_data['Churn - Yes'] )
    
    if ax is None:
      ax = plt.gca()

    if timeline is None:
      timeline = model.timeline

    CDL_EMP = "empirical CDF"
    if CensoringType.is_left_censoring(model):
       emp_kmf = KaplanMeierFitter().fit_left_censoring(model.durations,
            model.event_observed, label=CDL_EMP,
            timeline=timeline, weights=model.weights,
            entry=model.entry)
    if CensoringType.is_right_censoring(model):
       emp_kmf = KaplanMeierFitter().fit_right_censoring(model.durations,
            model.event_observed, label=CDL_EMP,
            timeline=timeline, weights=model.weights,
            entry=model.entry)
    if CensoringType.is_interval_censoring(model):
       emp_kmf = KaplanMeierFitter().fit_interval_censoring(model.lower_bound,
            model.upper_bound, label=CDL_EMP,
            timeline=timeline, weights=model.weights,
            entry=model.entry)

