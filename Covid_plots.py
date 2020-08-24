# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # My COVID-19 data experiments
# These are my attempts at visualizing and analysing CoVID-19 data
# put out on the github datasets curated from the JHU data.
# *Do not take anything here too seriously*. 
# I am just a physicist trying to calm my mind and maybe provide some
# useful information to people during lock-down by doing some simple 
# data anlysing/modelling. If you want *real* research go to the
# *real* epidemiologists at some university/institute *not* to 
# the random guy on the internet (i.e. me).

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import numpy as np
from numpy import array, arange, polyfit, log, exp, polyval, linspace
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# %%
# Use logarithmic scales?
LOGY=True

# Select countries of general interest
selcnt = []
#selcnt += ['China', 'Korea, South']
selcnt += ['United Kingdom', 'US']
selcnt += ['Sweden','Germany','Norway']
selcnt += ['Italy', 'Spain']
selcnt += ['Russia','Brazil']

# Countries for plwiki plots
plwiki = ['Poland', 'Slovakia', 'Germany', 'Czechia', 'Ukraine', 'Belarus', 'Russia']

# Countries to read in
countries = list(set(plwiki).union(set(selcnt)))


# %% jupyter={"source_hidden": true}
# Prepare the data

def fix_names(c):
    '''
    Fix differences in naming in population and covid datasets
    '''
    mapa = {'Korea, Rep.':'Korea, South',
            'United States':'US',
            'Slovak Republic':'Slovakia',
            'Czech Republic':'Czechia',
            'Russian Federation':'Russia'
           }
    rmap = {v:k for k,v in mapa.items()}
    if c in mapa:
        return mapa[c]
    elif c in rmap:
        return rmap[c]
    else :
        return c

# Loding population data
pop = pd.read_csv('https://raw.githubusercontent.com/datasets/population/master/data/population.csv')

# Population uses different country names - map it
pop_cnt = [fix_names(c) for c in countries]
population = {fix_names(c):n for c, _, _, n in 
                   pop[pop['Country Name'].isin(pop_cnt) & (pop.Year==2018)].values}


# Loading covid data
df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv', parse_dates=['Date'])
# Limit to the selected countries
df = df[df['Country'].isin(countries)]

conf = df.pivot(index='Date', columns='Country', values='Confirmed')
recov = df.pivot(index='Date', columns='Country', values='Recovered')
vict = df.pivot(index='Date', columns='Country', values='Deaths')
relgr = conf.pct_change()

# Compute per capita values (per 100_000)
confpc = conf.copy()
for country in confpc:
    confpc[country] = 1e6*confpc[country]/population[country]

victpc = vict.copy()
for country in victpc:
    victpc[country] = 1e6*victpc[country]/population[country]
    
recovpc = recov.copy()
for country in victpc:
    recovpc[country] = 1e6*recovpc[country]/population[country]

# %% [markdown]
# ## Relative growth in time
#
# This one goes to the heart of exponential growth. If it is exponential its relative growth is constant. If not - we will get linear change or other curve.
# You can distinquish the exponent by its growth rate much easier.
#
# Additionally exponential decay seems to fit the growth rate curves quite well.

# %% jupyter={"source_hidden": true}
fig = plt.figure(figsize=(10,7))
span = 5
rel = relgr.ewm(halflife=span).mean()

for n, c in enumerate(selcnt):
    m = ~ (np.isnan(rel[c].values) | np.isinf(rel[c].values))
    t = np.arange(m.size)
    t = rel.index.to_pydatetime()
    for s, v in zip(t[m][::-1], rel[c].values[m][::-1]):
        if v>0.3 :
            break
    mm = m & (t > s)
    x = arange(rel.index.size)
    fit = polyfit(x[mm], log(rel[c].values[mm]), 1)
    p = plt.semilogy(rel.index[m], 100*rel[c].values[m], '.')[0]
    plt.plot(rel.index[mm], 100*rel[c].values[mm], 'o', label=c, color=p.get_color())
    plt.plot(rel.index[mm], 100 * exp(polyval(fit, x[mm])), color=p.get_color())

plt.axhline(5, ls='--', label='approx. critical value (5%)')
plt.axhline(2, ls=':', label='effective critical value (2%)')
plt.ylim(None,50)
plt.xlim(pd.Timestamp('2020-03-5'),None)
plt.title('Daily relative growth of COVID-19 cases', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel(f'Relative growth (%)\n{span}-day exponential weighted mean')
plt.xlabel('Date')
plt.grid()
plt.legend()
plt.savefig('relative_growth.png');

# %% [markdown]
# ## Trajectory 
# This one is inspired by the excellent https://aatishb.com/covidtrends/ page

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(10,7))
span = 7
val = confpc
gr = val.diff().ewm(halflife=span).mean()
for n, c in enumerate(selcnt):
    m = ~ gr[c].isnull()
    plt.loglog(val[c][m].values, gr[c][m].values, '-', label=c, lw=2)

plt.xlim(10,None)
plt.ylim(1,3e2)
plt.title(f'Trajectory of COVID-19 pandemic ({str(val.index[-1]).split()[0]})', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel('Average of daily growth per 1 mln people\n'+
           f'{span}-day exponential weighted mean')
plt.xlabel('Cases per 1 mln people')
plt.legend()
plt.grid()
plt.savefig('trajectory.png');

# %% [markdown]
# ## Other curves

# %% jupyter={"source_hidden": true}
percapitaplot = confpc.ewm(halflife=span).mean()[selcnt].plot(figsize=(12,8), linewidth=5, logy=LOGY)
percapitaplot.grid(color='#d4d4d4')
percapitaplot.set_xlabel('Date')
percapitaplot.set_ylabel('# of Cases per 1 mln People\n'+
                         f'{span}-day exponential weighted mean')
percapitaplot.set_xlim(pd.Timestamp('2020-03-1'),None)
percapitaplot.set_ylim(1e-1, None)
percapitaplot.set_title("Per Capita COVID-19 Cases", 
                        fontsize = 16, weight = 'bold', alpha = .75);

# %% jupyter={"source_hidden": true}
percapitaplot = (confpc - recovpc - victpc).ewm(halflife=span).mean()[selcnt].plot(figsize=(12,8), linewidth=5)
percapitaplot.grid(color='#d4d4d4')
percapitaplot.set_xlabel('Date')
percapitaplot.set_ylabel(f'# of Active cases per 1 mln people\n'+
                         f'({span}-day exponential weighted mean)')
percapitaplot.set_xlim(pd.Timestamp('2020-03-1'),None)
percapitaplot.set_title("Per Capita Active COVID-19 Cases", 
                        fontsize = 16, weight = 'bold', alpha = .75);
plt.gcf().savefig('percapita_active.png');

# %% jupyter={"source_hidden": true}
vplot = victpc.ewm(halflife=span).mean()[selcnt].plot(figsize=(12,8), linewidth=5, logy=False)
vplot.grid(color='#d4d4d4')
vplot.set_xlabel('Date')
vplot.set_ylabel('# of Deaths per 1 mln People\n'+
                 f'({span}-day exponential weighted mean)')
vplot.set_xlim(pd.Timestamp('2020-03-1'),None)
vplot.set_ylim(1e-2, None)
vplot.set_title("Per Capita deaths due to COVID-19 Cases", fontsize = 16, weight = 'bold', alpha = .75);

# %% jupyter={"source_hidden": true}
mortplt = (100*vict/conf).ewm(halflife=span).mean()[selcnt].plot(figsize=(12,8), linewidth=5, logy=False)
mortplt.grid(color='#d4d4d4')
mortplt.set_xlim(pd.Timestamp('2020-03-1'),None)
mortplt.set_ylim(0, 20)
mortplt.set_xlabel('Date')
mortplt.set_ylabel('Mortality rate (%)\n'+f'{span}-day exponential weighted mean')
mortplt.set_title('Mortality rate due to COVID-19', fontsize = 16, weight = 'bold', alpha = .75);

# %% [markdown]
# ## Polish Wikipedia plots
# These are plots created for Polish Wikipedia

# %% jupyter={"source_hidden": true}
fig = plt.figure(figsize=(10,7))

def plleg(c):
    pl = {
        'Poland':'Polska', 
        'Slovakia': 'Słowacja', 
        'Germany': 'Niemcy',
        'Czechia': 'Czechy',
        'Ukraine': 'Ukraina', 
        'Belarus': 'Białoruś', 
        'Russia': 'Rosja'
    }
    if c in pl:
        return pl[c]
    else :
        return c

span = 3
rel = relgr.ewm(halflife=span).mean()

model = {}

for n, c in enumerate(plwiki):
    m = ~ (np.isnan(rel[c].values) | np.isinf(rel[c].values))
    t = np.arange(m.size)
    t = rel.index.to_pydatetime()
    for s, v in zip(t[m][::-1], rel[c].values[m][::-1]):
        if v>0.3 :
            break
    mm = m & (t > s)
    x = arange(rel.index.size)
    fit = polyfit(x[mm], log(rel[c].values[mm]), 1)
    model[c] = fit, x[mm]
    p = plt.semilogy(rel.index[m], 100*rel[c].values[m], '.')[0]
    plt.plot(rel.index[mm], 100*rel[c].values[mm], 
             'd' if c=='Poland' else 'o', 
             color=p.get_color(), label=plleg(c),
             zorder = 3 if c=='Poland' else 2,
            )
    plt.plot(rel.index[mm], 100 * exp(polyval(fit, x[mm])),
             color=p.get_color(),
             lw=3 if c=='Poland' else 2,
             zorder = 3 if c=='Poland' else 2)

plt.axhline(5, ls='--', label='Przybliżony poziom krytyczny (5%)')
plt.axhline(2, ls=':', label='Efektywny poziom krytyczny (2%)')
plt.ylim(None,50)
plt.xlim(pd.Timestamp('2020-03-5'),None)
plt.title(f'Dzienny wzrost przypadków COVID-19 ({str(rel.index[-1]).split()[0]})', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel(f'Dzienny wzrost zakażeń (%, {span}-dniowa wykładnicza średnia krocząca)')
plt.xlabel('Data')
plt.grid()
plt.legend(loc='lower left')
plt.savefig('wzrosty_dzienne.png', dpi=72);

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(10,7))
val = confpc
span = 7
gr = val.diff().ewm(halflife=span).mean()

for n, c in enumerate(plwiki):
    m = ~ gr[c].isnull()
    p = plt.loglog(val[c][m].values, gr[c][m].values, 
               '-', lw=3 if c=='Poland' else 2, label=plleg(c),
               zorder = 3 if c=='Poland' else 2)
    (b, a), t = model[c]
    c0 = val[c][-1]/exp(exp(b*t.max()+a)/b)
    ci = gr[c][m].values[-1]/(exp(exp(b*t.max()+a)/b)*exp(b*t.max()+a))
    t = linspace(t.min(), t.max()+60, 100)    
    #plt.loglog(exp(exp(b*t+a)/b)*c0,exp(exp(b*t+a)/b)*exp(b*t+a)*ci, ls=':', color=p[0].get_color())

plt.xlim(2,None)
plt.ylim(0.1,None)
plt.title(f'Trajektoria epidemii COVID-19 ({str(val.index[-1]).split()[0]})', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel('Średni dzienny przyrost zakażeń na 1 mln mieszkańców\n'+
           f'{span}-dniowa wykładnicza średnia krocząca')
plt.xlabel('Liczba przypadków na 1 mln mieszkańców')
plt.legend(loc='upper left')
plt.grid()
plt.savefig('trajektoria_covid.png', dpi=72);

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(10,7))
val = (confpc - recovpc - victpc)
span = 3
val = val.ewm(halflife=span).mean()

for n, c in enumerate(plwiki):
    m = ~ (np.isnan(val[c].values) | np.isinf(val[c].values))
    plt.plot(val.index[m], val[c].values[m], '-',
             lw=3 if c=='Poland' else 2, label=plleg(c),
               zorder = 3 if c=='Poland' else 2
            )

plt.ylim(1e-2,None)
plt.xlim(pd.Timestamp('2020-03-5'),None)
plt.title(f'Liczba aktywnych przypadków COVID-19 ({str(val.index[-1]).split()[0]})', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel(f'Aktywne przypadki na 1 mln. mieszkańców\n({span}-dniowa wykładnicza średnia krocząca)')
plt.xlabel('Data')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('aktywne_przypadki.png', dpi=72);

# %% jupyter={"source_hidden": true}
plt.figure(figsize=(10,7))
val = (confpc - recovpc - victpc)
span = 3
val = val.diff().ewm(halflife=span).mean()

for n, c in enumerate(plwiki):
    m = ~ (np.isnan(val[c].values) | np.isinf(val[c].values))
    plt.plot(val.index[m], val[c].values[m], '-',
             lw=3 if c=='Poland' else 2, label=plleg(c),
               zorder = 3 if c=='Poland' else 2
            )

plt.ylim(None,None)
plt.axhline(ls='--', label='Zerowy wzrost')
plt.xlim(pd.Timestamp('2020-03-5'),None)
plt.title(f'Wzrost aktywnych przypadków COVID-19 ({str(val.index[-1]).split()[0]})', fontsize = 16, weight = 'bold', alpha = .75)
plt.ylabel(f'Wzrost dzienny aktywnych przypadków na 1 mln. mieszkańców\n({span}-dniowa wykładnicza średnia krocząca)')
plt.xlabel('Data')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('aktywne_wzrost.png', dpi=72);

# %% [markdown]
# ## Experiments
#
# Here are some experiments with modelling based on the remarkable good fit
# of the relative daily growth curves. This is not very much work in progress.
# And as everything here - this is just my experiments to calm my mind.

# %%
c = 'Germany'
m = ~ gr[c].isnull()
plt.plot(confpc[c][m].values, gr[c][m].values,'.-')    
(b, a), t = model[c]
c0 = confpc[c][-1]/exp(exp(b*t.max()+a)/b)
ci = gr[c][m].values[-1]/(exp(exp(b*t.max()+a)/b)*exp(b*t.max()+a))
t = linspace(t.min(), t.max()+60, 100)    
plt.loglog(exp(exp(b*t+a)/b)*c0,exp(exp(b*t+a)/b)*exp(b*t+a)*ci, '-');

# %%
model

# %%
conf.diff()[-3:][['Poland','US']]

# %%
(conf-recov-vict)[-3:][['Poland','US']]

# %%
