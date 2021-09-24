#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:15:41 2020

@author: martinsandberg
"""

from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.4f}'.format

tickers=[]

def decision():
    stock=input("Which stock you want to invest in? (please input ticker):\n Answer: ")
    tickers.append(stock)
    a = input("Do you want to invest in more stocks? Y/N:\n Answer: ")
    if  a =='Y':
         return decision()
    elif a == 'N':
        return tickers
    else:
        return tickers

decision()

start='2011-01-01'
end='2020-03-20'

def DA_data(tickers, start, end):
    stock_data=DataReader(tickers, data_source='yahoo', start=start, end=end)['Adj Close']
    return stock_data

result_data=DA_data(tickers, start, end)
stock_returns=result_data.pct_change()
stock_returns=stock_returns.dropna()
cov_matrix=stock_returns.cov()
mu=stock_returns.mean()

rfr_data=DataReader('^TNX', data_source='yahoo', start=start, end=end)['Adj Close']
rfr_data=rfr_data/100
rfr_mean=rfr_data.mean()

bond_data=DataReader('AGG', data_source='yahoo', start=start, end=end)['Adj Close']
bond_returns=bond_data.pct_change()
bond_returns=bond_returns.dropna()
bond_mean=bond_returns.mean()

iterations=1000

#Numpy arrays for all simulations
all_weights = np.zeros((iterations,len(tickers)))
all_annual_returns = np.zeros((iterations,1))
all_annual_stdev = np.zeros((iterations,1))

#Optimizing with Sharpe Ratio
all_SR = np.zeros((iterations,1))
list_SR = ['Annual return','Stdev','Sharpe Ratio']

def optimization_SR():    
    for i in range(iterations):
        portfolio_weights = np.array(np.random.random(len(tickers)))
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        all_weights[i,:] = portfolio_weights
       
        portfolio_return = np.sum(mu*portfolio_weights) * 252
        all_annual_returns[i,:] = portfolio_return
        
        portfolio_stdev = (np.sqrt(np.dot(portfolio_weights.T,np.dot(cov_matrix, portfolio_weights)))) * np.sqrt(252)
        all_annual_stdev[i,:] = portfolio_stdev
        
        sharpe_ratio = (portfolio_return - rfr_mean) / portfolio_stdev
        all_SR[i,:] = sharpe_ratio
          
    numpy_data = np.concatenate([all_weights, all_annual_returns, all_annual_stdev, all_SR], axis = 1)
    dframe = pd.DataFrame(data = numpy_data)
    dframe.columns = tickers + list_SR
    max_SR = dframe.iloc[dframe['Sharpe Ratio'].idxmax()]
    
    #Pie Chart
    max_SR_weights = dframe.iloc[dframe['Sharpe Ratio'].idxmax(),0:len(tickers)]
    labels = tickers
    pie_chart = plt.pie(max_SR_weights, labels=labels, autopct='%1.1f%%')
    pie_chart = plt.title('Optimized Portfolio Weights')
    
    #Efficient Frontier Scatterplot  
    scatterplot = plt.subplots(figsize=(15,10))
    scatterplot = plt.scatter(dframe['Stdev'],dframe['Annual return'],c=dframe['Sharpe Ratio'],cmap='RdYlBu')
    scatterplot = plt.xlabel('Standard Deviation')
    scatterplot = plt.ylabel('Returns')
    scatterplot = plt.title('Efficient Frontier')
    scatterplot = plt.colorbar()
    scatterplot = plt.scatter(max_SR[-2],max_SR[-3],marker=(5,1,0),color='r',s=500)
    
    return max_SR, pie_chart, scatterplot

#Optimizing sharpe ratio with 40% in aggregate bond market ETF

all_weights_balanced_portfolio = np.zeros((iterations,len(tickers)+1))
stock_weight = 0.6
def optimization_weight():
    for i in range(iterations):
        bond_weight = 1 - stock_weight
        stock_weights = np.array(np.random.random(len(tickers)))
        stock_weights = (stock_weights / np.sum(stock_weights)) * 0.6
        portfolio_weights = np.append(stock_weights,bond_weight)
        all_weights_balanced_portfolio[i,:] = portfolio_weights
       
        portfolio_returns = (np.sum(mu * stock_weights) + (bond_weight * bond_mean)) * 252
        all_annual_returns[i,:] = portfolio_returns
        
        stock_returns['AGG'] = bond_returns
        cov_matrix = stock_returns.cov()
        portfolio_stdev = (np.sqrt(np.dot(portfolio_weights.T,np.dot(cov_matrix, portfolio_weights)))) * np.sqrt(252)
        all_annual_stdev[i,:] = portfolio_stdev
        
        sharpe_ratio = (portfolio_returns - rfr_mean) / portfolio_stdev
        all_SR[i,:] = sharpe_ratio
           
    numpy_data = np.concatenate([all_weights_balanced_portfolio, all_annual_returns, all_annual_stdev, all_SR], axis = 1)
    dframe = pd.DataFrame(data = numpy_data)
    dframe.columns = tickers + ['AGG'] + list_SR
    max_SR = dframe.iloc[dframe['Sharpe Ratio'].idxmax()]
    
    #Pie Chart
    max_SR_weights = dframe.iloc[dframe['Sharpe Ratio'].idxmax(),0:len(tickers)+1]
    labels = tickers + ['AGG']
    
    pie_chart = plt.pie(max_SR_weights, labels=labels, autopct='%1.1f%%')
    pie_chart = plt.title('Optimized Portfolio Weights')
    
    
    #Efficient Frontier Scatterplot   
    scatterplot = plt.subplots(figsize=(15,10))
    scatterplot = plt.scatter(dframe['Stdev'],dframe['Annual return'],c=dframe['Sharpe Ratio'],cmap='RdYlBu')
    scatterplot = plt.xlabel('Standard Deviation')
    scatterplot = plt.ylabel('Returns')
    scatterplot = plt.title('Efficient Frontier')
    scatterplot = plt.colorbar()
    scatterplot = plt.scatter(max_SR[-2],max_SR[-3],marker=(5,1,0),color='r',s=500)
    
    return max_SR, pie_chart, scatterplot

#Optimizing with Treynor ratio
all_treynor_ratios = np.zeros((iterations,1))

def optimization_T():

    benchmark_data = DA_data('SPY', start, end)
    benchmark_returns = benchmark_data.pct_change()
    benchmark_returns = benchmark_returns.dropna()
    
    for i in range(iterations):
    
        weights = np.array(np.random.random(len(tickers)))
        weights = weights / np.sum(weights)
        all_weights[i,:] = weights
        
        weighted_returns = stock_returns.mul(weights, axis = 1)
        portfolio_returns = weighted_returns.sum(axis = 1)
    
        beta = stats.linregress(benchmark_returns,portfolio_returns)[0]
        
        portfolio_return = np.sum(mu*weights) * 252
        all_annual_returns[i,:] = portfolio_return
        
        portfolio_sigma = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
        all_annual_stdev[i,:] = portfolio_sigma
        
        treynor_ratio = (portfolio_return - rfr_mean) / beta
        all_treynor_ratios[i,:] = treynor_ratio
        
    numpy_data = np.concatenate([all_weights, all_annual_returns, all_annual_stdev, all_treynor_ratios], axis = 1)
    dframe = pd.DataFrame(data = numpy_data)
    measures = ['Annual return','Stdev','Treynor Ratio']
    columns = tickers + measures
    dframe.columns = columns
    max_Treynor=dframe.iloc[dframe['Treynor Ratio'].idxmax()]
    
    
    #Pie Chart
    max_TR_weights = dframe.iloc[dframe['Treynor Ratio'].idxmax(),0:len(tickers)]
    labels = tickers 
    
    pie_chart = plt.pie(max_TR_weights, labels=labels, autopct='%1.1f%%')
    pie_chart = plt.title('Optimized Portfolio Weights')
    
    
    #Efficient Frontier Scatterplot
    scatterplot = plt.subplots(figsize=(15,10))
    scatterplot = plt.scatter(dframe['Stdev'],dframe['Annual return'],c=dframe['Treynor Ratio'],cmap='RdYlBu')
    scatterplot = plt.xlabel('Standard Deviation')
    scatterplot = plt.ylabel('Returns')
    scatterplot = plt.title('Efficient Frontier')
    scatterplot = plt.colorbar()
    scatterplot = plt.scatter(max_Treynor[-2],max_Treynor[-3],marker=(5,1,0),color='r',s=500)
 
    return max_Treynor, pie_chart, scatterplot

#Optimizing with Risk Parity
list2=['tolerance']
list2 = list2+tickers
simulation_resT= np.zeros((len(tickers)+1,iterations))
listt=['AGG','DBMF','^TYX']

def get_weights():
    tickers2=tickers+listt
    prices = DA_data(tickers2, start, end)
    stock_returns=prices.pct_change()
    stock_returns=stock_returns.dropna()
    covariances = stock_returns.cov()
    simulation_resT= np.zeros((len(tickers2)+1,iterations))
    for i in range(iterations):
        weights = np.array(np.random.random(len(tickers2)))
        weights /=np.sum(weights)
        TV = np.sqrt(weights.T.dot(covariances.dot(weights)))    
        Individual_Contributions = TV/len(tickers2)
        IC_matrix = covariances.dot(weights)
        IC = []
        for j in range(0,len(tickers2)):
            IC.append(IC_matrix[j]*weights[j])
        vol = np.var(IC)
        simulation_resT[0,i]=vol
        for k in range(len(weights)):
            simulation_resT[k+1,i]=weights[k]
    dframe = pd.DataFrame(simulation_resT.T)
    list2=['tolerance']
    list2 = list2+tickers2
    dframe.columns=list2
    a = dframe.iloc[dframe['tolerance'].idxmin()]
    
    #Pie Chart
    riskp_parity_weights = dframe.iloc[dframe['tolerance'].idxmin(),1:len(tickers2)+1]
    labels = tickers2
    
    pie_chart = plt.pie(riskp_parity_weights, labels=labels, autopct='%1.1f%%')
    pie_chart = plt.title('Optimized Portfolio Weights')
    
    return a, pie_chart

#Optimizing with Sortino ratio
all_TDD = np.zeros((iterations,1))
all_sortino = np.zeros((iterations,1))

def optimization_S():    
    for i in range(iterations):
        portfolio_weights = np.array(np.random.random(len(tickers)))
        portfolio_weights = portfolio_weights / np.sum(portfolio_weights)
        all_weights[i,:] = portfolio_weights
        
        weighted_returns = stock_returns.mul(portfolio_weights, axis = 1)
        portfolio_returns = weighted_returns.sum(axis = 1)
        daily_avg_return = np.mean(portfolio_returns) * 252
        all_annual_returns[i,:] = daily_avg_return
        
        diff = (portfolio_returns - daily_avg_return)
        
        sordf = pd.DataFrame({'Portfolio returns':portfolio_returns,'Return - Avg':diff})
        sordf['Downside deviation'] = np.where(sordf['Return - Avg'] > 0, 0, sordf['Return - Avg'])
        sordf['Downside deviation squared'] = sordf['Downside deviation'] ** 2
        
        number_of_returns = sordf['Downside deviation squared'].count()
        sum_of_squared_dd = np.sum(sordf['Downside deviation squared'])
        tdd = np.sqrt(sum_of_squared_dd / number_of_returns)
        all_TDD[i,:] = tdd
        
        sortino_ratio = (daily_avg_return - rfr_mean) / tdd
        all_sortino[i,:] = sortino_ratio
        
        daily_cov_matrix = stock_returns.cov()
        daily_stdev = np.sqrt(np.dot(portfolio_weights.T, np.dot(daily_cov_matrix, portfolio_weights))) * np.sqrt(252)
        all_annual_stdev[i,:] = daily_stdev
        
    numpy_data = np.concatenate([all_weights, all_annual_returns, all_annual_stdev, all_TDD, all_sortino], axis = 1)
    dframe = pd.DataFrame(data = numpy_data)
    measures = ['Annual return','Stdev','TDD','Sortino Ratio']
    columns = tickers + measures
    dframe.columns = columns
    max_Sortino = dframe.iloc[dframe['Sortino Ratio'].idxmax()]
    
    #Create a pie chart
    max_sortino_weights = dframe.iloc[dframe['Sortino Ratio'].idxmax(),0:len(tickers)]
    labels = tickers
    
    pie_chart = plt.pie(max_sortino_weights, labels=labels, autopct='%1.1f%%')
    pie_chart = plt.title('Optimized Portfolio Weights')
    
    #Creating Scatterplot    
    scatterplot = plt.subplots(figsize=(15,10))
    scatterplot = plt.scatter(dframe['Stdev'],dframe['Annual return'],c=dframe['Sortino Ratio'],cmap='RdYlBu')
    scatterplot = plt.xlabel('Standard Deviation')
    scatterplot = plt.ylabel('Returns')
    scatterplot = plt.title('Efficient Frontier')
    scatterplot = plt.colorbar()
    scatterplot = plt.scatter(max_Sortino[-3],max_Sortino[-4],marker=(5,1,0),color='r',s=500)

    return max_Sortino, pie_chart, scatterplot


def questionnarie():
    points = 0
    print ("Your comfort level with investment risk is important in determining how aggressively or conservatively you choose to invest. Please answer the following questions to determine your risk tolerance level.")
    age = input("1. Please pick your age range?:\n(a)18 - 25\n(b)26 - 35\n(c)36 - 50\n(d)50 upwards\n\n Answer:")
    if age == "a":
        points += 3
    elif age == "b":
        points += 4
    elif age == "c":
        points += 2
    elif age == "d":
        points += 1

    income = input("2. What is your annual income range?:\n(a)below $150000 \n(b))$150000 - $500000\n(c)above $500000\n\n Answer:") 
    if income == "a":
        points += 1
    elif income == "b":
        points += 2
    elif income == "c":
        points += 3

    education = input("3. I would describe my knowledge of investment as:\n(a)None\n(b)Limited\n(c)Good\n(d)Extensive\n\n Answer:")
    if education == "a":
        points += 1
    elif education == "b":
        points += 3
    elif education == "c":
        points += 7
    elif education == "d":
        points += 10

    investment = input("4. When I invest my money, I am:\n(a)Most concerned about my investment losing value\n(b)Equally concerned about my investment losing or gaining value\n(c)Most concerned about my investment gaining value\n\n Answer:")
    if investment == "a":
        points += 0
    elif investment == "b":
        points += 4
    elif investment == "c":
        points += 8
    
    aversity = input("5. Which one of the following statements describes your feeling toward choosing your retirement investment choices?:\n\n(a)I would prefer investment options that have a low degree of risk associated with them\n\n(b)I prefer a mix of investment options that emphasizes those with a low degree of risk and includes a small portion of other choices that have a higher degree of risk but may yield greater returns\n\n(c)I prefer a balanced mix of investment options - some that have a low degree of risk and others that have a higher degree of risk but may yield greater returns\n\n(d)I prefer a mix of investment options - some would have a low degree of risk but the emphasis would be on investment options that have a higher degree of risk but may yield greater retruns\n\n(e)I would select only investment options that have a higher degree of risk but a greater potential for higher returns\n\n Answer: ")
    if aversity == "a":
        points += 1
    elif aversity == "b":
        points += 2
    elif aversity == "c":
        points += 3
    elif aversity == "d":
        points += 4
    elif aversity == "e":
        points += 7

    aversity2 = input("6. If you could increase your chances of improving your returns by taking more risk, would you:\n(a) Be willing to take a lot more risk with all your money?\n(b) Be willing to take a lot more risk with some of your money?\n(c) Be willing to take a little more risk with all of your money?\n(d)Be willing to take a little more risk with some of your money\n(e) Be unlikely to take much more risk?\n\n Answer:")
    if aversity2 == "a":
        points += 5
    elif aversity2 == "b":
        points += 4
    elif aversity2 == "c":
        points += 3
    elif aversity2 == "d":
        points += 2
    elif aversity2 == "e":
        points += 1
    
    aversity3 = input("7. Imagine in the past three months, the overall stock market lost 25% of it value. An individual stock investment you also own lost 25% of its value. What would you do?:\n(a)Sell all my shares\n(b)Sell some of my shares\n(c)Do nothing\n(d)Buy more shares\n\n Answer:")   
    if aversity3 == "a":
        points += 0
    elif aversity3 == "b":    
        points += 2
    elif aversity3 == "c":
        points += 5
    elif aversity3 == "d":
        points += 8

    if points <= 15:
        risk_aversion="H"
    elif points <= 35:
        risk_aversion="M"
    elif points > 35:
        risk_aversion="L"
        
    return risk_aversion
    

def strategy():
    familiar=input("Are you familiar with portfolio management techniques? Y/N:\n Answer: ")
    if familiar=='Y':
        optimization_method=input("How do you want to optimize your portfolio?\n-SR:Sharpe Ratio\n-T:Treynor Ratio\n-S:Sortino Ratio\n-RP:Risk Parity\n Answer:")
        if optimization_method=='SR':
            print("The portfolio that maximizes your Sharpe Ratio is:\n", optimization_SR())
        elif optimization_method=='T':
            print("The portfolio that maximizes your Treynor Ratio is: \n", optimization_T())
        elif optimization_method=='RP':
            print("Under risk parity, the weight would be:", get_weights())
        elif optimization_method=='S':
            print("The portfolio that maximizes your Sortino Ratio is: \n", optimization_S())
    else:
        risk_aversion=questionnarie()
        if risk_aversion=='H':
            print("Your risk aversion was classified as high, therefore, we recommend using Risk Parity for portfolio optimization")
            print("Under risk parity, the weight would be:", get_weights())
        elif risk_aversion=='M':
            print("Your risk aversion was classified as medium, therefore, we recommend using a 60/40 portfolio with Sharpe Ratio optimization")
            print("Your optimized Sharpe ratio is:\n",optimization_weight())
        elif risk_aversion=='L':
            print("Your risk aversion was classified as low, therefore, we recommend using Sharpe Ratio for portfolio optimization")
            print("The portfolio that maximizes your Sharpe Ratio is\n", optimization_SR())

strategy()