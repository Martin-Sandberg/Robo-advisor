# Robo-advisor for Fintech project

Created a robo-advisor that matches an investment strategy with the investor's risk profile. 

### Risk Assessment and Investment Strategy

The robo-advisor determines the investor's risk profile by scoring answers of an initial questionarre which places the investor into one of three categories: low risk aversity, medium risk aversity, and high risk aversity. 

If the investor is classified as high risk averse the program will optimize the portfolio using the risk parity investment strategy, balancing the risk exposure of the portfolio so that each asset in our portfolio contributes the same proportion to the total risk of the portfolio. The assets used for optimization: 1) the stock portfolio selected by the investor, 2) the "AGG" core bond fund ETF, 3) Treasury Yield Index-30 Yr bond, and 4) iM DBi Managed Futures Strategy ETF. 

For investors with moderate risk aversion, we provide them with a classic 60/40 investment strategy. A 60/40 portfolio means that 60% of assets will be invested in stocks and another 40% of assets will be invested in bonds. The assets used for optimization: 1) the stock portfolio selected by the investor, and 2) the "AGG" core bond fund ETF.

For investors with low risk aversity, the Sharpe Ratio of the portfolio is optimized with the stocks selected by the investor. 

Additionally, if the investor decides to decide the portfolio optimization technique by himself, then he would be given the possibility to choose the following portfolio optimization techniques: 
1.	Risk Parity
2.	Sharpe Ratio 
3.	Sortino Ratio 
4.	Treynor Ratio 

### Monte Carlo Simulation

The portfolio weights are then determined by running a Monte Carlo Simulation with 1000 iterations.

