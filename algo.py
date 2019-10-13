rebalance_frequency = 5

def initialize(context):
    context.stocks = {
        # iShares Europe ETF - Sensitive
        symbol('IEV'),
        # iShares MSCI Europe Financials ETF - Defensive
        symbol('EUFN'),
        # Direxion Daily Consumer Staples Bull 3X Shares - Defensive
        symbol('NEED'),
        # Invesco QQQ - Sensitive
        symbol('QQQ'),
        # Global X MSCI Consumer Staples ETF - Defensive
        symbol('CHIS'),
        # Global X MSCI China Consumer Discretionary ETF - Sensitive
        symbol('CHIQ'),
        # Shares MSCI Brazil Small-Cap ETF - Defensive
        symbol('EWZS'),
        # First Trust Brazil AlphaDEX Fund - Sensitive
        symbol('FBZ'),
        # WisdomTree Emerging Markets Consumer Growth Fund - Defensive
        symbol('EMCG'),
        # Emerging Markets Internet & Ecommerce ETF - Sensitive
        symbol('EMQQ'),
    }

def handle_data(context, data):
    context.ticks += 1
    if (context.tick < rebalance_frequency):
        return

    # Get today's 'real-time' inflation index value
    daily_inflation = data['Inflation_Index'].price
    context.days_since_rebal +=1
    current_CPI = daily_inflation

    prices = history(
        assets = context.stocks, fields="price",
        bar_count = 100, frequency = "1d"
    )
    returns = prices.pct_change().dropna()
    
    try:
        weights, _, _ = optimal_portfolio(returns.T)
        # Rebalance portfolio  
        for stock, weight in zip(prices.columns, weights):  
            if (currentCPI > 0):
                if (stock in {
                            # iShares Europe ETF - Sensitive
                            symbol('IEV'),
                            # Direxion Daily Consumer Staples Bull 3X Shares - Defensive
                            symbol('NEED'),
                            # Global X MSCI Consumer Staples ETF - Defensive
                            symbol('CHIS'),
                            # Shares MSCI Brazil Small-Cap ETF - Defensive
                            symbol('EWZS'),
                            # WisdomTree Emerging Markets Consumer Growth Fund - Defensive
                            symbol('EMCG'), 
                        }):
                    order_target_percent(stock, weight + 0.01) 
                else:
                    order_target_percent(stock, weight - 0.01)
            else:
                if (stock in {
                            # iShares Europe ETF - Sensitive
                            symbol('IEV'),
                            # Direxion Daily Consumer Staples Bull 3X Shares - Defensive
                            symbol('NEED'),
                            # Global X MSCI Consumer Staples ETF - Defensive
                            symbol('CHIS'),
                            # Shares MSCI Brazil Small-Cap ETF - Defensive
                            symbol('EWZS'),
                            # WisdomTree Emerging Markets Consumer Growth Fund - Defensive
                            symbol('EMCG'), 
                        }):
                    order_target_percent(stock, weight - 0.01) 
                else:
                    order_target_percent(stock, weight + 0.01)
    except ValueError as e:  
        pass

def optimal_portfolio(returns):
        n = len(returns)  
    returns = np.asmatrix(returns)  
    N = 100  
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
    # Convert to cvxopt matrices  
    S = opt.matrix(np.cov(returns))  
    pbar = opt.matrix(np.mean(returns, axis=1))  
    # Create constraint matrices  
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix  
    h = opt.matrix(0.0, (n ,1))  
    A = opt.matrix(1.0, (1, n))  
    b = opt.matrix(1.0)  
    # Calculate efficient frontier weights using quadratic programming  
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus]  
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
    returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
    m1 = np.polyfit(returns, risks, 2)  
    x1 = np.sqrt(m1[2] / m1[0])  
    # CALCULATE THE OPTIMAL PORTFOLIO  
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
    return np.asarray(wt), returns, risks