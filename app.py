#For Form
#from flask_wtf import FlaskForm
#from wtforms import StringField
#from wtforms.validators import InputRequired, DataRequired, ValidationError

import webull as wba
from webull import webull
try:
    import wbp
except:
    raise Exception("\n\nRemember to make a wbp.py file with your email and password.\n\nThe format should be as follows:\n\nIn wbp.py:\nWEBULLNAME='your@email.com'\nWEBULLPASS='your password'\nSECURITYCODE='your 6 digit code'\nquestionId='1001'\nquestionAnswer='your DOB or other answer'\nnickname='portfolio nickname'")
import numpy as np
import multiprocessing as mp
import datetime
import yfinance as yf
from pie import slice
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import webbrowser
from threading import Timer


hurdle = 1  # 1 = 100%

port = 5000
def open_browser():
	webbrowser.open_new(f"http://localhost:{port}")

class Portfolio:

    def __init__(self, wb):

        # Gather account info and all positions
        self.wb = wb
        self.account = wb.get_account()
        self.positions = wb.get_positions()
        self.networth = float(self.account['netLiquidation'])

        # Sort stocks and options
        self.sort()

        # Gather all options data
        self.gather_options_data()

        # Calculate portfolio greeks
        self.Theta()

        # Compute pie data
        self.PieData()

        # AROIC routines
        self.calculate_AROICS()
        self.findlow_AROICS()




    def sort(self):

        # Sort positions by stock vs option
        self.stocks = []
        self.options = []
        for position in self.positions:
            if position['assetType'] == 'OPTION':
                position['accounted'] = 0
                position['vertical'] = 0
                self.options.append(position)
            elif position['assetType'] == 'stock':
                self.stocks.append(position)

    def gather_options_data(self):

        # Gather all (unique) tickers
        self.option_tickers = []
        for option in self.options:
            ticker = option['ticker']['symbol']
            if ticker not in self.option_tickers:
                self.option_tickers.append(ticker)
        
        # Request all option data for all tickers in parallel
        self.option_data = dict(self.Parallelizer(self.option_gatherer, self.option_tickers))


    def Theta(self):
        
        # Initialize total profile theta
        self.theta = 0
        self.ticker_theta = dict(zip(self.option_tickers, np.zeros(len(self.option_tickers))))

        for option in self.options:

            # Gather quantity, ticker, and option's id
            quantity = float(option['position'])
            ticker = option['ticker']['symbol']
            brokerid = option['tickerId']

            # Find option in option_data
            option = self.option_data[ticker][brokerid]

            # Reduce to portfolio theta and ticker theta
            self.theta += quantity*float(option['theta'])*option['quoteLotSize']
            self.ticker_theta[ticker] += quantity*float(option['theta'])*option['quoteLotSize']

    def option_gatherer(self, Input):
        ticker = Input
        options = {}
        for date in self.wb.get_options_expiration_dates(ticker):
            for avail_option in self.wb.get_options(ticker, expireDate=date['date']):
                options[avail_option['call']['tickerId']] = avail_option['call']
                options[avail_option['put']['tickerId']]  = avail_option['put']
        return (ticker, options)
    
    def Parallelizer(self, f, Input):

        Parallel = mp.Pool()
        run = Parallel.map(f, Input)

        Parallel.close()
        Parallel.join()

        return run

    def calculate_AROICS(self):

        self.AROICS = {}
        for option in self.options:

            if option['accounted'] == option['vertical']:
                continue
                print('wrong one')

            # Get actual option data (option position --> option data)
            cost = float(option['costPrice'])
            qty = float(option['position'])
            ticker = option['ticker']['symbol']
            brokerid = option['tickerId']
            option = self.option_data[ticker][brokerid]

            if option['direction'] != 'put' or qty >=0:
                continue

            # Calculate days until expiration (+1 for end of day)
            today = datetime.date.today()
            expry = datetime.datetime.strptime(option['expireDate'], '%Y-%m-%d')
            DTE = (expry.date()-today).days + 1

            # Gather all AROIC parameters
            ask = np.min([float(exchange['price']) for exchange in option['askList']])
            bid = np.max([float(exchange['price']) for exchange in option['bidList']])
            mid = (ask + bid)/2
            strike = float(option['strikePrice'])
            price = yf.download(ticker, period='1d', interval='1m', progress=False)['Adj Close'][-1]

            # Non-intrinsic value and collateral (invested capital)
            NIV = mid + np.min([0, price-strike])
            col = strike-cost

            AROIC = (1 + NIV/col)**(365/DTE)-1

            key = f"${strike} {ticker} {option['direction']} for {expry.date()}"
            self.AROICS[key] = AROIC

    def findlow_AROICS(self):
    
        self.abovehurdle = []
        self.belowhurdle = []
        self.abovehurdle2 = []
        self.belowhurdle2 = []
        for key, value in self.AROICS.items():
            entry = f"{key} has an AROIC of {100*value:.2f}%."
            if value < hurdle:

                self.belowhurdle.append(entry)
                print(entry)

                strike, ticker, _, _, date = key.split()
                self.belowhurdle2.append([strike, ticker, date, f"{value:.2%}"])
            else:
                self.abovehurdle.append(entry)
                                
                strike, ticker, _, _, date = key.split()
                self.abovehurdle2.append([strike, ticker, date, f"{value:.2%}"])

        self.abovehurdle2 = pd.DataFrame(data=self.abovehurdle2, columns=["Strike","Ticker","Date","AROIC"])
        self.belowhurdle2 = pd.DataFrame(data=self.belowhurdle2, columns=["Strike","Ticker","Date","AROIC"])
        print("\n")

    def PieData(self):

        # Gather all (unique) tickers and initialize slice object
        # slice counts stock, (put) collateral, and long option value
        self.tickers = []
        self.pie_data = {}
        for position in self.positions:
            ticker = position['ticker']['symbol']
            if ticker not in self.tickers:
                self.tickers.append(ticker)
                self.pie_data[ticker] = slice(ticker)

        # Add up all stock
        for stock in self.stocks:

            ticker = stock['ticker']['symbol']
            self.pie_data[ticker].stock += float(stock['marketValue'])

        # Add up all options
        for q in range(len(self.options)):

            ticker = self.options[q]['ticker']['symbol']
            brokerId = self.options[q]['tickerId']
            direction = self.option_data[ticker][brokerId]['direction']
            qty = int(self.options[q]['position'])
 

            # Long call or put
            if qty >= 0 and self.options[q]['accounted'] != qty:

                # Account for long
                self.pie_data[ticker].options += float(self.options[q]['marketValue'])
                unaccounted = qty - self.options[q]['accounted'] 
                self.options[q]['accounted'] += unaccounted

                # Check for vertical
                expiry = self.option_data[ticker][brokerId]['expireDate']
                for p in range(len(self.options)):
                    
                    # Check other options
                    if p != q: 
                        ticker2 = self.options[p]['ticker']['symbol']
                        if ticker == ticker2:
                            brokerId2 = self.options[p]['tickerId']
                            option2 = self.option_data[ticker][brokerId2]
                            expiry2 = option2['expireDate']
                            direction2 = self.option_data[ticker][brokerId2]['direction']
                            qty2 = int(self.options[p]['position'])
                            if expiry == expiry2 and direction == direction2 and np.sign(qty) == -np.sign(qty2) and self.options[p]['accounted'] < -qty2:
                                print(f"{ticker} vertical detected")
                                qty_accted = np.min([qty, -qty2-self.options[p]['accounted']])
                                self.options[p]['accounted'] += qty_accted
                                self.options[q]['vertical'] += qty_accted
                                self.options[p]['vertical'] += qty_accted

                                # check if short vertical, add collateral if so:
                                lower_strike = np.argmin([self.option_data[ticker][brokerId]['strikePrice'], self.option_data[ticker][brokerId2]['strikePrice']])
                                if (direction == 'put' and int([self.options[q]['position'], self.options[p]['position']][lower_strike]) > 0) or (direction == 'call' and int([self.options[q]['position'], self.options[p]['position']][lower_strike]) < 0):

                                    
                                    dS = np.abs(float(self.option_data[ticker][brokerId]['strikePrice'])- float(self.option_data[ticker][brokerId2]['strikePrice']))
                                    collateral = float(option2['quoteLotSize'])*dS*qty_accted
                                    print(f"{ticker} short vertical detected. collateral = ${collateral:.2f}")
                                    self.pie_data[ticker].collateral += option2['quoteLotSize']*dS*qty_accted + float(self.options[p]['lastPrice'])*(-qty_accted)*option2['quoteLotSize'] - float(self.options[q]['lastPrice'])*(-qty_accted)*option2['quoteLotSize']
                                    self.pie_data[ticker].options -= float(self.options[q]['marketValue'])

                                else:
                                    print(f"Before = ${self.pie_data[ticker].options:.2f}")
                                    self.pie_data[ticker].options += float(self.options[p]['lastPrice'])*(-qty_accted)*option2['quoteLotSize']
                                    print(f"After = ${self.pie_data[ticker].options:.2f}")
                                break


        # Put collateral
        for q in range(len(self.options)):

            ticker = self.options[q]['ticker']['symbol']
            brokerId = self.options[q]['tickerId']
            direction = self.option_data[ticker][brokerId]['direction']
            qty = int(self.options[q]['position'])


            # Accounted attribute tracks verticals (which have no collateral) 
            if direction == 'put' and qty < 0 and self.options[q]['accounted'] < -qty:

                # Gather option position data
                price = float(self.options[q]['lastPrice'])
                cost = float(self.options[q]['costPrice'])
                qty = -qty - self.options[q]['accounted'] # Only non-verticals; 
                self.options[q]['accounted'] += qty
                brokerid = self.options[q]['tickerId']

                # Convert option position to option quote and get data
                option = self.option_data[ticker][brokerid]
                strike = float(option['strikePrice'])
                lot = float(option['quoteLotSize'])

                #collateral = (strike - cost)*qty*lot
                collateral = (strike - price)*qty*lot
                #print('Collateral', ticker, strike, strike*lot*qty, cost, collateral)

                self.pie_data[ticker].collateral += collateral
    
        # Total 
        for key, _ in self.pie_data.items():
            self.pie_data[key].total(True)
            # Convert to Proportion
            #self.pie_data[key].proportion(self.networth)



def app_maker():

    # Log into webull
    wb = webull()
    wb.login(wbp.WEBULLNAME, wbp.WEBULLPASS, wbp.nickname, wbp.SECURITYCODE, wbp.questionId, wbp.questionAnswer)
    try:
        if not wb.get_account()['success']:
            print("Login Unsuccessful. Check your password.")

            print("Sending a new 6 digit security code to your email -- update code in wbp.py")
            wb.get_mfa(wbp.WEBULLNAME)

            print("Make sure your questionId and answer are correct (if DOB, should be in format 'MM/DD')")
            print(wb.get_security(wbp.WEBULLNAME))
            
            raise Exception("Unsuccessful Login. Follow steps above and try again.")
    except:
        print("Login Successful\n")

    # Test Portfolio Initialization
    portfolio = Portfolio(wb)

    # Theta breakdown by ticker
    tickers = [key for key in portfolio.ticker_theta.keys()]
    thetas  = [theta for theta in portfolio.ticker_theta.values()]
    df = pd.DataFrame(data=thetas, columns=['Theta']); 
    df['Ticker'] = tickers
    theta_fig = px.bar(df, x='Ticker', y='Theta')


    # Create Data Frame
    entries = []
    totals = []
    maint = 0
    for ticker, slice in portfolio.pie_data.items():
        if slice.total <=0: 
            continue
        for i in range(4):
            
            # Total
            if i == 0:
                maint += slice.total
                totals.append(slice.total)
                #entries.append((ticker, 'Total', slice.total))
            # Stock
            elif i == 1:
                entries.append((ticker, 'Stock', slice.stock, slice.stock/portfolio.networth))
            # Collateral
            elif i == 2:
                entries.append((ticker, 'Collateral', slice.collateral, slice.collateral/portfolio.networth))
            # Options
            elif i == 3:
                entries.append((ticker, 'Options', slice.options, slice.options/portfolio.networth))
    if portfolio.networth-maint > 0:
        entries.append(('Cash', 'Cash', portfolio.networth-maint, (portfolio.networth-maint)/portfolio.networth))
        print(f"Cash = ${entries[-1][-2]:.2f}")
    
    df2 = pd.DataFrame(entries, columns=['Ticker', 'Type', 'Amount', 'Allocation'])

    
    layout = go.Layout(
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
            )
        )

    # Attempt 1
    #fig = go.Figure(data=[go.Pie(df, labels=labels, textinfo='label+percent', hover_data=['Totals', 'Stock', 'Collateral', 'Options'])], layout=layout)
    
    # Attempt 2
    """
     Any combination of ['label', 'text', 'value', 'current path', 'percent root', 'percent entry', 'percent parent'] 
     joined with '+' characters (I found this info by giving a mode which was not valid, the error message is actually very informative). For example:
    """
    fig = px.sunburst(df2, path=['Ticker','Type'], values='Amount', branchvalues='total')#, layout=layout)
    #fig = go.Figure(go.Sunburst(labels=df2['Type'], parents=df2['Ticker'], values=df2['Amount'], branchvalues='total'))#, textinfo='label+percent entry'))
    #fig = px.sunburst(labels=df2['Type'].astype("string"), parents=df2['Ticker'].astype("string"), values=df2['Amount'])
    fig.update_traces(textinfo='label+percent root')
    fig.update_layout(layout)


    # Initialize dashboard object
    app = dash.Dash()

    # Configure app layout
    app.layout = html.Div([
        html.H1(f"{wbp.nickname} Portfolio Dashboard", style={'textAlign': 'center'}),

        # Capital Allocation
        ################################################################################
        html.H4(f"Capital Allocation", style={'textAlign': 'center'}),
        html.Div(children=[
        dcc.Graph(figure=fig), #, style={'display': 'inline-block'}),
        ]),
        ################################################################################

        # Portfolio Theta
        ################################################################################
        html.H4(f"Total Portfolio Theta ${portfolio.theta:.2f}", style={'textAlign': 'center'}),
        html.Div(children=[
        dcc.Graph(figure=theta_fig), #, style={'display': 'inline-block'}),
        ]),
        ################################################################################

        # Puts Above/Below Hurdle
        ################################################################################
        dbc.Row([
            
            #html.Div([
            #    html.H3(f'Puts above {hurdle:.2%} hurdle:'),
            #    *[html.H6(entry) for entry in portfolio.abovehurdle]
            #], className="six columns"),

            #html.Div([
            #    html.H3(f'Puts below {hurdle:.2%} hurdle:'),
            #    *[html.H6(entry) for entry in portfolio.belowhurdle]
            #], className="six columns"),
            html.Div([

                html.H3(f'Puts above {hurdle:.2%} hurdle:'),

                dash_table.DataTable(
                    id='datatable-above',
                    columns=[
                        {"name": i, "id": i} for i in portfolio.abovehurdle2.columns #sorted(df.columns)
                    ],
                    data=portfolio.abovehurdle2.to_dict('records'),
                    sort_action='native'
                ),
            ]),
            html.Div([



                html.H3(f'Puts below {hurdle:.2%} hurdle:'),

                dash_table.DataTable(
                    id='datatable-below',
                    columns=[
                        {"name": i, "id": i} for i in portfolio.belowhurdle2.columns #sorted(df.columns)
                    ],
                    data=portfolio.belowhurdle2.to_dict('records'),
                    sort_action='native'
                )
            ])
        ])
        ################################################################################

        #html.Div([dcc.Graph(figure=fig), dcc.Graph(figure=fig)])
        ])

    
    # bit of CSS required for multi-column
    app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })

    Timer(2, open_browser).start()
    return app


if __name__ == '__main__':

    app = app_maker()

    app.run_server(debug=False, use_reloader=False, port=port) 

