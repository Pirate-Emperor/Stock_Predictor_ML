import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
# DASH SERVER
app = dash.Dash()
server = app.server
# NORMAL DISTRIBUTION
scaler=MinMaxScaler(feature_range=(0,1))

# SET OF DATASET

set_df=["ADANIPORTS","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL",
"BPCL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT","GAIL","GRASIM","HCLTECH","HDFC",
"HDFCBANK","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFRATEL","INFY","IOC",
"ITC","JSWSTEEL","KOTAKBANK","LT","MARUTI","MM","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN",
"SHREECEM","SUNPHARMA","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","VEDL","WIPRO",
"ZEEL"
]
dfs=pd.read_csv("archive//NIFTY50_all.csv")
dfs2=dfs
dfs2["Date"]=pd.to_datetime(dfs.Date,format="%Y-%m-%d")
# READ DATASET
df_nse = pd.read_csv("ADANIPORTS.csv")
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.sort_values("Date", inplace=True)
df_nse.index=df_nse['Date']
data=df_nse.sort_index(ascending=True,axis=0)

# DATASET TO BE OPERATED ON
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:1000,:]
valid=dataset[1000:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# Training Data
x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# Loading Model 
model=load_model("saved_lstm_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

# Testing Phase
X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:1000]
valid=new_data[1000:]
valid['Predictions']=closing_price


# Displaying Plots 
df= pd.read_csv("./stock_data.csv")

app.layout = html.Div(
    children=[
    html.Div(
            children=[
                html.P(children=[html.Img(src="C:\\Users\\Rahul\\Downloads\\materials-python-dash\\apps\\app_4\\assets\\images.png",alt="Yo")], className="header-emoji"),
                html.H1(
                    children="Stock Price Analysis Dashboard", className="header-title"
                ),
                html.P(
                    children="Study the future stock price of the top firms to invest in stocks and sell at the right time to get the optimum profit",
                    className="header-description",
                ),
            ],
            className="header",
        ),
    
    html.Div(
        children=[
        dcc.Tabs(
            id="tabs", 
            parent_className='custom-tabs',
            className='custom-tabs-container',children=[
            dcc.Tab(
                label='Predictor Zone',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
            html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Company", className="menu-title"),
                        dcc.Dropdown(
                            id="region-filter",
                            options=[
                                {"label": region, "value": region}
                                for region in np.sort(dfs.Symbol.unique())
                            ],
                            value="Albany",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range", className="menu-title"
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=dfs2.Date.min().date(),
                            max_date_allowed=dfs2.Date.max().date(),
                            start_date=dfs2.Date.min().date(),
                            end_date=dfs2.Date.max().date(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="price-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="volume-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
        ]),
        dcc.Tab(label='Developer Tab',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
			html.Div([
				html.H2("Model Creation",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data",
					figure={
                        "data":[
                            go.Scatter(
                                x=train.index,
                                y=train["Close"],
                                #mode='markers'
                            ),
                        ],
                        "layout":go.Layout(
                            title='Actual Closing Price',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    },
                    className="card",
				),
                html.H2("Model Testing",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual1 Data",
					figure={
                        "data":[
                            go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								#mode='markers'
							),
                        ],
                        "layout":go.Layout(
                            title='Predicted Closing Price',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    },
                    className="card",

				),
				html.H2("Actual-LSTM Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data",
					figure={
						"data":[
                            go.Scatter(
								x=train.index,
								y=train["Close"],
								#mode='markers'
							),
                            go.Scatter(
								x=valid.index,
								y=valid["Close"],
								#mode='markers'
							),
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								#mode='markers'
							),

						],
						"layout":go.Layout(
							title='Combined Plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					},
                    className="card",
				)				
			])        		


        ]),
        dcc.Tab(label='DataBase Verification',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
                dcc.Dropdown(id='my-dropdown',
                            options=[{'label':x,'value':x} for x in set_df],
                             multi=True,value=['ADANIPORTS'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow',className="card",),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label':x,'value':x} for x in set_df],
                             multi=True,value=['ADANIPORTS'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume',className="card",)
            ], className="container"),
        ])


    ])
    ],
    className="tabs1")
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    #dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=dfs[dfs["Symbol"] == stock]["Date"],
                     y=dfs[dfs["Symbol"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {stock}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=dfs[dfs["Symbol"] == stock]["Date"],
                     y=dfs[dfs["Symbol"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {stock}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(i) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 1, 'label': '1Y',
                                                       'step': 'year', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    #dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=dfs[dfs["Symbol"] == stock]["Date"],
                     y=dfs[dfs["Symbol"] == stock]["Volume"],
                     hovertemplate= "Stock: $%{y:.2f}",
                     mode='lines', opacity=0.7,
                     name=f'Volume {stock}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(i) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                       {'count': 1, 'label': '1Y',
                                                       'step': 'year', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure


@app.callback(
    [Output("price-chart", "figure"), Output("volume-chart", "figure")],
    [
        Input("region-filter", "value"),
        
    ],
)
def update_charts(region):
    mask = (
        (dfs.Symbol == region)
    )
    filtered_data = dfs.loc[mask, :]
    print(filtered_data)
    df_nse=filtered_data
    """new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    dataset=new_data.values

    train=dataset[:,:]
    valid=pd. date_range(start=start_date,end=end_date)
    valid["Date"]=pd. date_range(start=start_date,end=end_date)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    # Training Data
    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    # Loading Model 
    model=load_model("saved_lstm_model"+region+".h5")

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    # Testing Phase
    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data
    valid['Predictions']=closing_price"""
    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Close"],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Predicted Close Price for future interval",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "$", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    volume_chart_figure = {
        "data": [
            {
                "x": filtered_data["Date"],
                "y": filtered_data["Close"],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {"text": "Predicted Close Price for past interval", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure, volume_chart_figure



if __name__=='__main__':
	app.run_server(debug=True)