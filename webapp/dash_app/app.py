# webapp/app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)
app.title = "Realtime Candles"

app.layout = html.Div([
    html.H3("Realtime 1H Candles"),
    dcc.Dropdown(
        id="symbol-dropdown",
        options=[
            {"label": "BTCUSDT", "value": "BTCUSDT"},
            {"label": "ETHUSDT", "value": "ETHUSDT"}
        ],
        value="BTCUSDT"
    ),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=5_000, n_intervals=0)
])

@app.callback(
    Output("live-graph", "figure"),
    [Input("interval-component", "n_intervals"),
     Input("symbol-dropdown", "value")]
)
def update_graph(n, symbol):
    from run import global_aggregators
    agg = global_aggregators.get(symbol)
    if not agg:
        return go.Figure()

    candles = agg.candles[-50:]
    if not candles:
        return go.Figure()

    x = [c["start"] for c in candles]
    open_  = [c["open"]  for c in candles]
    high_  = [c["high"]  for c in candles]
    low_   = [c["low"]   for c in candles]
    close_ = [c["close"] for c in candles]

    fig = go.Figure(data=[
        go.Candlestick(
            x=x, open=open_, high=high_, low=low_, close=close_,
            name=f"{symbol} 1H"
        )
    ])
    fig.update_layout(
        title=f"{symbol} - Realtime Candles",
        xaxis_rangeslider_visible=False
    )
    return fig

if __name__=="__main__":
    app.run_server(debug=True)
