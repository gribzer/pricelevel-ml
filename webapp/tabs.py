# webapp/tabs.py
from dash import dcc

def get_tabs():
    instruments = ["BTCUSDT", "ETHUSDT"]
    tabs = []
    for inst in instruments:
        tabs.append(dcc.Tab(label=inst, value=inst))
    return tabs
