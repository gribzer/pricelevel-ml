import React, { useState, useRef } from 'react'
import ChartComponent from './Chart'

function App() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [timeframe, setTimeframe] = useState('1H') // "D","4H","1H","15M"
  const chartRef = useRef(null)

  // Наборы для <select>
  const symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
  const timeframes = ["D", "4H", "1H", "15M"]

  const handleLoad = async () => {
    try {
      // Просто делаем fetch на /api/history (через Vite proxy)
      // /api/history?symbol=BTCUSDT&timeframe=4H
      const url = `/api/history?symbol=${symbol}&timeframe=${timeframe}`
      const resp = await fetch(url)
      const data = await resp.json()
      if (!data.candles) {
        console.error("No candles in response", data)
        return
      }
      // chartRef.current -> ChartComponent
      chartRef.current.setData(data.candles)
    } catch (err) {
      console.error("Error loading", err)
    }
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1>Bybit Candles (no realtime)</h1>
      <div style={{ display: 'flex', gap: '10px' }}>
        <div>
          Symbol:
          <select value={symbol} onChange={e => setSymbol(e.target.value)}>
            {symbols.map(sym => <option key={sym} value={sym}>{sym}</option>)}
          </select>
        </div>
        <div>
          Timeframe:
          <select value={timeframe} onChange={e => setTimeframe(e.target.value)}>
            {timeframes.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
        <button onClick={handleLoad}>Load</button>
      </div>

      <ChartComponent ref={chartRef} />
    </div>
  )
}

export default App
