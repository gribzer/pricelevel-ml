import { useRef, useState } from 'react'
import Chart from './Chart'

function App() {
  const [symbol, setSymbol] = useState('BTCUSDT')
  const [interval, setInterval] = useState('1')
  const [limit, setLimit] = useState('10')
  const chartRef = useRef(null)

  const loadData = async () => {
    const url = `http://127.0.0.1:5000/api/history?symbol=${symbol}&interval=${interval}&limit=${limit}`
    try {
      const resp = await fetch(url)
      const data = await resp.json()
      if (data.candles && chartRef.current) {
        chartRef.current.setData(data.candles)
      }
    } catch (err) {
      console.error("Error fetching data:", err)
    }
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1>Bybit Chart</h1>
      <div style={{ display: 'flex', gap: '10px' }}>
        <select value={symbol} onChange={e => setSymbol(e.target.value)}>
          <option value="BTCUSDT">BTCUSDT</option>
          <option value="ETHUSDT">ETHUSDT</option>
        </select>
        <select value={interval} onChange={e => setInterval(e.target.value)}>
          <option value="1">1m</option>
          <option value="15">15m</option>
          <option value="60">1h</option>
        </select>
        <input
          type="number"
          min="1"
          max="1000"
          value={limit}
          onChange={e => setLimit(e.target.value)}
        />
        <button onClick={loadData}>Load</button>
      </div>

      <Chart ref={chartRef} />
    </div>
  )
}

export default App
