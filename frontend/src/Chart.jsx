import { forwardRef, useEffect, useRef, useImperativeHandle } from 'react'
import { createChart } from 'lightweight-charts'

const Chart = forwardRef((props, ref) => {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)

  useEffect(() => {
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: {
        backgroundColor: '#fff',
        textColor: '#000'
      },
      grid: {
        vertLines: { color: '#eee' },
        horzLines: { color: '#eee' }
      },
    })
    const series = chart.addCandlestickSeries()
    chartRef.current = chart
    candleSeriesRef.current = series

    return () => {
      chart.remove()
    }
  }, [])

  useImperativeHandle(ref, () => ({
    setData(candles) {
      if (candleSeriesRef.current) {
        candleSeriesRef.current.setData(candles)
      }
    },
  }))

  return (
    <div
      ref={containerRef}
      style={{ width: '800px', height: '400px', border: '1px solid #ccc' }}
    />
  )
})

Chart.displayName = 'Chart'
export default Chart
