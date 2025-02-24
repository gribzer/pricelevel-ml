import React, { forwardRef, useEffect, useRef, useImperativeHandle } from 'react'
import { createChart } from 'lightweight-charts'

const ChartComponent = forwardRef((props, ref) => {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)

  useEffect(() => {
    const chart = createChart(containerRef.current, {
      width: 600,
      height: 400,
      layout: { backgroundColor: '#ffffff', textColor: '#000000' },
    })

    // В v5:
    // addSeries(<SeriesType>, options?)
    // SeriesType может быть "Line", "Area", "Bar", "Candlestick", "Baseline", "Histogram"
    const candleSeries = chart.addSeries('Candlestick')
    candleSeriesRef.current = candleSeries

    chartRef.current = chart

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
      style={{ width: '600px', height: '400px', border: '1px solid #ccc' }}
    />
  )
})

ChartComponent.displayName = 'ChartComponent'
export default ChartComponent
