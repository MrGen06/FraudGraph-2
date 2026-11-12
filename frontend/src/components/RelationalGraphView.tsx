import React, { useEffect, useRef, useState } from 'react'
import { fetchGraphSubgraph } from '../api'
import type { GraphExplanation, GraphNode } from '../types'

interface Props {
  initialExplanation?: GraphExplanation | null
  focusedTransactionId?: string
}

export const RelationalGraphView: React.FC<Props> = ({
  initialExplanation,
  focusedTransactionId,
}) => {
  const [graphData, setGraphData] = useState<GraphExplanation | null>(initialExplanation || null)
  const [inputTxId, setInputTxId] = useState<string>(focusedTransactionId || '')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [errorNotice, setErrorNotice] = useState<string | null>(null)

  // Zoom & Pan transformation state
  const [zoom, setZoom] = useState<number>(1)
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState<boolean>(false)
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  // Node position drag state
  const [dragNodeId, setDragNodeId] = useState<string | null>(null)
  const [customPositions, setCustomPositions] = useState<Record<string, { x: number; y: number }>>({})
  const svgRef = useRef<SVGSVGElement | null>(null)

  const width = 800
  const height = 480
  const centerX = width / 2
  const centerY = height / 2

  // Update when prop changes
  useEffect(() => {
    if (focusedTransactionId) {
      setInputTxId(focusedTransactionId)
      loadSubgraph(focusedTransactionId)
    } else if (initialExplanation) {
      setGraphData(initialExplanation)
      setCustomPositions({})
    }
  }, [focusedTransactionId, initialExplanation])

  const loadSubgraph = async (txId: string) => {
    if (!txId.trim()) return
    setIsLoading(true)
    setErrorNotice(null)
    setSelectedNode(null)
    setCustomPositions({})
    try {
      const data = await fetchGraphSubgraph(txId.trim())
      setGraphData(data)
    } catch (err: any) {
      setErrorNotice(err.message || `No graph records found for ${txId}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputTxId.trim()) {
      loadSubgraph(inputTxId.trim())
    }
  }

  // Calculate default node layout positions
  const nodes = graphData?.nodes || []
  const links = graphData?.links || []

  const centerNode = nodes.find((n) => n.is_center) || nodes[0]
  const entityNodes = nodes.filter((n) => n.id !== centerNode?.id)

  const defaultPositions: Record<string, { x: number; y: number }> = {}
  if (centerNode) {
    defaultPositions[centerNode.id] = { x: centerX, y: centerY }
  }

  const radius = 150
  entityNodes.forEach((node, idx) => {
    const angle = (idx * 2 * Math.PI) / Math.max(1, entityNodes.length) - Math.PI / 2
    defaultPositions[node.id] = {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    }
  })

  // Merged positions
  const positions: Record<string, { x: number; y: number }> = {
    ...defaultPositions,
    ...customPositions,
  }

  const getNodeColor = (node: GraphNode) => {
    if (node.risk_flag) return '#EF4444'
    if (node.is_center) return '#3B82F6'
    switch (node.type) {
      case 'card':
        return '#8B5CF6'
      case 'device':
        return '#06B6D4'
      case 'email':
        return '#10B981'
      case 'account':
        return '#F59E0B'
      case 'merchant':
        return '#EC4899'
      default:
        return '#64748B'
    }
  }

  // Interactive mouse handlers for dragging nodes and panning canvas
  const handleMouseDownSvg = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).tagName === 'svg') {
      setIsPanning(true)
      setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }
  }

  const handleMouseMoveSvg = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan({ x: e.clientX - startPan.x, y: e.clientY - startPan.y })
    } else if (dragNodeId && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect()
      const mouseX = (e.clientX - rect.left - pan.x) / zoom
      const mouseY = (e.clientY - rect.top - pan.y) / zoom
      setCustomPositions((prev) => ({
        ...prev,
        [dragNodeId]: { x: mouseX, y: mouseY },
      }))
    }
  }

  const handleMouseUpSvg = () => {
    setIsPanning(false)
    setDragNodeId(null)
  }

  const resetViewport = () => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
    setCustomPositions({})
  }

  return (
    <div className="panel interpretability-card">
      <div className="panel-header">
        <div className="header-title-group">
          <span className="icon-badge">🕸️</span>
          <div>
            <h3>Interactive Relational Graph Explorer</h3>
            <p className="subtitle">
              Cytoscape-grade canvas: inspect multi-hop entity hops, card sharing, and device clusters
            </p>
          </div>
        </div>

        {/* Search for transaction graph */}
        <form onSubmit={handleSearchSubmit} className="graph-search-form">
          <input
            type="text"
            placeholder="Search Transaction ID..."
            value={inputTxId}
            onChange={(e) => setInputTxId(e.target.value)}
            className="input-search-sm"
          />
          <button type="submit" className="btn-tiny btn-accent">
            Fetch Graph
          </button>
        </form>
      </div>

      {/* Graph Toolbar */}
      <div className="graph-toolbar">
        <div className="legend-items">
          <span className="legend-item"><span className="color-dot dot-tx" /> Transaction</span>
          <span className="legend-item"><span className="color-dot dot-card" /> Card</span>
          <span className="legend-item"><span className="color-dot dot-device" /> Device</span>
          <span className="legend-item"><span className="color-dot dot-account" /> Account</span>
          <span className="legend-item"><span className="color-dot dot-email" /> Domain</span>
          <span className="legend-item"><span className="color-dot dot-fraud" /> Flagged</span>
        </div>

        <div className="canvas-controls">
          <button type="button" className="ctrl-btn" onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}>+</button>
          <button type="button" className="ctrl-btn" onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}>−</button>
          <button type="button" className="ctrl-btn" onClick={resetViewport}>Reset View</button>
        </div>
      </div>

      {errorNotice && <div className="error-notice">{errorNotice}</div>}

      <div className="graph-container">
        {isLoading ? (
          <div className="loading-state">Querying graph topology from database...</div>
        ) : nodes.length === 0 ? (
          <div className="empty-notice">No relational graph nodes found. Enter a transaction ID above.</div>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${width} ${height}`}
            className="graph-canvas"
            onMouseDown={handleMouseDownSvg}
            onMouseMove={handleMouseMoveSvg}
            onMouseUp={handleMouseUpSvg}
            style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
          >
            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="22"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#94A3B8" />
              </marker>
            </defs>

            {/* Transform group for Zoom and Pan */}
            <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
              {/* Directed Edges */}
              {links.map((link, idx) => {
                const src = positions[link.source]
                const dst = positions[link.target]
                if (!src || !dst) return null

                return (
                  <g key={`link-${idx}`}>
                    <line
                      x1={src.x}
                      y1={src.y}
                      x2={dst.x}
                      y2={dst.y}
                      stroke="#475569"
                      strokeWidth="2"
                      markerEnd="url(#arrow)"
                      strokeDasharray={link.relationship.includes('DEVICE') ? '4 2' : 'none'}
                    />
                    <text
                      x={(src.x + dst.x) / 2}
                      y={(src.y + dst.y) / 2 - 6}
                      fill="#94A3B8"
                      fontSize="9"
                      textAnchor="middle"
                      className="edge-label"
                    >
                      {link.relationship}
                    </text>
                  </g>
                )
              })}

              {/* Node Groups */}
              {nodes.map((node) => {
                const pos = positions[node.id] || { x: centerX, y: centerY }
                const radius = node.is_center ? 26 : node.risk_flag ? 20 : 16
                const isSelected = selectedNode?.id === node.id

                return (
                  <g
                    key={node.id}
                    className="graph-node-group"
                    onMouseDown={(e) => {
                      e.stopPropagation()
                      setDragNodeId(node.id)
                      setSelectedNode(node)
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    {node.risk_flag && (
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={radius + 8}
                        fill="none"
                        stroke="#EF4444"
                        strokeWidth="2"
                        strokeDasharray="4 2"
                        className="pulsing-halo"
                      />
                    )}

                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={radius}
                      fill={getNodeColor(node)}
                      stroke={isSelected ? '#F8FAFC' : '#1E293B'}
                      strokeWidth={isSelected ? '3' : '2'}
                      filter="drop-shadow(0 2px 5px rgba(0,0,0,0.6))"
                    />

                    <text
                      x={pos.x}
                      y={pos.y + radius + 14}
                      textAnchor="middle"
                      fill="#E2E8F0"
                      fontSize="10"
                      fontWeight={node.is_center ? 'bold' : 'normal'}
                      className="node-text"
                    >
                      {node.label.length > 20 ? `${node.label.slice(0, 18)}…` : node.label}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}

        {/* Selected Entity Inspector Panel */}
        {selectedNode && (
          <div className="node-detail-card">
            <div className="node-detail-title">
              <span>Entity Inspector: {selectedNode.label}</span>
              <button
                type="button"
                className="close-detail-btn"
                onClick={() => setSelectedNode(null)}
              >
                ✕
              </button>
            </div>
            <div className="node-detail-grid">
              <div>Type: <strong>{selectedNode.type.toUpperCase()}</strong></div>
              <div>ID: <strong>{selectedNode.id}</strong></div>
              <div>
                Status:{' '}
                <strong className={selectedNode.risk_flag ? 'text-danger' : 'text-success'}>
                  {selectedNode.risk_flag ? '🚨 Flagged High Risk' : '✓ Normal Behavior'}
                </strong>
              </div>
              <div>
                Central Node:{' '}
                <strong>{selectedNode.is_center ? 'Primary Transaction' : 'Connected Neighbor'}</strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {graphData?.reasoning && (
        <div className="graph-footer-summary">
          <p className="graph-reasoning-text">
            <strong>Topology Insight:</strong> {graphData.reasoning}
          </p>
        </div>
      )}
    </div>
  )
}
