import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: '32px',
          color: '#ff8080',
          background: '#1a0a0a',
          fontFamily: 'monospace',
          whiteSpace: 'pre-wrap',
          height: '100%',
          overflow: 'auto',
        }}>
          <strong>Render hatası:</strong>{'\n'}
          {this.state.error.message}{'\n\n'}
          {this.state.error.stack}
        </div>
      )
    }
    return this.props.children
  }
}
