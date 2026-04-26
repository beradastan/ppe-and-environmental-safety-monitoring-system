import { io } from 'socket.io-client'

// Vite proxy: /socket.io → localhost:5050
const socket = io('/', {
  transports: ['websocket', 'polling'],
  autoConnect: true,
})

export default socket
