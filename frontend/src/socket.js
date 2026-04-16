import { io } from 'socket.io-client'

// Vite proxy üzerinden bağlan (dev modunda /socket.io → localhost:5050)
const socket = io('http://localhost:5050', {
  transports: ['websocket', 'polling'],
  autoConnect: true,
})

export default socket
