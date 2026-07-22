import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const webStorageFlag = '--no-experimental-webstorage'
const existingNodeOptions = process.env.NODE_OPTIONS?.trim()
const nodeOptions = process.allowedNodeEnvironmentFlags.has(webStorageFlag)
  ? [existingNodeOptions, webStorageFlag].filter(Boolean).join(' ')
  : existingNodeOptions
const vitestPath = fileURLToPath(new URL('../../../node_modules/vitest/vitest.mjs', import.meta.url))
const child = spawn(process.execPath, [vitestPath, 'run', ...process.argv.slice(2)], {
  env: {
    ...process.env,
    ...(nodeOptions ? { NODE_OPTIONS: nodeOptions } : {})
  },
  stdio: 'inherit'
})

const signalHandlers = new Map(['SIGINT', 'SIGTERM'].map(signal => [signal, () => child.kill(signal)]))

for (const [signal, handler] of signalHandlers) {
  process.on(signal, handler)
}

const result = await new Promise((resolve, reject) => {
  child.once('error', reject)
  child.once('close', (status, signal) => resolve({ status, signal }))
}).finally(() => {
  for (const [signal, handler] of signalHandlers) {
    process.off(signal, handler)
  }
})

if (result.signal) {
  process.kill(process.pid, result.signal)
} else {
  process.exitCode = result.status ?? 1
}
