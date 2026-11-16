import { atom } from 'tldraw'

import { api, LAN_ID, setLan, setLanGuard } from './comfy'
import { t } from './i18n'
import { reportSuccess } from './ui/ErrorToasts'

export interface Instance { id: string; label: string; host: string; port: number; enabled: boolean; ctl_url?: string }
export type Power = 'OFF' | 'SUBMITTED' | 'PENDING' | 'RUNNABLE' | 'STARTING' | 'RUNNING' | 'UNKNOWN'
export interface InstanceStatus { id: string; online: boolean; queue?: number; models?: string[]; error?: string; power?: Power; ready?: boolean; ctl_error?: string }
export type InstanceFields = Partial<Omit<Instance, 'id'>>

export const instancesAtom = atom<Instance[]>('lan instances', [])
export const instanceStatusAtom = atom<Record<string, InstanceStatus>>('lan instance status', {})

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await api(path, init)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error?.message ?? body.error ?? `${path} failed: ${res.status}`)
  return body as T
}

const send = (body: unknown, method: string): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export async function refreshInstanceStatus() {
  const { statuses } = await json<{ statuses: InstanceStatus[] }>('/comfyart/instances/status')
  instanceStatusAtom.set(Object.fromEntries(statuses.map((s) => [s.id, s])))
}

export async function loadInstances() {
  const { instances } = await json<{ instances: Instance[] }>('/comfyart/instances')
  setLan(instances.filter((i) => i.enabled).map((i) => i.id))
  instancesAtom.set(instances)
  await refreshInstanceStatus()
}

export async function addInstance(fields: InstanceFields) {
  await json('/comfyart/instances', send(fields, 'POST'))
  await loadInstances()
}

export async function updateInstance(id: string, fields: InstanceFields) {
  await json(`/comfyart/instances/${id}`, send(fields, 'PATCH'))
  await loadInstances()
}

export async function removeInstance(id: string) {
  await json(`/comfyart/instances/${id}`, { method: 'DELETE' })
  await loadInstances()
}

export type TargetOption = [id: string, label: string, hint?: string]

const BOOTING: Power[] = ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']
export const isOff = (status: InstanceStatus | undefined) => !!status && !status.online && status.power === 'OFF'
// RUNNING without online yet means the machine is up but ComfyUI is still loading
export const isStarting = (status: InstanceStatus | undefined) => !!status && !status.online && !!status.power && BOOTING.includes(status.power)

export function instanceState(status: InstanceStatus | undefined): string {
  if (!status) return t('instances.disabled')
  if (status.online) return t('instances.online', status.queue ?? 0)
  return isOff(status) ? t('instances.off') : isStarting(status) ? t('instances.starting') : t('instances.offline')
}

const WAKE_POLL_MS = 10_000
const WAKE_TIMEOUT_MS = 20 * 60_000
const waiting = new Map<string, Promise<void>>()
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// Polls the status until the instance answers; one poller per instance, shared by every caller.
export function waitOnline(id: string): Promise<void> {
  const known = waiting.get(id)
  if (known) return known
  const task = (async () => {
    const until = Date.now() + WAKE_TIMEOUT_MS
    let lost = 0
    while (Date.now() < until) {
      await refreshInstanceStatus().catch(() => undefined)
      const status = instanceStatusAtom.get()[id]
      if (status?.online) return
      // the control endpoint may blip; give up only after three misses in a row
      lost = !status || status.power === 'UNKNOWN' ? lost + 1 : 0
      if (lost >= 3) throw new Error(t('instances.wakeLost', status?.ctl_error ?? status?.error ?? ''))
      await sleep(WAKE_POLL_MS)
    }
    throw new Error(t('instances.wakeTimeout'))
  })().finally(() => waiting.delete(id))
  waiting.set(id, task)
  return task
}

export async function startInstance(id: string) {
  await json(`/comfyart/instances/${id}/start`, { method: 'POST' })
  await refreshInstanceStatus()
}

// Before a job goes to an instance that is switched off: power it on, tell the user, and wait until ComfyUI answers.
async function wakeBeforeDispatch(target: string) {
  const inst = instancesAtom.get().find((i) => i.id === target)
  if (!inst?.ctl_url) return
  await refreshInstanceStatus().catch(() => undefined)
  const status = instanceStatusAtom.get()[target]
  if (!status || status.online) return
  // UNKNOWN (control endpoint not answering): do not guess, let the server report "cannot reach"
  if (!isOff(status) && !isStarting(status)) return
  if (isOff(status)) await startInstance(target)
  reportSuccess(t('instances.waking', inst.label))
  await waitOnline(target)
}

setLanGuard(wakeBeforeDispatch)

export function targetOptions(instances: Instance[], statuses: Record<string, InstanceStatus>): TargetOption[] {
  return [
    ['auto', t('runOn.auto')],
    ['local', t('settings.backendLocal')],
    ...instances.filter((i) => i.enabled).map((i): TargetOption => [i.id, i.label, instanceState(statuses[i.id])]),
  ]
}

export function lanHas(files: string[], runOn: string): boolean {
  const statuses = Object.values(instanceStatusAtom.get()).filter((s) => s.online && (runOn === 'auto' || s.id === runOn))
  return files.length > 0 && statuses.some((s) => files.every((f) => s.models?.includes(f)))
}

export function runsOn(jobId: string): string {
  if (!jobId.startsWith(LAN_ID)) return t('settings.backendLocal')
  const id = jobId.slice(LAN_ID.length).split(':')[0]
  return instancesAtom.get().find((i) => i.id === id)?.label ?? id
}
