import { useEffect, useState } from 'react'
import { useValue } from 'tldraw'

import { t } from '../i18n'
import { addInstance, instancesAtom, instanceState, instanceStatusAtom, isOff, isStarting, loadInstances, removeInstance, startInstance, updateInstance, waitOnline } from '../instances'
import { Icons } from './icons'
import { Modal } from './Modal'

export function InstancesDialog({ onClose }: { onClose: () => void }) {
  const instances = useValue(instancesAtom)
  const statuses = useValue(instanceStatusAtom)
  const [label, setLabel] = useState('')
  const [host, setHost] = useState('')
  const [port, setPort] = useState(8188)
  const [ctlUrl, setCtlUrl] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { loadInstances().catch((e) => setError(String(e))) }, [])

  // while a machine is booting, poll its status (every ~10 s) until ComfyUI answers
  const booting = instances.filter((i) => i.enabled && isStarting(statuses[i.id])).map((i) => i.id).join(',')
  useEffect(() => {
    booting.split(',').filter(Boolean).forEach((id) => waitOnline(id).catch((e) => setError(e instanceof Error ? e.message : String(e))))
  }, [booting])

  const run = (task: Promise<unknown>) => task.then(() => setError('')).catch((e) => setError(e instanceof Error ? e.message : String(e)))
  const add = () => run(addInstance({ label, host, port, ...(ctlUrl.trim() ? { ctl_url: ctlUrl.trim() } : {}) }).then(() => { setLabel(''); setHost(''); setCtlUrl('') }))
  const stop = (e: React.KeyboardEvent) => { e.stopPropagation(); if (e.key === 'Enter' && host.trim()) void add() }

  return (
    <Modal title={t('instances.title')} className="ca-rename ca-instances" onClose={onClose}>
      <div className="ca-rename__body">
        <div className="ca-chat__hint">{t('instances.hint')}</div>
        {instances.map((inst) => (
          <div key={inst.id} className="ca-instances__row">
            <div className="ca-instances__name">
              <span>{inst.label}</span>
              <span className="ca-chat__hint">{inst.host}:{inst.port} · {inst.enabled ? instanceState(statuses[inst.id]) : t('instances.disabled')}</span>
            </div>
            {inst.enabled && inst.ctl_url && isOff(statuses[inst.id]) && (
              <button type="button" className="ca-panel__secondary" onClick={() => void run(startInstance(inst.id))}>{t('instances.start')}</button>
            )}
            <label className="ca-instances__toggle">
              <input type="checkbox" checked={inst.enabled} onChange={(e) => void run(updateInstance(inst.id, { enabled: e.target.checked }))} />
              {t('instances.enabled')}
            </label>
            <button type="button" className="ca-panel__icon" title={t('common.delete')} onClick={() => void run(removeInstance(inst.id))}>{Icons.trash}</button>
          </div>
        ))}
        <div className="ca-instances__add">
          <input className="ca-panel__input" value={label} placeholder={t('instances.name')} onChange={(e) => setLabel(e.target.value)} onKeyDown={stop} />
          <input className="ca-panel__input" value={host} placeholder="192.168.0.140" onChange={(e) => setHost(e.target.value)} onKeyDown={stop} />
          <input className="ca-panel__input" type="number" min={1} max={65535} value={port} onChange={(e) => setPort(Number(e.target.value) || 8188)} onKeyDown={stop} />
          <input className="ca-panel__input ca-instances__ctl" value={ctlUrl} placeholder={t('instances.ctlUrl')} title={t('instances.ctlUrlHint')} onChange={(e) => setCtlUrl(e.target.value)} onKeyDown={stop} />
        </div>
        {error && <div className="ca-instances__error">{error}</div>}
        <div className="ca-panel__foot">
          <button type="button" className="ca-panel__secondary" onClick={() => void run(loadInstances())}>{t('instances.refresh')}</button>
          <button type="button" className="ca-panel__primary" disabled={!host.trim()} onClick={() => void add()}>{t('instances.add')}</button>
        </div>
      </div>
    </Modal>
  )
}
