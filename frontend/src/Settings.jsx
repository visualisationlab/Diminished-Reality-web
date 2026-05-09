import { ALL_PRODUCTS, SCHIJF_VAN_VIJF } from './constants';
import './Settings.css';

export default function Settings({
  isOpen, onClose,
  effect, setEffect,
  outlineMode, setOutlineMode,
  outlineColor, setOutlineColor,
  classOverrides, setClassOverrides,
}) {
  const toggle = (product) => {
    const current = classOverrides[product] ?? SCHIJF_VAN_VIJF[product];
    setClassOverrides({ ...classOverrides, [product]: !current });
  };

  return (
    <div className={`settings-panel ${isOpen ? 'open' : ''}`}>
      <div className="settings-header">
        <h1>Settings</h1>
        <button onClick={onClose} className="close-btn" aria-label="Close">&times;</button>
      </div>

      <div className="settings-body">
        <div className="setting-row">
          <label>Diminish effect</label>
          <select value={effect} onChange={e => setEffect(Number(e.target.value))}>
            <option value={0}>None</option>
            <option value={1}>Blur</option>
            <option value={2}>Overlay</option>
            <option value={3}>Desaturate</option>
          </select>
        </div>

        <div className="setting-row">
          <label>Outline</label>
          <select value={outlineMode} onChange={e => setOutlineMode(Number(e.target.value))}>
            <option value={0}>Off</option>
            <option value={1}>Healthy only</option>
            <option value={2}>All</option>
          </select>
        </div>

        <div className="setting-row">
          <label>Outline color*</label>
          <select value={outlineColor} onChange={e => setOutlineColor(e.target.value)}>
            <option value="health_based">Health based</option>
            <option value="gray">Gray</option>
            <option value="#22c55e">Green</option>
            <option value="#ef4444">Red</option>
          </select>
        </div>

        <div className="setting-section">
          <h3>Product overrides</h3>
          <p className="setting-hint">Checked = healthy (in Schijf van Vijf)</p>
          <div className="product-list">
            {ALL_PRODUCTS.map(p => {
              const healthy = classOverrides[p] ?? SCHIJF_VAN_VIJF[p];
              return (
                <label key={p} className="product-item">
                  <input type="checkbox" checked={healthy} onChange={() => toggle(p)} />
                  <span>{p}</span>
                </label>
              );
            })}
          </div>
        </div>
      </div>

      <div className="settings-footer">
        <small>*Only applies when outline is enabled</small>
      </div>
    </div>
  );
}
