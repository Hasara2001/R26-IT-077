import React from 'react';
import { Settings as SettingsIcon, Shield, Bell, Database, Lock } from 'lucide-react';

const Settings = () => {
  return (
    <div style={{ padding: '3rem 2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ marginBottom: '3rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '2.2rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <SettingsIcon size={36} color="var(--primary-accent)" style={{ filter: 'drop-shadow(0 0 10px var(--accent-glow))' }} />
          Platform Configurations
        </h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '3rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ padding: '1rem 1.25rem', background: 'var(--primary-accent)', color: '#fff', borderRadius: '0.75rem', fontWeight: 600, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', boxShadow: '0 8px 20px var(--accent-glow)' }}><Shield size={20}/> Security & Privacy</div>
              <div style={{ padding: '1rem 1.25rem', color: 'var(--text-secondary)', fontWeight: 600, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', transition: 'all 0.2s', borderRadius: '0.75rem' }}><Database size={20}/> API Infrastructure</div>
              <div style={{ padding: '1rem 1.25rem', color: 'var(--text-secondary)', fontWeight: 600, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', transition: 'all 0.2s', borderRadius: '0.75rem' }}><Bell size={20}/> Alerts & Telemetry</div>
          </div>
          
          <div className="card glass-panel" style={{ borderRadius: '1.5rem', padding: '3rem' }}>
              <h2 style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '1.8rem', fontWeight: 800 }}>Security Preferences</h2>
              <p style={{ color: 'var(--text-secondary)', margin: '0 0 2rem 0', fontSize: '1.1rem' }}>Manage your clinical authentication and data protocols.</p>
              
              <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '2rem 0' }} />
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
                  <div>
                      <h4 style={{ margin: '0 0 0.75rem 0', color: 'var(--text-primary)', fontSize: '1.25rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Lock size={18} color="var(--primary-accent)" /> Two-Factor Authentication (2FA)</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: '1.6' }}>Enforce cryptographic hardware keys or authenticator apps for all clinical access.</p>
                      <button style={{ marginTop: '1.5rem', background: 'var(--text-primary)', color: 'var(--bg-main)', border: 'none', padding: '0.8rem 1.5rem', borderRadius: '99px', cursor: 'pointer', fontWeight: 700, transition: 'all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)' }}>Enable 2FA Enforcement</button>
                  </div>
                  
                  <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)' }} />
                  
                  <div>
                      <h4 style={{ margin: '0 0 0.75rem 0', color: 'var(--text-primary)', fontSize: '1.25rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Shield size={18} color="var(--primary-accent)" /> Data Anonymization (HIPAA)</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: '1.6' }}>Ensure absolute patient PII obfuscation before payload transmission to the AI inference engine.</p>
                      <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', background: 'var(--input-bg)', padding: '1rem', borderRadius: '0.75rem', border: '1px solid var(--border-color)', width: 'fit-content' }}>
                          <input type="checkbox" id="anon" defaultChecked style={{ width: '22px', height: '22px', cursor: 'pointer', accentColor: 'var(--primary-accent)' }} />
                          <label htmlFor="anon" style={{ cursor: 'pointer', fontWeight: 600, color: 'var(--text-primary)', fontSize: '1.05rem' }}>Enforce Automatic PII Masking</label>
                      </div>
                  </div>
              </div>
          </div>
      </div>
    </div>
  );
};

export default Settings;
