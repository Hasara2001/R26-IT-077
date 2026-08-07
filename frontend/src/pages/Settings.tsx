import React from 'react';
import { Settings as SettingsIcon, Shield, Bell, Database } from 'lucide-react';

const Settings = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <SettingsIcon size={32} color="var(--primary-accent)" />
          Platform Settings
        </h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '2rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ padding: '0.75rem 1rem', background: 'var(--nav-hover)', color: 'var(--primary-accent)', borderRadius: '0.5rem', fontWeight: 600, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', boxShadow: 'inset 2px 0 0 var(--primary-accent)' }}><Shield size={18}/> Security & Privacy</div>
              <div style={{ padding: '0.75rem 1rem', color: 'var(--text-secondary)', fontWeight: 500, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', transition: 'all 0.2s' }}><Database size={18}/> API Configuration</div>
              <div style={{ padding: '0.75rem 1rem', color: 'var(--text-secondary)', fontWeight: 500, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer', transition: 'all 0.2s' }}><Bell size={18}/> Notifications</div>
          </div>
          
          <div className="card">
              <h2 style={{ marginTop: 0, color: 'var(--text-primary)', fontSize: '1.5rem', fontWeight: 700 }}>Security Preferences</h2>
              <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '1.5rem 0' }} />
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                  <div>
                      <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '1.1rem', fontWeight: 600 }}>Two-Factor Authentication (2FA)</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.5' }}>Add an extra layer of security to your clinical account.</p>
                      <button style={{ marginTop: '1.25rem', background: 'var(--text-primary)', color: 'var(--bg-card)', border: 'none', padding: '0.6rem 1.25rem', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 600, transition: 'transform 0.2s' }}>Enable 2FA</button>
                  </div>
                  
                  <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)' }} />
                  
                  <div>
                      <h4 style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)', fontSize: '1.1rem', fontWeight: 600 }}>Data Anonymization (HIPAA Compliance)</h4>
                      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.5' }}>Ensure patient PII is masked before API transmission to the AI inference engine.</p>
                      <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <input type="checkbox" id="anon" defaultChecked style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: 'var(--primary-accent)' }} />
                          <label htmlFor="anon" style={{ cursor: 'pointer', fontWeight: 500, color: 'var(--text-primary)' }}>Enable automatic PII masking</label>
                      </div>
                  </div>
              </div>
          </div>
      </div>
    </div>
  );
};

export default Settings;
