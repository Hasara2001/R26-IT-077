import React from 'react';
import { Settings as SettingsIcon, Shield, Bell, Database } from 'lucide-react';

const Settings = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', fontFamily: 'sans-serif', color: '#334155' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>
          <SettingsIcon size={32} color="#3b82f6" />
          Platform Settings
        </h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '2rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ padding: '0.75rem 1rem', background: '#eff6ff', color: '#2563eb', borderRadius: '0.5rem', fontWeight: 600, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer' }}><Shield size={18}/> Security & Privacy</div>
              <div style={{ padding: '0.75rem 1rem', color: '#64748b', fontWeight: 500, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer' }}><Database size={18}/> API Configuration</div>
              <div style={{ padding: '0.75rem 1rem', color: '#64748b', fontWeight: 500, display: 'flex', gap: '0.75rem', alignItems: 'center', cursor: 'pointer' }}><Bell size={18}/> Notifications</div>
          </div>
          
          <div style={{ background: '#fff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '2.5rem', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
              <h2 style={{ marginTop: 0, color: '#0f172a' }}>Security Preferences</h2>
              <hr style={{ border: 'none', borderTop: '1px solid #e2e8f0', margin: '1.5rem 0' }} />
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                  <div>
                      <h4 style={{ margin: '0 0 0.5rem 0', color: '#1e293b', fontSize: '1.1rem' }}>Two-Factor Authentication (2FA)</h4>
                      <p style={{ margin: 0, color: '#64748b', fontSize: '0.95rem' }}>Add an extra layer of security to your clinical account.</p>
                      <button style={{ marginTop: '1rem', background: '#0f172a', color: '#fff', border: 'none', padding: '0.6rem 1.2rem', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 600 }}>Enable 2FA</button>
                  </div>
                  
                  <hr style={{ border: 'none', borderTop: '1px solid #f1f5f9' }} />
                  
                  <div>
                      <h4 style={{ margin: '0 0 0.5rem 0', color: '#1e293b', fontSize: '1.1rem' }}>Data Anonymization (HIPAA Compliance)</h4>
                      <p style={{ margin: 0, color: '#64748b', fontSize: '0.95rem' }}>Ensure patient PII is masked before API transmission to the AI inference engine.</p>
                      <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <input type="checkbox" id="anon" defaultChecked style={{ width: '18px', height: '18px', cursor: 'pointer' }} />
                          <label htmlFor="anon" style={{ cursor: 'pointer', fontWeight: 500 }}>Enable automatic PII masking</label>
                      </div>
                  </div>
              </div>
          </div>
      </div>
    </div>
  );
};

export default Settings;
