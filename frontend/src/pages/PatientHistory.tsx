import React from 'react';
import { Users, Search, Filter } from 'lucide-react';

const PatientHistory = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <Users size={32} color="var(--primary-accent)" />
          Patient History
        </h1>
        <div style={{ display: 'flex', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', background: 'var(--input-bg)', border: '1px solid var(--border-color)', padding: '0.6rem 1rem', borderRadius: '0.5rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)', transition: 'all 0.2s' }}>
                <Search size={18} color="var(--text-secondary)" style={{ marginRight: '0.5rem' }} />
                <input type="text" placeholder="Search Patient ID..." style={{ border: 'none', outline: 'none', width: '200px', background: 'transparent', color: 'var(--text-primary)' }} />
            </div>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--input-bg)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', padding: '0.6rem 1rem', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 600, boxShadow: '0 2px 4px rgba(0,0,0,0.05)', transition: 'all 0.2s' }}>
                <Filter size={18} />
                Filters
            </button>
        </div>
      </div>

      <div className="card" style={{ textAlign: 'center', minHeight: '500px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
         <Users size={64} opacity={0.2} color="var(--primary-accent)" />
         <h2 style={{ color: 'var(--text-primary)', margin: 0, fontSize: '1.5rem' }}>No Patient Records Found</h2>
         <p style={{ maxWidth: '400px', margin: 0, color: 'var(--text-secondary)', lineHeight: '1.6' }}>Connect to the central clinical database to view historical patient risk assessments and medical history.</p>
         <button style={{ marginTop: '1.5rem', background: 'var(--primary-accent)', color: '#fff', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer', boxShadow: '0 4px 12px var(--accent-glow)', transition: 'all 0.2s' }}>Connect Database</button>
      </div>
    </div>
  );
};

export default PatientHistory;
