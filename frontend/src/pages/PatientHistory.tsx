import React from 'react';
import { Users, Search, Filter } from 'lucide-react';

const PatientHistory = () => {
  return (
    <div style={{ padding: '3rem 2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '2.2rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <Users size={36} color="var(--primary-accent)" style={{ filter: 'drop-shadow(0 0 10px var(--accent-glow))' }} />
          Patient Database
        </h1>
        <div style={{ display: 'flex', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', background: 'var(--input-bg)', border: '1px solid var(--glass-border)', padding: '0.75rem 1.25rem', borderRadius: '99px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', backdropFilter: 'blur(10px)' }}>
                <Search size={20} color="var(--text-secondary)" style={{ marginRight: '0.5rem' }} />
                <input type="text" placeholder="Search Patient ID..." style={{ border: 'none', outline: 'none', width: '220px', background: 'transparent', color: 'var(--text-primary)', fontSize: '1rem', fontWeight: 500 }} />
            </div>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--glass-bg)', color: 'var(--text-primary)', border: '1px solid var(--glass-border)', padding: '0.75rem 1.25rem', borderRadius: '99px', cursor: 'pointer', fontWeight: 700, boxShadow: '0 4px 15px rgba(0,0,0,0.05)', transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)', backdropFilter: 'blur(10px)' }}>
                <Filter size={18} />
                Filters
            </button>
        </div>
      </div>

      <div className="card glass-panel" style={{ textAlign: 'center', minHeight: '600px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem', borderRadius: '1.5rem' }}>
         <Users size={80} opacity={0.15} color="var(--primary-accent)" />
         <h2 style={{ color: 'var(--text-primary)', margin: 0, fontSize: '2rem', fontWeight: 800 }}>No Patient Records Indexed</h2>
         <p style={{ maxWidth: '450px', margin: 0, color: 'var(--text-secondary)', lineHeight: '1.6', fontSize: '1.1rem' }}>Establish connection to the central clinical data lake to stream historical patient risk assessments.</p>
         <button style={{ marginTop: '1.5rem', background: 'linear-gradient(135deg, var(--primary-accent), #3b82f6)', color: '#fff', border: 'none', padding: '1rem 2rem', borderRadius: '99px', fontWeight: 700, fontSize: '1.1rem', cursor: 'pointer', boxShadow: '0 8px 25px var(--accent-glow)', transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)' }}>Establish Secure Connection</button>
      </div>
    </div>
  );
};

export default PatientHistory;
