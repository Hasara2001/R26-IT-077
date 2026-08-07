import React from 'react';
import { Users, Search, Filter } from 'lucide-react';

const PatientHistory = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', fontFamily: 'sans-serif', color: '#334155' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>
          <Users size={32} color="#3b82f6" />
          Patient History
        </h1>
        <div style={{ display: 'flex', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', background: '#fff', border: '1px solid #e2e8f0', padding: '0.5rem 1rem', borderRadius: '0.5rem', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                <Search size={18} color="#94a3b8" style={{ marginRight: '0.5rem' }} />
                <input type="text" placeholder="Search Patient ID..." style={{ border: 'none', outline: 'none', width: '200px' }} />
            </div>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#fff', border: '1px solid #e2e8f0', padding: '0.5rem 1rem', borderRadius: '0.5rem', cursor: 'pointer', fontWeight: 500, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                <Filter size={18} />
                Filters
            </button>
        </div>
      </div>

      <div style={{ background: '#fff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '2rem', textAlign: 'center', color: '#64748b', minHeight: '500px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
         <Users size={64} opacity={0.2} color="#3b82f6" />
         <h2 style={{ color: '#1e293b', margin: 0 }}>No Patient Records Found</h2>
         <p style={{ maxWidth: '400px', margin: 0 }}>Connect to the central clinical database to view historical patient risk assessments and medical history.</p>
         <button style={{ marginTop: '1rem', background: '#3b82f6', color: '#fff', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '0.5rem', fontWeight: 600, cursor: 'pointer' }}>Connect Database</button>
      </div>
    </div>
  );
};

export default PatientHistory;
