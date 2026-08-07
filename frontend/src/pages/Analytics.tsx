import React from 'react';
import { PieChart as PieChartIcon, TrendingUp, Activity } from 'lucide-react';

const Analytics = () => {
  return (
    <div style={{ padding: '3rem 2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ marginBottom: '3rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '2.2rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <PieChartIcon size={36} color="var(--primary-accent)" style={{ filter: 'drop-shadow(0 0 10px var(--accent-glow))' }} />
          System Analytics Core
        </h1>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.75rem', fontSize: '1.15rem' }}>Real-time aggregated performance and demographic telemetry of the AI model.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2.5rem', marginBottom: '2.5rem' }}>
         {[
             { label: 'Inferences Processed', value: '1,248', trend: '+12.4% this month' },
             { label: 'High Risk Detections', value: '312', color: 'var(--danger)', trend: '25% baseline' },
             { label: 'Model Confidence', value: '94.2%', color: 'var(--success)', trend: '+1.2% over epoch' }
         ].map((stat, i) => (
             <div key={i} className="card glass-panel" style={{ padding: '2rem', borderRadius: '1.25rem' }}>
                 <p style={{ color: 'var(--text-secondary)', margin: '0 0 1rem 0', fontWeight: 700, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{stat.label}</p>
                 <h3 style={{ fontSize: '3rem', margin: '0 0 1rem 0', color: stat.color || 'var(--text-primary)', fontWeight: 800, textShadow: stat.color ? `0 0 20px ${stat.color}40` : 'none' }}>{stat.value}</h3>
                 <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600, background: 'var(--input-bg)', padding: '0.5rem 1rem', borderRadius: '99px', width: 'fit-content', border: '1px solid var(--border-color)' }}>
                    <TrendingUp size={16} color="var(--success)" /> {stat.trend}
                 </span>
             </div>
         ))}
      </div>
      
      <div className="card glass-panel" style={{ height: '500px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', borderRadius: '1.5rem' }}>
          <Activity size={64} opacity={0.2} style={{ marginBottom: '1rem', color: 'var(--primary-accent)' }} />
          <p style={{ fontSize: '1.1rem', fontWeight: 500 }}>Demographic visualization matrix loading...</p>
      </div>
    </div>
  );
};

export default Analytics;
