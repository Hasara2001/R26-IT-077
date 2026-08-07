import React from 'react';
import { PieChart as PieChartIcon, TrendingUp } from 'lucide-react';

const Analytics = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
          <PieChartIcon size={32} color="var(--primary-accent)" />
          System Analytics
        </h1>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '1.05rem' }}>Aggregated performance and demographic insights of the AI model.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
         {[
             { label: 'Total Assessments Processed', value: '1,248', trend: '+12% this month' },
             { label: 'High Risk Detections', value: '312', color: 'var(--danger)', trend: '25% of total' },
             { label: 'Model Accuracy', value: '94.2%', color: 'var(--success)', trend: '+1.2% improvement' }
         ].map((stat, i) => (
             <div key={i} className="card" style={{ padding: '1.75rem' }}>
                 <p style={{ color: 'var(--text-secondary)', margin: '0 0 0.5rem 0', fontWeight: 600, fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</p>
                 <h3 style={{ fontSize: '2.5rem', margin: '0 0 0.5rem 0', color: stat.color || 'var(--text-primary)', fontWeight: 800 }}>{stat.value}</h3>
                 <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: 'var(--success)', fontSize: '0.875rem', fontWeight: 600 }}>
                    <TrendingUp size={16} /> {stat.trend}
                 </span>
             </div>
         ))}
      </div>
      
      <div className="card" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
          <PieChartIcon size={48} opacity={0.2} style={{ marginRight: '1rem' }} />
          Detailed demographic charts will be rendered here.
      </div>
    </div>
  );
};

export default Analytics;
