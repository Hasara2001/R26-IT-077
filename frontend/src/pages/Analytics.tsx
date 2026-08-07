import React from 'react';
import { PieChart as PieChartIcon, TrendingUp } from 'lucide-react';

const Analytics = () => {
  return (
    <div style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', fontFamily: 'sans-serif', color: '#334155' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.875rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>
          <PieChartIcon size={32} color="#3b82f6" />
          System Analytics
        </h1>
        <p style={{ color: '#64748b', marginTop: '0.5rem' }}>Aggregated performance and demographic insights of the AI model.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
         {[
             { label: 'Total Assessments Processed', value: '1,248', trend: '+12% this month' },
             { label: 'High Risk Detections', value: '312', color: '#ef4444', trend: '25% of total' },
             { label: 'Model Accuracy', value: '94.2%', color: '#10b981', trend: '+1.2% improvement' }
         ].map((stat, i) => (
             <div key={i} style={{ background: '#fff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '1.5rem', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                 <p style={{ color: '#64748b', margin: '0 0 0.5rem 0', fontWeight: 600, fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</p>
                 <h3 style={{ fontSize: '2.5rem', margin: '0 0 0.5rem 0', color: stat.color || '#0f172a', fontWeight: 700 }}>{stat.value}</h3>
                 <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#10b981', fontSize: '0.875rem', fontWeight: 500 }}>
                    <TrendingUp size={14} /> {stat.trend}
                 </span>
             </div>
         ))}
      </div>
      
      <div style={{ background: '#fff', borderRadius: '0.75rem', border: '1px solid #e2e8f0', padding: '2rem', height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
          Detailed demographic charts will be rendered here.
      </div>
    </div>
  );
};

export default Analytics;
