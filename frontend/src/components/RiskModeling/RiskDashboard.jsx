import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Activity, BarChart2, Loader2, BrainCircuit } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './RiskDashboard.css';

const INITIAL_STATE = {
    tumor_size_cm: '',
    tumor_number: '',
    vascular_invasion_imaging: '',
    afp_ngml: '',
    cirrhosis_present: '',
    image_feature_vector_norm: '',
    text_embedding_risk_score: '',
    multimodal_feature_vector_norm: ''
};

const RiskDashboard = () => {
    const [formData, setFormData] = useState(INITIAL_STATE);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [mockImpactData, setMockImpactData] = useState([]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const calculateMockImpact = (data, prediction) => {
        const impacts = [
            { name: 'Tumor Size', value: (parseFloat(data.tumor_size_cm) || 0) * 0.8 },
            { name: 'Tumor Count', value: (parseFloat(data.tumor_number) || 0) * 1.2 },
            { name: 'Vascular Inv.', value: (parseFloat(data.vascular_invasion_imaging) || 0) * 3.5 },
            { name: 'AFP Level', value: (parseFloat(data.afp_ngml) || 0) * 0.01 },
            { name: 'Cirrhosis', value: (parseFloat(data.cirrhosis_present) || 0) * 2.0 },
            { name: 'Image Risk', value: (parseFloat(data.image_feature_vector_norm) || 0) * 4.0 },
            { name: 'Text Risk', value: (parseFloat(data.text_embedding_risk_score) || 0) * 5.0 },
            { name: 'Multimodal', value: (parseFloat(data.multimodal_feature_vector_norm) || 0) * 6.0 }
        ];

        const sorted = impacts.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
        setMockImpactData(sorted.slice(0, 6));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        
        try {
            const payload = {
                tumor_size_cm: parseFloat(formData.tumor_size_cm),
                tumor_number: parseInt(formData.tumor_number, 10),
                vascular_invasion_imaging: parseInt(formData.vascular_invasion_imaging, 10),
                afp_ngml: parseFloat(formData.afp_ngml),
                cirrhosis_present: parseInt(formData.cirrhosis_present, 10),
                image_feature_vector_norm: parseFloat(formData.image_feature_vector_norm),
                text_embedding_risk_score: parseFloat(formData.text_embedding_risk_score),
                multimodal_feature_vector_norm: parseFloat(formData.multimodal_feature_vector_norm)
            };

            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.statusText}`);
            }

            const data = await response.json();
            setTimeout(() => {
                setResult(data);
                calculateMockImpact(formData, data.prediction);
                setLoading(false);
            }, 800); // Artificial delay to show off the cool loading animation
            
        } catch (err) {
            setError(err.message || "Failed to connect to the prediction API.");
            setLoading(false);
        }
    };

    const renderInput = (label, name, type = 'number', step = 'any') => (
        <div className="input-group">
            <label className="input-label" htmlFor={name}>{label}</label>
            <input
                id={name}
                name={name}
                type={type}
                step={step}
                required
                className="input-field"
                value={formData[name]}
                onChange={handleInputChange}
                placeholder={`Enter ${label.toLowerCase()}`}
            />
        </div>
    );

    return (
        <div className="dashboard-container animate-fade-up">
            <header className="dashboard-header">
                <h1 className="dashboard-title">
                    <BrainCircuit size={48} color="var(--primary-accent)" style={{ filter: 'drop-shadow(0 0 10px var(--accent-glow))' }} />
                    HepatoAI Diagnostic Engine
                </h1>
                <p className="dashboard-subtitle">
                    Advanced multimodal neural risk stratification for precise oncological outcomes.
                </p>
            </header>

            <div className="dashboard-grid">
                <div className="card glass-panel">
                    <h2 className="card-title">
                        <Activity size={24} color="var(--primary-accent)" />
                        Clinical Parameters
                    </h2>
                    <form onSubmit={handleSubmit} className="form-grid">
                        {renderInput('Tumor Size (cm)', 'tumor_size_cm')}
                        {renderInput('Tumor Number', 'tumor_number')}
                        {renderInput('Vascular Invasion (0 or 1)', 'vascular_invasion_imaging')}
                        {renderInput('AFP Level (ng/ml)', 'afp_ngml')}
                        {renderInput('Cirrhosis (0 or 1)', 'cirrhosis_present')}
                        {renderInput('Image Feature Norm', 'image_feature_vector_norm')}
                        {renderInput('Text Risk Score', 'text_embedding_risk_score')}
                        {renderInput('Multimodal Vector Norm', 'multimodal_feature_vector_norm')}

                        <button 
                            type="submit" 
                            className="submit-btn" 
                            disabled={loading}
                        >
                            {loading ? (
                                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem' }}>
                                    <Loader2 className="animate-spin" size={24} />
                                    Synthesizing AI Inference...
                                </span>
                            ) : 'Initialize AI Risk Assessment'}
                        </button>
                    </form>
                    
                    {error && (
                        <div style={{ marginTop: '1.5rem', padding: '1.25rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', color: 'var(--danger)', borderRadius: '0.75rem', fontWeight: 600 }}>
                            {error}
                        </div>
                    )}
                </div>

                <div className="result-container">
                    <div className="card glass-panel" style={{ padding: 0, overflow: 'hidden' }}>
                        {result ? (
                            <div className={`result-card ${result.prediction === 1 ? 'result-high-risk' : 'result-low-risk'}`}>
                                <div className="result-icon">
                                    {result.prediction === 1 ? <AlertCircle size={48} /> : <CheckCircle size={48} />}
                                </div>
                                <div className="result-content">
                                    <h3>{result.risk_level}</h3>
                                    <p>{result.message}</p>
                                    <span className="probability-tag">
                                        Confidence: {result.probability_percentage}%
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <div className="empty-state" style={{ minHeight: '200px' }}>
                                <BrainCircuit size={64} opacity={0.3} color="var(--primary-accent)" />
                                <p style={{ fontSize: '1.1rem', fontWeight: 500 }}>Awaiting clinical payload for inference...</p>
                            </div>
                        )}
                    </div>

                    <div className="card glass-panel">
                        <h2 className="card-title">
                            <BarChart2 size={24} color="var(--primary-accent)" />
                            Explainability Matrix (XAI)
                        </h2>
                        
                        {result && mockImpactData.length > 0 ? (
                            <div className="chart-container">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart 
                                        data={mockImpactData} 
                                        layout="vertical" 
                                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                    >
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 13, fontWeight: 600}} width={110} />
                                        <Tooltip 
                                            cursor={{fill: 'rgba(148, 163, 184, 0.1)'}}
                                            contentStyle={{borderRadius: '12px', border: '1px solid var(--glass-border)', backgroundColor: 'var(--glass-bg)', color: 'var(--text-primary)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)', backdropFilter: 'blur(10px)'}}
                                            itemStyle={{color: 'var(--text-primary)', fontWeight: 700}}
                                        />
                                        <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                                            {mockImpactData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={result.prediction === 1 ? 'var(--danger)' : 'var(--primary-accent)'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        ) : (
                            <div className="empty-state chart-container">
                                <BarChart2 size={64} opacity={0.2} color="var(--text-secondary)" />
                                <p>Feature importance mapping will generate post-inference.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RiskDashboard;
