import React from 'react';
import { User, Mail, Building, Phone, BadgeCheck, Clock, ShieldCheck, LogOut } from 'lucide-react';
import { Link } from 'react-router-dom';

const Profile = () => {
    return (
        <div style={{ padding: '3rem 2rem', maxWidth: '1400px', margin: '0 auto', color: 'var(--text-primary)' }}>
            <div style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '2.2rem', fontWeight: 800, margin: 0, letterSpacing: '-0.025em' }}>
                    <User size={36} color="var(--primary-accent)" style={{ filter: 'drop-shadow(0 0 10px var(--accent-glow))' }} />
                    Clinician Profile
                </h1>
                <Link to="/login" style={{ textDecoration: 'none' }}>
                    <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--danger)', color: '#fff', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '99px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s', boxShadow: '0 4px 15px rgba(239, 68, 68, 0.2)' }}>
                        <LogOut size={18} /> Logout
                    </button>
                </Link>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '3rem' }}>
                {/* Left Column: ID Card */}
                <div className="card glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', padding: '3rem 2rem', borderRadius: '1.5rem', height: 'fit-content' }}>
                    <div style={{ width: '130px', height: '130px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--primary-accent), #3b82f6)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.5rem', boxShadow: '0 10px 30px var(--accent-glow)' }}>
                        <User size={64} color="#fff" />
                    </div>
                    <h2 style={{ margin: '0 0 0.5rem 0', fontSize: '1.8rem', fontWeight: 800 }}>Dr. Sarah Jenkins</h2>
                    <p style={{ margin: '0 0 1.5rem 0', color: 'var(--text-secondary)', fontSize: '1.1rem', fontWeight: 500 }}>Lead Oncologist</p>
                    
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--success)', padding: '0.6rem 1.25rem', borderRadius: '99px', fontSize: '0.9rem', fontWeight: 700, border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                        <BadgeCheck size={18} /> Verified Medical Professional
                    </span>

                    <hr style={{ width: '100%', border: 'none', borderTop: '1px solid var(--border-color)', margin: '2.5rem 0' }} />

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', width: '100%', textAlign: 'left' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
                            <Mail size={20} color="var(--primary-accent)" /> 
                            <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '1.05rem' }}>s.jenkins@hepatoai.med</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
                            <Phone size={20} color="var(--primary-accent)" /> 
                            <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '1.05rem' }}>+1 (555) 019-2834</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
                            <Building size={20} color="var(--primary-accent)" /> 
                            <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '1.05rem' }}>Central Oncology Wing, Dept 4</span>
                        </div>
                    </div>
                </div>

                {/* Right Column: Activity and Clearance */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                    <div className="card glass-panel" style={{ padding: '2.5rem', borderRadius: '1.5rem' }}>
                        <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <ShieldCheck size={26} color="var(--primary-accent)" /> System Clearance
                        </h3>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                            <div style={{ background: 'var(--input-bg)', padding: '1.5rem', borderRadius: '1rem', border: '1px solid var(--border-color)', transition: 'transform 0.2s', cursor: 'pointer' }} onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                                <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Access Level</p>
                                <h4 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-primary)', fontWeight: 800 }}>Level 4 (Attending)</h4>
                            </div>
                            <div style={{ background: 'var(--input-bg)', padding: '1.5rem', borderRadius: '1rem', border: '1px solid var(--border-color)', transition: 'transform 0.2s', cursor: 'pointer' }} onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
                                <p style={{ margin: '0 0 0.5rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>API Quota</p>
                                <h4 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-primary)', fontWeight: 800 }}>Unlimited / Priority</h4>
                            </div>
                        </div>
                    </div>

                    <div className="card glass-panel" style={{ padding: '2.5rem', borderRadius: '1.5rem' }}>
                        <h3 style={{ margin: '0 0 1.5rem 0', fontSize: '1.5rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <Clock size={26} color="var(--primary-accent)" /> Recent Inference Logs
                        </h3>
                        
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            {[
                                { id: 'PAT-8834', risk: 'High', time: '10 mins ago' },
                                { id: 'PAT-1029', risk: 'Low', time: '2 hours ago' },
                                { id: 'PAT-4492', risk: 'Low', time: 'Yesterday' },
                            ].map((log, i) => (
                                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.25rem 1.5rem', background: 'var(--input-bg)', border: '1px solid var(--border-color)', borderRadius: '1rem', transition: 'all 0.2s' }} onMouseOver={(e) => e.currentTarget.style.borderColor = 'var(--primary-accent)'} onMouseOut={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                        <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: log.risk === 'High' ? 'var(--danger)' : 'var(--success)', boxShadow: `0 0 10px ${log.risk === 'High' ? 'var(--danger)' : 'var(--success)'}` }}></div>
                                        <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1.1rem' }}>{log.id}</span>
                                    </div>
                                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', fontWeight: 600 }}>{log.time}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Profile;
