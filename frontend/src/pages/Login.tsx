import React, { useState } from 'react';
import { Mail, Lock, ArrowRight, BrainCircuit } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Login = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);

    const handleLogin = (e) => {
        e.preventDefault();
        setLoading(true);
        setTimeout(() => {
            setLoading(false);
            navigate('/');
        }, 1500);
    };

    return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', zIndex: 10 }}>
            <div className="card glass-panel animate-fade-up" style={{ maxWidth: '480px', width: '100%', padding: '3.5rem', textAlign: 'center', borderRadius: '1.5rem', border: '1px solid var(--glass-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
                    <div style={{ background: 'var(--glass-bg)', padding: '1.25rem', borderRadius: '1rem', border: '1px solid var(--glass-border)', boxShadow: '0 0 25px var(--accent-glow)' }}>
                        <BrainCircuit size={48} color="var(--primary-accent)" />
                    </div>
                </div>
                
                <h1 style={{ margin: '0 0 0.5rem 0', fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.025em' }}>HepatoAI Access</h1>
                <p style={{ margin: '0 0 2.5rem 0', color: 'var(--text-secondary)', fontSize: '1.1rem' }}>Secure clinician portal authentication.</p>

                <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                    <div style={{ position: 'relative' }}>
                        <Mail size={20} color="var(--text-secondary)" style={{ position: 'absolute', left: '1.25rem', top: '50%', transform: 'translateY(-50%)' }} />
                        <input type="email" placeholder="Clinical Email Address" required style={{ width: '100%', padding: '1.1rem 1rem 1.1rem 3.5rem', background: 'var(--input-bg)', border: '1px solid var(--border-color)', borderRadius: '0.75rem', color: 'var(--text-primary)', outline: 'none', fontSize: '1rem', boxSizing: 'border-box' }} className="input-field" />
                    </div>
                    
                    <div style={{ position: 'relative' }}>
                        <Lock size={20} color="var(--text-secondary)" style={{ position: 'absolute', left: '1.25rem', top: '50%', transform: 'translateY(-50%)' }} />
                        <input type="password" placeholder="Password" required style={{ width: '100%', padding: '1.1rem 1rem 1.1rem 3.5rem', background: 'var(--input-bg)', border: '1px solid var(--border-color)', borderRadius: '0.75rem', color: 'var(--text-primary)', outline: 'none', fontSize: '1rem', boxSizing: 'border-box' }} className="input-field" />
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.95rem', color: 'var(--text-secondary)' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontWeight: 500 }}>
                            <input type="checkbox" style={{ accentColor: 'var(--primary-accent)', width: '18px', height: '18px' }} /> Remember me
                        </label>
                        <span style={{ color: 'var(--primary-accent)', cursor: 'pointer', fontWeight: 600 }}>Forgot Password?</span>
                    </div>

                    <button type="submit" disabled={loading} style={{ background: 'linear-gradient(135deg, var(--primary-accent), #3b82f6)', color: '#fff', padding: '1.125rem', borderRadius: '0.75rem', border: 'none', fontSize: '1.15rem', fontWeight: 800, cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', boxShadow: '0 8px 25px var(--accent-glow)', transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)', marginTop: '1.5rem', letterSpacing: '0.05em' }} onMouseOver={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 12px 30px var(--accent-glow)'; }} onMouseOut={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 8px 25px var(--accent-glow)'; }}>
                        {loading ? 'Authenticating...' : 'Secure Login'}
                        {!loading && <ArrowRight size={22} />}
                    </button>
                </form>

                <p style={{ marginTop: '3rem', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                    Protected by Enterprise SSO and AES-256 Encryption.
                </p>
            </div>
        </div>
    );
};

export default Login;
