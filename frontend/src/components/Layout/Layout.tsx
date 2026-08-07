import React, { useState, useEffect } from 'react';
import { Outlet, NavLink, Link } from 'react-router-dom';
import { Activity, LayoutDashboard, Users, PieChart, Settings, User, Moon, Sun } from 'lucide-react';
import './Layout.css';

const Layout = () => {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    // Check local storage or system preference
    const savedTheme = localStorage.getItem('hepatoai-theme');
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.setAttribute('data-theme', savedTheme);
    } else {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (prefersDark) {
        setTheme('dark');
        document.documentElement.setAttribute('data-theme', 'dark');
      }
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('hepatoai-theme', newTheme);
  };

  return (
    <div className="layout-container">
      <header className="header">
        <div className="header-content">
          <Link to="/" className="logo-section">
            <Activity size={30} color="#3b82f6" />
            <span className="logo-text">HepatoAI</span>
          </Link>
          
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <LayoutDashboard size={18} />
              Dashboard
            </NavLink>
            <NavLink to="/patients" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <Users size={18} />
              History
            </NavLink>
            <NavLink to="/analytics" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <PieChart size={18} />
              Analytics
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <Settings size={18} />
              Settings
            </NavLink>
          </nav>

          <div className="header-actions">
            <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle Theme">
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>
            <div className="user-profile">
              <div className="avatar">
                <User size={20} />
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content animate-fade-up">
        <Outlet />
      </main>

      <footer className="footer">
        <div className="footer-content">
          <p className="footer-disclaimer">
            <strong>Clinical Disclaimer:</strong> HepatoAI is a predictive support tool designed for clinical professionals. It does not substitute professional medical judgment, diagnosis, or treatment. Always verify insights with clinical evidence before making medical decisions.
          </p>
          <p className="footer-copyright">
            &copy; {new Date().getFullYear()} HepatoAI Enterprise. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
