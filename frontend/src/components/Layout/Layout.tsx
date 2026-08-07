import React from 'react';
import { Outlet, NavLink, Link } from 'react-router-dom';
import { Activity, LayoutDashboard, Users, PieChart, Settings, User } from 'lucide-react';
import './Layout.css';

const Layout = () => {
  return (
    <div className="layout-container">
      <header className="header">
        <div className="header-content">
          <Link to="/" className="logo-section">
            <Activity size={30} color="#2563eb" />
            <span className="logo-text">HepatoAI</span>
          </Link>
          
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <LayoutDashboard size={18} />
              Dashboard
            </NavLink>
            <NavLink to="/patients" className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}>
              <Users size={18} />
              Patient History
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

          <div className="user-profile">
            <div className="avatar">
              <User size={20} />
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
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
